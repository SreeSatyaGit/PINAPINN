import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
import csv
import json
from collections import defaultdict
from pina import Condition, LabelTensor, Trainer
from pina.solver import PINN
from pina.problem import AbstractProblem
from pina.model import FeedForward
from pina.equation import Equation
from pina.callback import MetricTracker
from pina.optim import TorchOptimizer
from pina.loss import ScalarWeighting
from pina.operator import grad as pina_grad

from data_utils import prepare_training_tensors, get_collocation_points, SPECIES_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _time_bucket(time_hours: float) -> str:
    return "early_0_8h" if time_hours <= 8.0 else "late_24_48h"


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, t: np.ndarray, conditions: np.ndarray):
    rows = []
    grouped = defaultdict(list)
    for i in range(len(t)):
        grouped[(conditions[i], _time_bucket(float(t[i])))].append(i)

    for (condition, bucket), idxs in grouped.items():
        yt = y_true[idxs]
        yp = y_pred[idxs]
        for s_idx, species in enumerate(SPECIES_ORDER):
            yts = yt[:, s_idx]
            yps = yp[:, s_idx]
            abs_err = np.abs(yps - yts)
            sq_err = (yps - yts) ** 2
            denom = np.clip(np.abs(yts), 1e-6, None)
            mape = np.mean(abs_err / denom) * 100.0
            rows.append({
                "condition": condition,
                "time_bucket": bucket,
                "species": species,
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(sq_err))),
                "mape_percent": float(mape),
                "n": int(len(idxs)),
            })
    return rows


def save_metrics_csv(rows, path: str) -> None:
    fieldnames = ["condition", "time_bucket", "species", "mae", "rmse", "mape_percent", "n"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ── Initial guesses for learnable kinetic parameters ──────────────────────
INITIAL_K_PARAMS = {
    'hill_coeff': 2.0, 'IC50_vem': 0.1, 'IC50_tram': 0.1, 'IC50_pi3k': 0.1, 'IC50_ras': 0.1,
    'k_paradox': 0.1, 'k_egfr': 0.5, 'k_egfr_deg': 0.1, 'k_her2': 0.5, 'k_her2_deg': 0.1,
    'k_her3': 0.5, 'k_her3_deg': 0.1, 'k_igf': 0.5, 'k_igf_deg': 0.1,
    'k_erk_rtk': 0.5, 'Km_rtk': 0.5, 'k_up': 0.1, 'k_erk_sos': 0.5, 'Km_sos': 0.5,
    'k_akt_rtk': 0.5, 'Km_artk': 0.5,
    'k_craf': 1.0, 'k_craf_deg': 0.2, 'k_mek': 1.0, 'k_mek_deg': 0.2, 'k_mek_braf': 0.1,
    'k_erk': 1.0, 'k_erk_deg': 0.2,
    'k_dusp_synth': 1.0, 'k_dusp_deg': 0.2, 'k_dusp_cat': 0.5,
    'Km_dusp': 0.5, 'Km_dusp_s': 0.5, 'n_dusp': 2.0,
    'k_raf_pi3k': 0.1, 'Km_raf_pi3k': 0.5,
    'k_erk_pi3k': 0.1, 'Km_erk_pi3k': 0.5,
    'k_akt': 1.0, 'k_akt_deg': 0.2,
    'k_4ebp1': 1.0, 'k_4ebp1_deg': 0.2, 'k_4ebp1_comp': 0.05, 'Km_4ebp1': 0.5,
    'k_akt_raf': 0.1, 'Km_akt_raf': 0.5,
    'k_her2_tx': 0.1, 'k_her3_tx': 0.1, 'k_ras_pi3k_frac': 0.1,
    'K_sat_egfr': 1.0, 'K_sat_her2': 1.0, 'K_sat_her3': 1.0, 'K_sat_igfr': 1.0,
    'K_sat_craf': 1.0, 'K_sat_mek': 1.0, 'K_sat_erk': 1.0, 'K_sat_akt': 1.0, 'K_sat_4ebp1': 1.0,
    'w_egfr': 0.1, 'w_her2': 0.1, 'w_her3': 0.1, 'w_igf1r': 0.1, 'w_craf': 0.1,
    'w_mek': 0.1, 'w_erk': 0.1, 'w_dusp6': 0.1, 'w_akt': 0.1, 'w_4ebp1': 0.1,
}

# ── Model ─────────────────────────────────────────────────────────────────
class SignalingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FeedForward(
            input_dimensions=5, output_dimensions=10,
            layers=[64, 64], func=nn.Tanh,
        )
        self.k_params = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=torch.float32))
            for k, v in INITIAL_K_PARAMS.items()
        })

    def forward(self, x):
        # Pass through backbone
        out = self.net(x)
        # Ensure non-negative biological concentrations
        return F.softplus(out)

# ── Problem ───────────────────────────────────────────────────────────────
class SignalingProblem(AbstractProblem):
    def __init__(self, train_data, scalers, model):
        self._output_variables = SPECIES_ORDER
        self.temporal_variable = ['t']
        self.parameters = ['vem', 'tram', 'pi3k', 'ras']
        self.scalers = scalers
        self._model = model          

        # Data condition
        t_norm = torch.tensor(train_data['t_norm'], dtype=torch.float32)
        drugs  = torch.tensor(train_data['drugs'], dtype=torch.float32)
        X_data = LabelTensor(torch.cat([t_norm, drugs], dim=1), self.input_variables)
        Y_data = LabelTensor(
            torch.tensor(train_data['y_norm'], dtype=torch.float32),
            self.output_variables,
        )

        # Collocation points for physics (Denser for better convergence)
        t_col, drugs_col = get_collocation_points(n_points=6000)
        X_phys_raw = torch.cat([t_col, drugs_col], dim=1)
        X_phys = LabelTensor(X_phys_raw, self.input_variables)
        
        # Dedicated set for steady state (strictly No-Drug points)
        ss_mask = (X_phys_raw[:, 1:].sum(dim=1) < 1e-4) # all drugs near 0
        X_ss = LabelTensor(X_phys_raw[ss_mask], self.input_variables)

        self._conditions = {
            'data':         Condition(input=X_data, target=Y_data),
            'physics':      Condition(input=X_phys, equation=Equation(self.signaling_odes)),
            'steady_state': Condition(input=X_ss, equation=Equation(self.steady_state_odes)),
        }
        super().__init__()

    @property
    def output_variables(self): return self._output_variables

    @property
    def conditions(self): return self._conditions

    # ── ODE residuals (PINA‑native, stable arithmetic) ───────────
    def signaling_odes(self, input_, output_):
        k = self._model.k_params
        eps = 1e-7

        # ── dy/dt via PINA grad ──────────────
        dy_dt_lt = pina_grad(output_, input_, components=self.output_variables, d='t')
        
        # Convert to plain tensors
        dy_dt_norm = dy_dt_lt.as_subclass(torch.Tensor)   # (N, 10)
        y_norm     = output_.as_subclass(torch.Tensor)     # (N, 10)
        inp        = input_.as_subclass(torch.Tensor)      # (N, 5)

        # Un‑normalise
        y_range = self.scalers['y_range']
        y_min   = self.scalers['y_min']
        t_range = self.scalers['t_range']

        y     = y_norm * y_range + y_min
        dy_dt = (dy_dt_norm * y_range) / t_range

        # STRICTLY CLAMP y TO PREVENT NaN FROM NEGATIVE CONCENTRATIONS
        y_safe = torch.abs(y)

        # Species
        pEGFR, HER2, HER3, IGF1R = y_safe[:, 0], y_safe[:, 1], y_safe[:, 2], y_safe[:, 3]
        pCRAF, pMEK, pERK, DUSP6 = y_safe[:, 4], y_safe[:, 5], y_safe[:, 6], y_safe[:, 7]
        pAKT, p4EBP1              = y_safe[:, 8], y_safe[:, 9]

        # Drug concentrations from input (clamped just in case)
        Vem    = torch.abs(inp[:, 1])
        Tram   = torch.abs(inp[:, 2])
        PI3Ki  = torch.abs(inp[:, 3])
        RasInh = torch.abs(inp[:, 4])

        # Hill coefficient
        n = torch.clamp(k['hill_coeff'], 1.0, 4.0)

        # Drug effects (Hill functions)
        Vem_inh   = (Vem + eps)**n   / (torch.abs(k['IC50_vem'])**n   + (Vem + eps)**n   + 1e-8)
        Tram_eff  = (Tram + eps)**n  / (torch.abs(k['IC50_tram'])**n  + (Tram + eps)**n  + 1e-8)
        PI3Ki_eff = (PI3Ki + eps)**n / (torch.abs(k['IC50_pi3k'])**n  + (PI3Ki + eps)**n + 1e-8)
        Ras_eff   = (RasInh + eps)**n/ (torch.abs(k['IC50_ras'])**n   + (RasInh+ eps)**n + 1e-8)

        # Saturation constants
        Ks_egfr  = torch.abs(k['K_sat_egfr'])
        Ks_her2  = torch.abs(k['K_sat_her2'])
        Ks_her3  = torch.abs(k['K_sat_her3'])
        Ks_igfr  = torch.abs(k['K_sat_igfr'])
        Ks_craf  = torch.abs(k['K_sat_craf'])
        Ks_mek   = torch.abs(k['K_sat_mek'])
        Ks_erk   = torch.abs(k['K_sat_erk'])
        Ks_akt   = torch.abs(k['K_sat_akt'])
        Ks_4ebp1 = torch.abs(k['K_sat_4ebp1'])

        # Feedback / crosstalk terms
        k_erk_rtk = torch.abs(k['k_erk_rtk'])
        Km_rtk    = torch.abs(k['Km_rtk'])
        ERK_feedback = k_erk_rtk * pERK / (Km_rtk + pERK + 1e-8)

        k_up = torch.abs(k['k_up'])
        drug_relief = k_up * (Vem_inh + Tram_eff + PI3Ki_eff)

        k_erk_sos = torch.abs(k['k_erk_sos'])
        k_akt_rtk = torch.abs(k['k_akt_rtk'])
        Km_sos    = torch.abs(k['Km_sos'])
        Km_artk   = torch.abs(k['Km_artk'])
        ERK_to_SOS = k_erk_sos * pERK / (Km_sos + pERK + 1e-8)
        AKT_to_RTK = k_akt_rtk * pAKT / (Km_artk + pAKT + 1e-8)

        RTK_total = pEGFR + HER2 + 1.5 * HER3 + IGF1R
        RAS_GTP   = RTK_total * (1.0 - ERK_to_SOS) * (1.0 - AKT_to_RTK) * (1.0 - Ras_eff)

        k_raf_pi3k  = torch.abs(k['k_raf_pi3k'])
        Km_raf_pi3k = torch.abs(k['Km_raf_pi3k'])
        k_erk_pi3k  = torch.abs(k['k_erk_pi3k'])
        Km_erk_pi3k = torch.abs(k['Km_erk_pi3k'])
        RAF_to_PI3K = k_raf_pi3k * pCRAF / (Km_raf_pi3k + pCRAF + 1e-8)
        ERK_to_PI3K = k_erk_pi3k * pERK  / (Km_erk_pi3k + pERK  + 1e-8)

        k_ras_frac = torch.abs(k['k_ras_pi3k_frac'])
        PI3K_input = RTK_total * (1.0 - ERK_to_PI3K) * (1.0 - k_ras_frac * Ras_eff) + RAF_to_PI3K

        k_akt_raf  = torch.abs(k['k_akt_raf'])
        Km_akt_raf = torch.abs(k['Km_akt_raf'])
        AKT_RAF_inhib = k_akt_raf * pAKT / (Km_akt_raf + pAKT + 1e-8)

        k_paradox = torch.abs(k['k_paradox'])
        pCRAF_fl  = pCRAF.clamp(min=0.05)
        pi3ki_att = 1.0 - 0.7 * PI3Ki_eff
        Vem_paradox = k_paradox * Vem * Ks_craf / (Ks_craf + pCRAF_fl + 1e-8) * pi3ki_att

        # ── Per‑species ODE residuals ─────────────────────────────────
        ke  = torch.abs(k['k_egfr']);   ked = torch.abs(k['k_egfr_deg'])
        res0 = dy_dt[:, 0] - (ke * (1.0 + drug_relief) * Ks_egfr / (Ks_egfr + pEGFR + 1e-8) - (ked + ERK_feedback) * pEGFR)

        kh2 = torch.abs(k['k_her2']);  kh2d = torch.abs(k['k_her2_deg']); kh2tx = torch.abs(k['k_her2_tx'])
        res1 = dy_dt[:, 1] - (kh2 * Ks_her2 / (Ks_her2 + HER2 + 1e-8) + kh2tx * (1.0 - pERK / (Ks_erk + pERK + 1e-8)) - kh2d * HER2)

        kh3 = torch.abs(k['k_her3']);  kh3d = torch.abs(k['k_her3_deg']); kh3tx = torch.abs(k['k_her3_tx'])
        res2 = dy_dt[:, 2] - (kh3 * Ks_her3 / (Ks_her3 + HER3 + 1e-8) + kh3tx * (1.0 - pERK / (Ks_erk + pERK + 1e-8)) - kh3d * HER3)

        ki  = torch.abs(k['k_igf']);   kid = torch.abs(k['k_igf_deg'])
        res3 = dy_dt[:, 3] - (ki * Ks_igfr / (Ks_igfr + IGF1R + 1e-8) - (kid + ERK_feedback + AKT_to_RTK) * IGF1R)

        kc  = torch.abs(k['k_craf']);  kcd = torch.abs(k['k_craf_deg'])
        res4 = dy_dt[:, 4] - (kc * RAS_GTP * Ks_craf / (Ks_craf + pCRAF + 1e-8) + Vem_paradox - (kcd + AKT_RAF_inhib) * pCRAF)

        km  = torch.abs(k['k_mek']);   kmd = torch.abs(k['k_mek_deg']); kmb = torch.abs(k['k_mek_braf'])
        braf_ic50 = 0.04
        Vem_braf = (Vem + eps)**n / (braf_ic50**n + (Vem + eps)**n + 1e-8)
        res5 = dy_dt[:, 5] - ((km * pCRAF + kmb * (1.0 - Vem_braf)) * (1.0 - Tram_eff) * Ks_mek / (Ks_mek + pMEK + 1e-8) - kmd * pMEK)

        kdusp_cat = torch.abs(k['k_dusp_cat']); Km_dusp = torch.abs(k['Km_dusp'])
        DUSP_act  = kdusp_cat * DUSP6 / (Km_dusp + DUSP6 + 1e-8)
        ker = torch.abs(k['k_erk']);   kerd = torch.abs(k['k_erk_deg'])
        res6 = dy_dt[:, 6] - (ker * pMEK * Ks_erk / (Ks_erk + pERK + 1e-8) - (kerd + DUSP_act) * pERK)

        n_dusp    = torch.clamp(k['n_dusp'], 1.5, 3.5)
        kds       = torch.abs(k['k_dusp_synth']); Km_ds = torch.abs(k['Km_dusp_s'])
        kdd       = torch.abs(k['k_dusp_deg'])
        DUSP_ind  = kds * (pERK + eps)**n_dusp / (Km_ds**n_dusp + (pERK + eps)**n_dusp + 1e-8)
        res7 = dy_dt[:, 7] - (DUSP_ind - kdd * DUSP6)

        ka  = torch.abs(k['k_akt']);   kad = torch.abs(k['k_akt_deg'])
        res8 = dy_dt[:, 8] - (ka * PI3K_input * (1.0 - PI3Ki_eff) * Ks_akt / (Ks_akt + pAKT + 1e-8) - kad * pAKT)

        k4  = torch.abs(k['k_4ebp1']); k4d = torch.abs(k['k_4ebp1_deg']); k4b = torch.abs(k['k_4ebp1_comp'])
        res9 = dy_dt[:, 9] - (k4 * pAKT * Ks_4ebp1 / (Ks_4ebp1 + p4EBP1 + 1e-8) + k4b - k4d * p4EBP1)

        w = [torch.sqrt(torch.abs(k[wn]) + 1e-8)
             for wn in ['w_egfr','w_her2','w_her3','w_igf1r','w_craf',
                        'w_mek','w_erk','w_dusp6','w_akt','w_4ebp1']]

        residuals = torch.stack([
            w[0]*res0, w[1]*res1, w[2]*res2, w[3]*res3, w[4]*res4,
            w[5]*res5, w[6]*res6, w[7]*res7, w[8]*res8, w[9]*res9,
        ], dim=1)
        
        # Replace explicit NaN or Inf with 0.0 to rescue stability if gradient spikes
        residuals = torch.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)

        out = residuals.as_subclass(LabelTensor)
        out.labels = [f"res_{s}" for s in SPECIES_ORDER]
        return out

    def steady_state_odes(self, input_, output_):
        # Input contains only no-drug points (filtered in __init__)
        dy_dt_lt = pina_grad(output_, input_, components=self.output_variables, d='t')
        dy_dt_norm = dy_dt_lt.as_subclass(torch.Tensor)
        dy_dt = (dy_dt_norm * self.scalers['y_range']) / self.scalers['t_range']
        
        # We return the raw gradient vector; PINN will compute MSE(dy_dt, 0)
        out = dy_dt.as_subclass(LabelTensor)
        out.labels = [f"ss_{s}" for s in SPECIES_ORDER]
        return torch.nan_to_num(out, nan=0.0)

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    seed = 42
    normalization_mode = "train_only"
    split_mode = "partial_condition_holdout"
    holdout_condition = "Vem + PI3Ki Combo"
    partial_condition_train_timepoints = [0.0, 1.0, 4.0]
    max_epochs = 3000
    learning_rate = 2e-4
    set_seed(seed)

    LOGGER.info("Initialising PINA Signaling Model")
    LOGGER.info("Seed=%d", seed)

    train_data, test_data, scalers = prepare_training_tensors(
        split_mode=split_mode,
        holdout_condition=holdout_condition,
        partial_condition_train_timepoints=partial_condition_train_timepoints,
        normalization_mode=normalization_mode,
    )
    LOGGER.info("Normalization mode=%s", scalers["normalization_mode"])
    LOGGER.info("Train samples=%d | Test samples=%d", len(train_data["t"]), len(test_data["t"]))

    model   = SignalingModel()
    problem = SignalingProblem(train_data, scalers, model)

    solver = PINN(
        problem=problem,
        model=model,
        optimizer=TorchOptimizer(torch.optim.Adam, lr=learning_rate),
        weighting=ScalarWeighting(weights={'data': 50.0, 'steady_state': 100.0})
    )

    trainer = Trainer(
        solver=solver,
        max_epochs=max_epochs,
        accelerator="cpu",
        callbacks=[MetricTracker()],
        enable_model_summary=False,
        gradient_clip_val=1.0,
    )

    LOGGER.info("Training for 1000 epochs")
    trainer.train()

    # ── Evaluation ────────────────────────────────────────────────────
    LOGGER.info("Evaluating on held‑out test set")
    t_test = torch.tensor(test_data['t_norm'], dtype=torch.float32)
    d_test = torch.tensor(test_data['drugs'],  dtype=torch.float32)
    X_test = LabelTensor(
        torch.cat([t_test, d_test], dim=1),
        ['t', 'vem', 'tram', 'pi3k', 'ras'],
    )

    with torch.no_grad():
        predictions = solver.forward(X_test)

    y_pred_norm = predictions.as_subclass(torch.Tensor)
    y_true_norm = torch.tensor(test_data['y_norm'], dtype=torch.float32)
    
    # Calculate MSE on normalized space
    test_mse = nn.MSELoss()(y_pred_norm, y_true_norm)
    assert not torch.isnan(test_mse), "NaN detected in normalized test loss"
    LOGGER.info("Test MSE (normalized): %.6f", test_mse.item())
    
    # Calculate MSE on actual un-normalized space
    y_range = scalers['y_range']
    y_min = scalers['y_min']
    y_pred = y_pred_norm * y_range + y_min
    y_true = y_true_norm * y_range + y_min
    
    true_mse = nn.MSELoss()(y_pred, y_true)
    assert not torch.isnan(true_mse), "NaN detected in un-normalized test loss"
    LOGGER.info("Test MSE (actual biological scale): %.6f", true_mse.item())

    LOGGER.info("Sample prediction (pERK, un-normalized): %.4f", y_pred[0, 6].item())

    metrics_rows = compute_detailed_metrics(
        y_true.detach().cpu().numpy(),
        y_pred.detach().cpu().numpy(),
        test_data["t"],
        test_data["condition"],
    )
    save_metrics_csv(metrics_rows, "detailed_metrics.csv")
    LOGGER.info("Saved detailed metrics → detailed_metrics.csv")

    run_summary = {
        "seed": seed,
        "split_mode": split_mode,
        "holdout_condition": holdout_condition,
        "partial_condition_train_timepoints": partial_condition_train_timepoints,
        "normalization_mode": normalization_mode,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "train_samples": int(len(train_data["t"])),
        "test_samples": int(len(test_data["t"])),
        "test_mse_normalized": float(test_mse.item()),
        "test_mse_biological_scale": float(true_mse.item()),
    }
    with open("run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    LOGGER.info("Saved run summary → run_summary.json")

    torch.save(model.state_dict(), "pina_signaling_model.pth")
    LOGGER.info("Saved model → pina_signaling_model.pth")
