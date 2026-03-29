import torch 
import torch .nn as nn 
import torch .nn .functional as F 
import numpy as np 
import random 
import logging 
import csv 
import json 
from pathlib import Path
from collections import defaultdict 
from pina import Condition ,LabelTensor ,Trainer 
from pina .solver import PINN 
from pina .problem import AbstractProblem 
from pina .model import FeedForward 
from pina .equation import Equation 
from pina .callback import MetricTracker 
from lightning .pytorch .callbacks import Callback 
from pina .optim import TorchOptimizer 
from pina .loss import ScalarWeighting 
from pina .operator import grad as pina_grad 

from data_utils import prepare_training_tensors ,get_collocation_points ,SPECIES_ORDER 

logging .basicConfig (level =logging .INFO ,format ="%(asctime)s | %(levelname)s | %(message)s")
LOGGER =logging .getLogger (__name__ )

# ── Condition filter ──────────────────────────────────────────────────────────
TARGET_CONDITION = "Vemurafenib Only"   # must match substring of condition labels in dataset
VEM_CONCENTRATION = 0.5                 # normalised drug concentration used in collocation
# ─────────────────────────────────────────────────────────────────────────────

def filter_to_condition(data: dict, condition_substr: str) -> dict:
    """
    Return a copy of `data` containing only rows whose condition label contains
    `condition_substr` (case-insensitive substring match).

    Filters every array/list in `data` that has the same leading dimension as
    `data["condition"]`. Raises ValueError if no rows match.
    """
    conditions = np.array(data["condition"])
    mask = np.array([condition_substr.lower() in c.lower() for c in conditions])
    if mask.sum() == 0:
        raise ValueError(
            f"No rows found matching condition '{condition_substr}'. "
            f"Available: {np.unique(conditions).tolist()}"
        )
    filtered = {}
    n_total = len(conditions)
    for key, val in data.items():
        if isinstance(val, np.ndarray) and len(val) == n_total:
            filtered[key] = val[mask]
        elif isinstance(val, list) and len(val) == n_total:
            filtered[key] = [v for v, m in zip(val, mask) if m]
        elif isinstance(val, torch.Tensor) and val.shape[0] == n_total:
            filtered[key] = val[mask]
        else:
            filtered[key] = val   # scalars, dicts, etc — pass through unchanged
    LOGGER.info(
        "Filtered to condition '%s': %d / %d rows retained",
        condition_substr, int(mask.sum()), n_total,
    )
    return filtered

def temporal_train_val_split(data: dict, val_fraction: float = 0.2) -> tuple[dict, dict]:
    """
    Split data into train and validation by time.
    The latest `val_fraction` of unique time points go to validation.
    This tests temporal extrapolation within the condition.
    """
    t_hours = np.array(data["t"]).squeeze()
    unique_times = np.sort(np.unique(t_hours))
    n_val_times = max(1, int(len(unique_times) * val_fraction))
    val_times = set(unique_times[-n_val_times:])

    val_mask  = np.array([t in val_times for t in t_hours])
    train_mask = ~val_mask

    def apply_mask(d, mask):
        out = {}
        n = len(t_hours)
        for key, val in d.items():
            if isinstance(val, np.ndarray) and len(val) == n:
                out[key] = val[mask]
            elif isinstance(val, list) and len(val) == n:
                out[key] = [v for v, m in zip(val, mask) if m]
            elif isinstance(val, torch.Tensor) and val.shape[0] == n:
                out[key] = val[mask]
            else:
                out[key] = val
        return out

    train_split = apply_mask(data, train_mask)
    val_split   = apply_mask(data, val_mask)

    LOGGER.info(
        "Temporal split — train: %d samples (t<=%.1fh) | val: %d samples (t>=%.1fh)",
        train_mask.sum(), t_hours[train_mask].max(),
        val_mask.sum(),   t_hours[val_mask].min(),
    )
    return train_split, val_split

def get_vem_collocation_points(
    n_points: int = 4000,
    vem_concentration: float = VEM_CONCENTRATION,
    late_time_extra: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate collocation points for the Vemurafenib-only condition.

    Sampling strategy (three components):
      1. Log-uniform over [0.1, 48h] — denser near t=0, covers early dynamics
      2. Uniform over [24, 48h]       — explicit late-time coverage for rebound
      3. Fixed grid at [1, 2, 4, 8, 12, 24, 36, 48h] — anchor at biological landmarks

    Drug columns: vem=vem_concentration, tram=pi3k=ras=0.
    """
    # Component 1: log-uniform (main body)
    t_log = np.random.uniform(np.log(0.1), np.log(49.0), size=(n_points,))
    t_main = np.exp(t_log)

    # Component 2: uniform late-time supplement
    t_late = np.random.uniform(24.0, 48.0, size=(late_time_extra,))

    # Component 3: fixed biological landmark times (repeated 10× each for weight)
    t_landmarks = np.repeat(
        np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0]),
        repeats=10
    )

    # Combine and clip
    t_hours = np.concatenate([t_main, t_late, t_landmarks])
    t_hours = np.clip(t_hours, 0.0, 48.0)
    total_points = len(t_hours)

    t_norm = (t_hours / 48.0).reshape(-1, 1).astype(np.float32)

    drug_col = np.zeros((total_points, 4), dtype=np.float32)
    drug_col[:, 0] = vem_concentration

    LOGGER.debug(
        "Collocation: %d total points | t<8h: %d | t>=24h: %d",
        total_points,
        int((t_hours < 8.0).sum()),
        int((t_hours >= 24.0).sum()),
    )

    return (
        torch.tensor(t_norm, dtype=torch.float32),
        torch.tensor(drug_col, dtype=torch.float32),
    )

def build_initial_condition_tensors(
    train_data: dict,
    scalers: dict,
    input_variables: list[str],
    output_variables: list[str],
    t_threshold_hours: float = 4.0,
    n_replicate: int = 128,
    vem_concentration: float = VEM_CONCENTRATION,
    species_ic_weights: dict | None = None,
) -> tuple[LabelTensor, LabelTensor]:
    """
    Build X_ic and Y_ic for pinning the model at early time points.

    Instead of averaging all early samples into one anchor, this version:
    1. Uses ALL individual early-time samples as separate targets
    2. Replicates each sample n_replicate times for gradient stability
    3. Uses the actual t_norm values from the data (not forced to 0.0)

    This is more robust when there are very few training samples.
    """
    t_hours = np.array(train_data["t"]).squeeze()
    t_norm_arr = np.array(train_data["t_norm"]).squeeze()
    early_mask = t_hours <= t_threshold_hours

    if early_mask.sum() == 0:
        raise ValueError(
            f"No training samples found at t <= {t_threshold_hours}h. "
            "Cannot build initial conditions. Try raising t_threshold_hours."
        )

    y_norm_arr = np.array(train_data["y_norm"])
    drugs_arr  = np.array(train_data["drugs"])

    y_early     = y_norm_arr[early_mask]    # shape (n_early, 10)
    t_early     = t_norm_arr[early_mask]    # shape (n_early,)
    drugs_early = drugs_arr[early_mask]     # shape (n_early, 4)
    n_early = int(early_mask.sum())

    LOGGER.info(
        "IC anchors: %d individual samples at t<=%.1fh (not averaged):",
        n_early, t_threshold_hours
    )
    for i in range(n_early):
        t_h = float(t_hours[early_mask][i])
        vals = {sp: round(float(y_early[i, j]), 3) for j, sp in enumerate(output_variables)}
        LOGGER.info("  t=%.2fh: %s", t_h, vals)

    # Build replicated inputs and targets
    # Each early sample is replicated n_replicate times
    x_rows = []
    y_rows = []
    for i in range(n_early):
        x_row = np.concatenate([[t_early[i]], drugs_early[i]])   # shape (5,)
        x_rows.append(np.repeat(x_row[np.newaxis, :], n_replicate, axis=0))
        y_rows.append(np.repeat(y_early[i:i+1, :], n_replicate, axis=0))

    x_ic = np.concatenate(x_rows, axis=0).astype(np.float32)   # (n_early*n_replicate, 5)
    y_ic = np.concatenate(y_rows, axis=0).astype(np.float32)   # (n_early*n_replicate, 10)

    LOGGER.info(
        "IC tensor: %d points (%d samples × %d replicates)",
        len(x_ic), n_early, n_replicate
    )

    # Apply per-species IC weights by scaling the targets.
    # Species with weight=0.0 are effectively excluded from the IC condition.
    # Species with weight=1.0 contribute at full strength.
    # The weight scales the target — when weight=0.0, target=0.0 regardless of
    # actual measurement. PINA MSE(pred, 0) penalises the model for predicting
    # non-zero values, which would WORSEN those species. Therefore use a small
    # positive floor (0.05) instead of 0.0 for "excluded" species to avoid
    # actively penalising non-zero predictions.
    if species_ic_weights is not None:
        weight_arr = np.array(
            [species_ic_weights.get(sp, 1.0) for sp in output_variables],
            dtype=np.float32
        )
        LOGGER.info("Applying per-species IC weights: %s",
                    dict(zip(output_variables, weight_arr.round(2).tolist())))
        # Scale targets by weights: species with low weight get pulled toward
        # a "neutral" value (their weighted target) rather than their measured value
        # Neutral = 0.5 in normalised space (mid-range), not 0.0
        neutral_norm = 0.5   # mid-point of [0, 1] normalised space
        y_ic_weighted = y_ic * weight_arr + neutral_norm * (1.0 - weight_arr)
        y_ic = y_ic_weighted
        LOGGER.info("IC targets after weighting (first replicate, normalised): %s",
                    dict(zip(output_variables, y_ic[0].round(3).tolist())))

    X_ic = LabelTensor(torch.tensor(x_ic, dtype=torch.float32), input_variables)
    Y_ic = LabelTensor(torch.tensor(y_ic, dtype=torch.float32), output_variables)

    return X_ic, Y_ic

def set_seed (seed :int )->None :
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )
    torch .backends .cudnn .deterministic =True 
    torch .backends .cudnn .benchmark =False 

def _time_bucket (time_hours :float )->str :
    return "early_0_8h"if time_hours <=8.0 else "late_24_48h"

def compute_detailed_metrics (y_true :np .ndarray ,y_pred :np .ndarray ,t :np .ndarray ,conditions :np .ndarray ):
    rows =[]
    grouped =defaultdict (list )
    for i in range (len (t )):
        grouped [(conditions [i ],_time_bucket (float (t [i ])))].append (i )

    for (condition ,bucket ),idxs in grouped .items ():
        yt =y_true [idxs ]
        yp =y_pred [idxs ]
        for s_idx ,species in enumerate (SPECIES_ORDER ):
            yts =yt [:,s_idx ]
            yps =yp [:,s_idx ]
            abs_err =np .abs (yps -yts )
            sq_err =(yps -yts )**2 
            denom =np .clip (np .abs (yts ),1e-6 ,None )
            mape =np .mean (abs_err /denom )*100.0 
            rows .append ({
            "condition":condition ,
            "time_bucket":bucket ,
            "species":species ,
            "mae":float (np .mean (abs_err )),
            "rmse":float (np .sqrt (np .mean (sq_err ))),
            "mape_percent":float (mape ),
            "n":int (len (idxs )),
            })
    return rows 

def save_metrics_csv (rows ,path :str )->None :
    fieldnames =["condition","time_bucket","species","mae","rmse","mape_percent","n"]
    with open (path ,"w",newline ="",encoding ="utf-8")as f :
        writer =csv .DictWriter (f ,fieldnames =fieldnames )
        writer .writeheader ()
        writer .writerows (rows )

class OptimizationSnapshotCallback (Callback ):
    """
    Save optimization snapshots every N epochs to support dynamic visualization.
    """
    def __init__(
        self,
        model: nn.Module,
        solver: PINN,
        problem: "SignalingProblem",
        train_data,
        scalers,
        every_n_epochs: int = 10,
        max_snapshots: int = 10,
        output_dir: str = "optimization_snapshots",
    ) -> None:
        self.model = model
        self.solver = solver
        self.problem = problem
        self.every_n_epochs = every_n_epochs
        self.max_snapshots = max_snapshots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "snapshot_history.jsonl"
        self.snapshots_written = 0

        self._y_min = scalers["y_min"]
        self._y_range = scalers["y_range"]
        t_train = torch.tensor(train_data["t_norm"], dtype=torch.float32)
        d_train = torch.tensor(train_data["drugs"], dtype=torch.float32)
        self._x_train = LabelTensor(
            torch.cat([t_train, d_train], dim=1),
            ["t", "vem", "tram", "pi3k", "ras"],
        )
        if "y_raw" in train_data:
            self._y_train = torch.tensor(train_data["y_raw"], dtype=torch.float32)
            self._y_is_raw = True
        else:
            self._y_train = torch.tensor(train_data["y_norm"], dtype=torch.float32)
            self._y_is_raw = False
            LOGGER.warning(
                "OptimizationSnapshotCallback: 'y_raw' not in train_data. "
                "Snapshot data MSE will be in normalised space."
            )

        if self.history_path.exists():
            self.history_path.unlink()

    def _effective_param(self, name: str, value: torch.Tensor) -> float:
        if name in {"hill_coeff", "n_dusp"}:
            return float(torch.clamp(value.detach(), min=1.0, max=4.0).item())
        return float(F.softplus(value.detach()).item())

    def _collect_snapshot(self, epoch: int):
        snapshot = {}
        with torch.no_grad():
            pred_norm = self.solver.forward(self._x_train).as_subclass(torch.Tensor)
            if self._y_is_raw:
                pred = pred_norm * self._y_range + self._y_min
            else:
                pred = pred_norm
            data_mse_per_species = ((pred - self._y_train) ** 2).mean(dim=0).cpu().tolist()

        phys_input = self.problem._conditions["physics"].input
        phys_input_g = phys_input.clone().detach().requires_grad_(True)
        phys_pred_g = self.solver.forward(phys_input_g)
        phys_residual = self.problem.signaling_odes(phys_input_g, phys_pred_g)
        phys_tensor = phys_residual.as_subclass(torch.Tensor)
        physics_mse_per_species = ((phys_tensor ** 2).mean(dim=0)).detach().cpu().tolist()

        # Alert on per-species ODE residuals above threshold
        CRITICAL_THRESHOLD = 0.05
        for sp_name, sp_mse in zip(SPECIES_ORDER, physics_mse_per_species):
            if sp_mse > CRITICAL_THRESHOLD:
                LOGGER.debug(
                    "Epoch %d | HIGH ODE residual: %-12s %.6f",
                    epoch, sp_name, sp_mse
                )
        snapshot["physics_mse_max_species"] = SPECIES_ORDER[
            int(np.argmax(physics_mse_per_species))
        ]
        snapshot["physics_mse_max_value"] = float(max(physics_mse_per_species))

        raw_params = {
            k: float(v.detach().cpu().item())
            for k, v in self.model.k_params.items()
        }
        effective_params = {
            k: self._effective_param(k, v)
            for k, v in self.model.k_params.items()
        }

        # MEK cascade summary for fast diagnostic scanning
        snapshot["mek_cascade_summary"] = {
            "k_mek":        self._effective_param("k_mek",      self.model.k_params["k_mek"]),
            "k_mek_deg":    self._effective_param("k_mek_deg",  self.model.k_params["k_mek_deg"]),
            "k_mek_braf":   self._effective_param("k_mek_braf", self.model.k_params["k_mek_braf"]),
            "k_erk":        self._effective_param("k_erk",      self.model.k_params["k_erk"]),
            "k_erk_deg":    self._effective_param("k_erk_deg",  self.model.k_params["k_erk_deg"]),
            "k_her2_tx":    self._effective_param("k_her2_tx",  self.model.k_params["k_her2_tx"]),
            "k_her3_tx":    self._effective_param("k_her3_tx",  self.model.k_params["k_her3_tx"]),
            "k_dusp_synth": self._effective_param("k_dusp_synth", self.model.k_params["k_dusp_synth"]),
        }

        # IC50 drift monitoring — track whether IC50_vem is staying within bounds
        ic50_monitor = {}
        for key in ['IC50_vem', 'IC50_tram', 'IC50_pi3k', 'IC50_ras']:
            raw_val = float(self.model.k_params[key].detach().cpu())
            eff_val = float(F.softplus(self.model.k_params[key].detach().cpu()))
            n_hill  = float(torch.clamp(self.model.k_params['hill_coeff'].detach(), 1.0, 4.0))
            eps     = 1e-7
            dose    = float(self._x_train.as_subclass(torch.Tensor)[:, 1].mean())
            drug_effect = (dose + eps)**n_hill / (eff_val**n_hill + (dose + eps)**n_hill + 1e-8)
            ic50_monitor[key] = {
                "raw": round(raw_val, 4),
                "effective": round(eff_val, 4),
                "drug_effect": round(float(drug_effect), 4),
            }
        snapshot["ic50_monitor"] = ic50_monitor

        # Contrast loss monitoring — track drug sensitivity over training
        if self.problem._solver_ref is not None:
            try:
                contrast_val = float(
                    self.problem.drug_contrast_loss(
                        solver=self.problem._solver_ref,
                        n_times=20,
                        t_range=(0.05, 0.70),
                    ).item()
                )
            except Exception:
                contrast_val = -1.0
        else:
            contrast_val = -1.0
        snapshot["contrast_loss"] = contrast_val

        with torch.no_grad():
            inp_t = self.problem._conditions['physics'].input.as_subclass(torch.Tensor)
            pred_t = self.solver.forward(self.problem._conditions['physics'].input).as_subclass(torch.Tensor)
            y_unorm = pred_t * self._y_range + self._y_min
            pred_pERK_mean = float(y_unorm[:, 6].mean().item())
            pred_pMEK_mean = float(y_unorm[:, 5].mean().item())
            pred_pCRAF_mean = float(y_unorm[:, 4].mean().item())

        grad_norms = {}
        for loss_name, cond in self.problem._conditions.items():
            try:
                if hasattr(cond, 'equation') and cond.equation is not None:
                    inp_g = cond.input.clone().detach().requires_grad_(True)
                    pred_g = self.solver.forward(inp_g)
                    res = cond.equation.residual(inp_g, pred_g)
                    loss = (res.as_subclass(torch.Tensor) ** 2).mean()
                else:
                    inp_g = cond.input.clone().detach()
                    pred_g = self.solver.forward(inp_g)
                    output_tensor = cond.target.as_subclass(torch.Tensor) if hasattr(cond, 'target') else cond.output_pts.as_subclass(torch.Tensor)
                    loss = nn.MSELoss()(pred_g.as_subclass(torch.Tensor), output_tensor)
                loss.backward(retain_graph=True)
                total_norm = 0.0
                for p in self.model.net.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms[loss_name] = float(total_norm ** 0.5)
                self.model.net.zero_grad()
            except Exception:
                grad_norms[loss_name] = -1.0

        snapshot.update({
            "epoch": int(epoch),
            "pred_pERK_mean": pred_pERK_mean,
            "pred_pMEK_mean": pred_pMEK_mean,
            "pred_pCRAF_mean": pred_pCRAF_mean,
            "gradient_norms": grad_norms,
            "data_mse_per_species": data_mse_per_species,
            "physics_mse_per_species": physics_mse_per_species,
            "raw_parameters": raw_params,
            "effective_parameters": effective_params,
        })
        return snapshot

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.every_n_epochs != 0:
            return
        if self.snapshots_written >= self.max_snapshots:
            return

        snapshot = self._collect_snapshot(epoch)
        ckpt_path = self.output_dir / f"model_epoch_{epoch:04d}.pth"
        torch.save(self.model.state_dict(), ckpt_path)
        snapshot["checkpoint"] = str(ckpt_path)

        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot) + "\n")

        self.snapshots_written += 1
        LOGGER.info(
            "Saved optimization snapshot %d/%d at epoch %d",
            self.snapshots_written,
            self.max_snapshots,
            epoch,
        )

class ValidationCallback(Callback):
    def __init__(self, solver, val_data, scalers, every_n=100):
        self.solver = solver
        self.scalers = scalers
        self.every_n = every_n
        t_v = torch.tensor(val_data["t_norm"], dtype=torch.float32)
        d_v = torch.tensor(val_data["drugs"],  dtype=torch.float32)
        self._x_val = LabelTensor(
            torch.cat([t_v, d_v], dim=1),
            ["t", "vem", "tram", "pi3k", "ras"],
        )
        self._y_val = torch.tensor(val_data["y_norm"], dtype=torch.float32)

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n != 0:
            return
        with torch.no_grad():
            pred = self.solver.forward(self._x_val).as_subclass(torch.Tensor)
        val_mse = nn.MSELoss()(pred, self._y_val).item()
        LOGGER.info("Epoch %d | Val MSE (normalised): %.6f", trainer.current_epoch + 1, val_mse)

class CollocationResampleCallback(Callback):
    """Resample physics collocation points every N epochs to prevent overfitting."""
    def __init__(self, problem, n_points=4000, every_n=200):
        self.problem = problem
        self.n_points = n_points
        self.every_n = every_n

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n != 0:
            return

        # Resample main physics condition
        t_col, drugs_col = get_vem_collocation_points(
            n_points=self.n_points,
            late_time_extra=500,
        )
        X_phys_raw = torch.cat([t_col, drugs_col], dim=1)
        X_phys_new = LabelTensor(X_phys_raw, self.problem.input_variables)
        self.problem._conditions['physics'] = Condition(
            input=X_phys_new,
            equation=Equation(self.problem.signaling_odes)
        )

        # Resample late-time physics condition independently
        t_late_hours = np.random.uniform(24.0, 48.0, size=(800,)).astype(np.float32)
        t_late_norm  = (t_late_hours / 48.0).reshape(-1, 1)
        drugs_late   = np.zeros((800, 4), dtype=np.float32)
        drugs_late[:, 0] = VEM_CONCENTRATION
        X_late_raw = torch.tensor(
            np.concatenate([t_late_norm, drugs_late], axis=1), dtype=torch.float32
        )
        X_late_new = LabelTensor(X_late_raw, self.problem.input_variables)
        self.problem._conditions['physics_late'] = Condition(
            input=X_late_new,
            equation=Equation(self.problem.signaling_odes)
        )

        # Resample mek_vem_ss condition
        t_mek_ss_new = torch.FloatTensor(200).uniform_(24.0 / 48.0, 1.0).unsqueeze(1)
        drugs_mek_ss_new = torch.zeros(200, 4, dtype=torch.float32)
        drugs_mek_ss_new[:, 0] = VEM_CONCENTRATION
        X_mek_ss_new = LabelTensor(
            torch.cat([t_mek_ss_new, drugs_mek_ss_new], dim=1),
            self.problem.input_variables
        )
        self.problem._conditions['mek_vem_ss'] = Condition(
            input=X_mek_ss_new,
            equation=Equation(self.problem.steady_state_odes)
        )

        # Resample contrast condition — early window only (t < 12h = 0.25 normalised)
        t_contrast_new = torch.FloatTensor(40).uniform_(0.05, 0.25).unsqueeze(1)
        drugs_contrast_new = torch.zeros(40, 4, dtype=torch.float32)
        drugs_contrast_new[:, 0] = VEM_CONCENTRATION
        X_contrast_new = LabelTensor(
            torch.cat([t_contrast_new, drugs_contrast_new], dim=1),
            self.problem.input_variables
        )
        self.problem._conditions['contrast'] = Condition(
            input=X_contrast_new,
            equation=Equation(self.problem.contrast_equation)
        )

        LOGGER.info(
            "Epoch %d: Resampled physics (%d pts), physics_late (800 pts), contrast, and mek_vem_ss.",
            trainer.current_epoch + 1, self.n_points
        )

class LRDecayCallback(Callback):
    """Halve learning rate if physics residual MSE hasn't improved in patience epochs."""
    def __init__(self, solver, patience=300, factor=0.5, min_lr=1e-5):
        self.solver = solver
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_phys_mse = float('inf')
        self.epochs_no_improve = 0

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 50 != 0:
            return
        phys_input = pl_module.problem._conditions['physics'].input
        phys_input_g = phys_input.clone().detach().requires_grad_(True)
        phys_pred_g = self.solver.forward(phys_input_g)
        phys_res = pl_module.problem.signaling_odes(phys_input_g, phys_pred_g)
        phys_mse = (phys_res.as_subclass(torch.Tensor) ** 2).mean().item()
        phys_mse = 0.0 if np.isnan(phys_mse) else phys_mse

        if phys_mse < self.best_phys_mse * 0.99:
            self.best_phys_mse = phys_mse
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 50

        if self.epochs_no_improve >= self.patience:
            for pg in self.solver.optimizers().param_groups:
                old_lr = pg['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                pg['lr'] = new_lr
                if old_lr != new_lr:
                    LOGGER.info(
                        "LR reduced %.2e → %.2e (physics MSE stalled at %.6f)",
                        old_lr, new_lr, phys_mse
                    )
            self.epochs_no_improve = 0

INITIAL_K_PARAMS ={
'hill_coeff':2.0 ,'IC50_vem':0.1 ,'IC50_tram':0.1 ,'IC50_pi3k':0.1 ,'IC50_ras':0.1 ,
'k_paradox':-0.5 ,'k_egfr':0.5 ,'k_egfr_deg':1.0 ,'k_her2':0.5 ,'k_her2_deg':1.0 ,
'k_her3':0.5 ,'k_her3_deg':1.0 ,'k_igf':0.5 ,'k_igf_deg':1.0 ,
'k_erk_rtk':0.5 ,'Km_rtk':0.5 ,'k_up':0.1 ,'k_erk_sos':0.5 ,'Km_sos':0.5 ,
'k_akt_rtk':0.5 ,'Km_artk':0.5 ,
'k_craf':0.3 ,'k_craf_deg':0.5 ,'k_mek':0.1 ,'k_mek_deg':1.0 ,'k_mek_braf':0.3 ,'braf_ic50':0.3 ,
'k_erk':1.0 ,'k_erk_deg':1.0 ,
'k_dusp_synth':1.0 ,'k_dusp_deg':1.0 ,'k_dusp_cat':0.5 ,
'Km_dusp':0.5 ,'Km_dusp_s':0.5 ,'n_dusp':2.0 ,
'k_raf_pi3k':0.1 ,'Km_raf_pi3k':0.5 ,
'k_erk_pi3k':0.1 ,'Km_erk_pi3k':0.5 ,
'k_akt':1.0 ,'k_akt_deg':1.0 ,
'k_4ebp1':1.0 ,'k_4ebp1_deg':1.0 ,'k_4ebp1_comp':0.05 ,'Km_4ebp1':0.5 ,
'k_akt_raf':0.1 ,'Km_akt_raf':0.5 ,
'k_her2_tx':0.5 ,'k_her3_tx':0.5 ,'k_ras_pi3k_frac':0.1 ,
'K_sat_egfr':1.0 ,'K_sat_her2':1.0 ,'K_sat_her3':1.0 ,'K_sat_igfr':1.0 ,
'K_sat_craf':1.0 ,'K_sat_mek':1.0 ,'K_sat_erk':1.0 ,'K_sat_akt':1.0 ,'K_sat_4ebp1':1.0 ,
}

class SignalingModel (nn .Module ):
    def __init__ (self ):
        super ().__init__ ()
        self .net =FeedForward (
        input_dimensions =5 ,output_dimensions =10 ,
        layers =[64 ,64 ],func =nn .Tanh ,
        )
        self .k_params =nn .ParameterDict ({
        k :nn .Parameter (torch .tensor (v ,dtype =torch .float32 ))
        for k ,v in INITIAL_K_PARAMS .items ()
        })
        self.register_buffer(
            "ode_species_weights",
            torch.tensor([
                1.0,   # pEGFR
                0.5,   # HER2
                0.5,   # HER3
                0.5,   # IGF1R
                2.0,   # pCRAF
                3.0,   # pMEK
                3.0,   # pERK
                2.0,   # DUSP6
                1.5,   # pAKT
                1.0,   # p4EBP1
            ], dtype=torch.float32)
        )

    def forward (self ,x ):

        out =self .net (x )

        return F .softplus (out )

class SignalingProblem (AbstractProblem ):
    def __init__ (self ,train_data ,scalers ,model ,ic_tensors=None, solver_ref=None):
        self ._output_variables =SPECIES_ORDER 
        self .temporal_variable =['t']
        self .parameters =['vem','tram','pi3k','ras']
        self .scalers =scalers 
        self ._model =model 
        self._solver_ref = solver_ref   # set after solver is created; None initially

        self._y_range_buf = torch.tensor(scalers['y_range'], dtype=torch.float32).squeeze()
        self._y_min_buf = torch.tensor(scalers['y_min'], dtype=torch.float32).squeeze()
        self._t_range_buf = torch.tensor(float(scalers['t_range']), dtype=torch.float32)

        t_norm =torch .tensor (train_data ['t_norm'],dtype =torch .float32 )
        drugs =torch .tensor (train_data ['drugs'],dtype =torch .float32 )
        X_data =LabelTensor (torch .cat ([t_norm ,drugs ],dim =1 ),self .input_variables )
        Y_data =LabelTensor (
        torch .tensor (train_data ['y_norm'],dtype =torch .float32 ),
        self .output_variables ,
        )

        t_col, drugs_col = get_vem_collocation_points(n_points=4000, late_time_extra=500)
        X_phys_raw = torch.cat([t_col, drugs_col], dim=1)
        X_phys = LabelTensor(X_phys_raw, self.input_variables)

        # Late-time physics condition: dedicated collocation for t in [24, 48h]
        # This forces the ODE to be satisfied during the adaptive resistance window.
        t_late_hours = np.random.uniform(24.0, 48.0, size=(800,)).astype(np.float32)
        t_late_norm  = (t_late_hours / 48.0).reshape(-1, 1)
        drugs_late   = np.zeros((800, 4), dtype=np.float32)
        drugs_late[:, 0] = VEM_CONCENTRATION   # will be overridden by ACTUAL_VEM_CONCENTRATION at runtime

        X_late_raw = torch.tensor(
            np.concatenate([t_late_norm, drugs_late], axis=1), dtype=torch.float32
        )
        X_late = LabelTensor(X_late_raw, self.input_variables)

        # Steady-state: no-drug collocation points only (drug columns all zero)
        # This anchors basal equilibrium separately from the Vem trajectory
        t_ss = torch.linspace(0, 1, 200).unsqueeze(1)
        drugs_ss = torch.zeros(200, 4)
        X_ss_raw = torch.cat([t_ss, drugs_ss], dim=1)
        X_ss = LabelTensor(X_ss_raw, self.input_variables)

        # Contrast condition: restricted to early suppression window (t < 12h)
        # Beyond t=12h, adaptive resistance causes pERK rebound — do not enforce
        # drug suppression there.
        t_contrast = torch.linspace(0.05, 0.25, 40).unsqueeze(1)   # t in [2.4h, 12h]
        drugs_contrast = torch.zeros(40, 4, dtype=torch.float32)
        drugs_contrast[:, 0] = VEM_CONCENTRATION
        X_contrast_raw = torch.cat([t_contrast, drugs_contrast], dim=1)
        X_contrast = LabelTensor(X_contrast_raw, self.input_variables)

        # pMEK drug steady-state collocation: 200 points at t in [24, 48h] under Vem.
        # Enforces that the MEK ODE derivative approaches zero at late times,
        # consistent with observed flat/low pMEK under Vemurafenib.
        t_mek_ss = torch.FloatTensor(200).uniform_(24.0 / 48.0, 1.0).unsqueeze(1)
        drugs_mek_ss = torch.zeros(200, 4, dtype=torch.float32)
        drugs_mek_ss[:, 0] = VEM_CONCENTRATION
        X_mek_ss = LabelTensor(
            torch.cat([t_mek_ss, drugs_mek_ss], dim=1),
            self.input_variables
        )

        self._conditions = {
            'data':         Condition(input=X_data, target=Y_data),
            'physics':      Condition(input=X_phys, equation=Equation(self.signaling_odes)),
            'physics_late': Condition(input=X_late, equation=Equation(self.signaling_odes)),
            'steady_state': Condition(input=X_ss,   equation=Equation(self.steady_state_odes)),
            'mek_vem_ss':   Condition(input=X_mek_ss, equation=Equation(self.steady_state_odes)),
            'contrast':     Condition(input=X_contrast, equation=Equation(self.contrast_equation)),
        }

        # Optional initial condition pin
        if ic_tensors is not None:
            X_ic, Y_ic = ic_tensors
            self._conditions['ic'] = Condition(input=X_ic, target=Y_ic)
            LOGGER.info("Added initial condition (ic) loss term with %d points", X_ic.shape[0])

        super ().__init__ ()

    @property 
    def output_variables (self ):return self ._output_variables 

    @property 
    def conditions (self ):return self ._conditions 

    def signaling_odes (self ,input_ ,output_ ):
        k =self ._model .k_params 
        eps =1e-7 

        dy_dt_lt = pina_grad (output_ ,input_ ,components =self .output_variables ,d ='t')
        dy_dt_norm =dy_dt_lt .as_subclass (torch .Tensor )
        y_norm =output_ .as_subclass (torch .Tensor )
        inp =input_ .as_subclass (torch .Tensor )

        device = input_.device
        y_range = self._y_range_buf.to(device)
        y_min = self._y_min_buf.to(device)
        t_range = self._t_range_buf.to(device)

        y = y_norm * y_range + y_min
        dy_dt = (dy_dt_norm * y_range) / t_range

        y_safe = torch.clamp(y, min=1e-6)

        pEGFR ,HER2 ,HER3 ,IGF1R =y_safe [:,0 ],y_safe [:,1 ],y_safe [:,2 ],y_safe [:,3 ]
        pCRAF ,pMEK ,pERK ,DUSP6 =y_safe [:,4 ],y_safe [:,5 ],y_safe [:,6 ],y_safe [:,7 ]
        pAKT ,p4EBP1 =y_safe [:,8 ],y_safe [:,9 ]

        Vem =torch .abs (inp [:,1 ])
        Tram =torch .abs (inp [:,2 ])
        PI3Ki =torch .abs (inp [:,3 ])
        RasInh =torch .abs (inp [:,4 ])

        n =torch .clamp (k ['hill_coeff'],1.0 ,4.0 )

        Vem_inh =(Vem +eps )**n /(F .softplus (k ['IC50_vem'])**n +(Vem +eps )**n +1e-8 )
        Tram_eff =(Tram +eps )**n /(F .softplus (k ['IC50_tram'])**n +(Tram +eps )**n +1e-8 )
        PI3Ki_eff =(PI3Ki +eps )**n /(F .softplus (k ['IC50_pi3k'])**n +(PI3Ki +eps )**n +1e-8 )
        Ras_eff =(RasInh +eps )**n /(F .softplus (k ['IC50_ras'])**n +(RasInh +eps )**n +1e-8 )

        Ks_egfr =F .softplus (k ['K_sat_egfr'])
        Ks_her2 =F .softplus (k ['K_sat_her2'])
        Ks_her3 =F .softplus (k ['K_sat_her3'])
        Ks_igfr =F .softplus (k ['K_sat_igfr'])
        Ks_craf =F .softplus (k ['K_sat_craf'])
        Ks_mek =F .softplus (k ['K_sat_mek'])
        Ks_erk =F .softplus (k ['K_sat_erk'])
        Ks_akt =F .softplus (k ['K_sat_akt'])
        Ks_4ebp1 =F .softplus (k ['K_sat_4ebp1'])

        k_erk_rtk =F .softplus (k ['k_erk_rtk'])
        Km_rtk =F .softplus (k ['Km_rtk'])
        ERK_feedback =k_erk_rtk *pERK /(Km_rtk +pERK +1e-8 )

        k_up =F .softplus (k ['k_up'])
        drug_relief =k_up *(Vem_inh +Tram_eff +PI3Ki_eff )

        k_erk_sos =F .softplus (k ['k_erk_sos'])
        k_akt_rtk =F .softplus (k ['k_akt_rtk'])
        Km_sos =F .softplus (k ['Km_sos'])
        Km_artk =F .softplus (k ['Km_artk'])
        ERK_to_SOS =k_erk_sos *pERK /(Km_sos +pERK +1e-8 )
        AKT_to_RTK =k_akt_rtk *pAKT /(Km_artk +pAKT +1e-8 )

        RTK_total =pEGFR +HER2 + HER3 +IGF1R 
        ras_suppression = (1.0 - ERK_to_SOS) * (1.0 - AKT_to_RTK) * (1.0 - Ras_eff)
        RAS_GTP = RTK_total * torch.clamp(ras_suppression, min=0.05)

        k_raf_pi3k =F .softplus (k ['k_raf_pi3k'])
        Km_raf_pi3k =F .softplus (k ['Km_raf_pi3k'])
        k_erk_pi3k =F .softplus (k ['k_erk_pi3k'])
        Km_erk_pi3k =F .softplus (k ['Km_erk_pi3k'])
        RAF_to_PI3K =k_raf_pi3k *pCRAF /(Km_raf_pi3k +pCRAF +1e-8 )
        ERK_to_PI3K =k_erk_pi3k *pERK /(Km_erk_pi3k +pERK +1e-8 )

        k_ras_frac =F .softplus (k ['k_ras_pi3k_frac'])
        PI3K_input =RTK_total *(1.0 -ERK_to_PI3K )*(1.0 -k_ras_frac *Ras_eff )+RAF_to_PI3K 

        k_akt_raf =F .softplus (k ['k_akt_raf'])
        Km_akt_raf =F .softplus (k ['Km_akt_raf'])
        AKT_RAF_inhib =k_akt_raf *pAKT /(Km_akt_raf +pAKT +1e-8 )

        k_paradox =F .softplus (k ['k_paradox'])
        pCRAF_fl =pCRAF .clamp (min =0.05 )
        pi3ki_att =1.0 -0.7 *PI3Ki_eff 
        Vem_paradox =k_paradox *Vem *Ks_craf /(Ks_craf +pCRAF_fl +1e-8 )*pi3ki_att 

        ke =F .softplus (k ['k_egfr']);ked =F .softplus (k ['k_egfr_deg'])
        res0 =dy_dt [:,0 ]-(ke *(1.0 +drug_relief )*Ks_egfr /(Ks_egfr +pEGFR +1e-8 )-(ked +ERK_feedback )*pEGFR )

        kh2 =F .softplus (k ['k_her2']);kh2d =F .softplus (k ['k_her2_deg']);kh2tx =F .softplus (k ['k_her2_tx'])
        res1 =dy_dt [:,1 ]-(kh2 *Ks_her2 /(Ks_her2 +HER2 +1e-8 )+kh2tx *(1.0 -pERK /(Ks_erk +pERK +1e-8 ))-kh2d *HER2 )

        kh3 =F .softplus (k ['k_her3']);kh3d =F .softplus (k ['k_her3_deg']);kh3tx =F .softplus (k ['k_her3_tx'])
        res2 =dy_dt [:,2 ]-(kh3 *Ks_her3 /(Ks_her3 +HER3 +1e-8 )+kh3tx *(1.0 -pERK /(Ks_erk +pERK +1e-8 ))-kh3d *HER3 )

        ki =F .softplus (k ['k_igf']);kid =F .softplus (k ['k_igf_deg'])
        res3 =dy_dt [:,3 ]-(ki *Ks_igfr /(Ks_igfr +IGF1R +1e-8 )-(kid +ERK_feedback +AKT_to_RTK )*IGF1R )

        kc =F .softplus (k ['k_craf']);kcd =F .softplus (k ['k_craf_deg'])
        res4 =dy_dt [:,4 ]-(kc *RAS_GTP *Ks_craf /(Ks_craf +pCRAF +1e-8 )+Vem_paradox -(kcd +AKT_RAF_inhib )*pCRAF )

        km  = F.softplus(k['k_mek'])
        kmd = F.softplus(k['k_mek_deg'])
        kmb = F.softplus(k['k_mek_braf'])
        # braf_ic50 is now learnable — constrained by braf_ic50_prior_loss
        # Using softplus to ensure positivity; initial effective value = softplus(0.3) ≈ 0.85 µM
        braf_ic50_eff = F.softplus(k['braf_ic50'])
        Vem_braf = (Vem + eps)**n / (braf_ic50_eff**n + (Vem + eps)**n + 1e-8)
        res5 = dy_dt[:, 5] - (
            (km * pCRAF + kmb * (1.0 - Vem_braf)) * (1.0 - Tram_eff)
            * Ks_mek / (Ks_mek + pMEK + 1e-8) - kmd * pMEK
        )

        kdusp_cat =F .softplus (k ['k_dusp_cat']);Km_dusp =F .softplus (k ['Km_dusp'])
        DUSP_act =kdusp_cat *DUSP6 /(Km_dusp +DUSP6 +1e-8 )
        ker =F .softplus (k ['k_erk']);kerd =F .softplus (k ['k_erk_deg'])
        res6 =dy_dt [:,6 ]-(ker *pMEK *Ks_erk /(Ks_erk +pERK +1e-8 )-(kerd +DUSP_act )*pERK )

        n_dusp =torch .clamp (k ['n_dusp'],1.5 ,3.5 )
        kds =F .softplus (k ['k_dusp_synth']);Km_ds =F .softplus (k ['Km_dusp_s'])
        kdd =F .softplus (k ['k_dusp_deg'])
        DUSP_ind =kds *(pERK +eps )**n_dusp /(Km_ds **n_dusp +(pERK +eps )**n_dusp +1e-8 )
        res7 =dy_dt [:,7 ]-(DUSP_ind -kdd *DUSP6 )

        ka =F .softplus (k ['k_akt']);kad =F .softplus (k ['k_akt_deg'])
        res8 =dy_dt [:,8 ]-(ka *PI3K_input *(1.0 -PI3Ki_eff )*Ks_akt /(Ks_akt +pAKT +1e-8 )-kad *pAKT )
        k4 =F .softplus (k ['k_4ebp1']);k4d =F .softplus (k ['k_4ebp1_deg']);k4b =F .softplus (k ['k_4ebp1_comp'])
        res9 =dy_dt [:,9 ]-(k4 *pAKT *Ks_4ebp1 /(Ks_4ebp1 +p4EBP1 +1e-8 )+k4b -k4d *p4EBP1 )

        residuals = torch.stack(
            [res0, res1, res2, res3, res4, res5, res6, res7, res8, res9],
            dim=1
        )
        w = self._model.ode_species_weights
        residuals = residuals * w.unsqueeze(0)

        residuals = torch.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)

        # Add IC50 prior as a scalar broadcast across all species residuals.
        # This ensures the IC50 prior contributes to the physics condition gradient
        # without requiring a separate PINA condition entry.
        # Divided by 10 to scale relative to per-species ODE residuals.
        ic50_prior = self.ic50_prior_loss() / 10.0
        residuals = residuals + ic50_prior

        out = residuals .as_subclass (LabelTensor )
        out .labels =[f"res_{s }"for s in SPECIES_ORDER ]
        return out 

    def steady_state_odes (self ,input_ ,output_ ):

        dy_dt_lt =pina_grad (output_ ,input_ ,components =self .output_variables ,d ='t')
        dy_dt_norm =dy_dt_lt .as_subclass (torch .Tensor )
        dy_dt = (dy_dt_norm * self._y_range_buf.to(input_.device)) / self._t_range_buf.to(input_.device)

        out =dy_dt .as_subclass (LabelTensor )
        out .labels =[f"ss_{s }"for s in SPECIES_ORDER ]
        return torch .nan_to_num (out ,nan =0.0 )

    def param_regularization_loss(self) -> torch.Tensor:
        """Penalize kinetic parameters that drift outside [0, 5]."""
        loss = torch.tensor(0.0)
        for param in self._model.k_params.values():
            loss = loss + torch.clamp(param - 5.0, min=0.0) ** 2
            loss = loss + torch.clamp(-param, min=0.0) ** 2
        return loss * 0.01

    def ic50_prior_loss(self) -> torch.Tensor:
        """
        Constrain IC50 parameters to stay within biologically plausible ranges.

        The experimental Vemurafenib dose is 0.5 µM. For the drug to have a meaningful
        effect (Hill term between 0.1 and 0.9), IC50_vem must be within roughly one
        order of magnitude of the dose: IC50_vem ∈ [0.05, 2.0].

        If IC50_vem drifts above 2.0, the drug term Vem/(IC50^n + Vem^n) at dose=0.5
        falls below ~0.05, making the drug nearly invisible to the ODE.

        Penalty structure:
          - Soft quadratic penalty outside the allowed range
          - Extra hard penalty for extreme drift (IC50_eff > 5.0)
          - Applied to all four IC50 parameters with drug-specific bounds
        """
        loss = torch.tensor(0.0, dtype=torch.float32)

        # (param_key, effective_min, effective_max, hard_max)
        ic50_bounds = [
            ('IC50_vem',  0.05, 2.0, 5.0),   # Vem dose = 0.5 µM
            ('IC50_tram', 0.01, 1.0, 3.0),   # Tram dose = 0.3 µM
            ('IC50_pi3k', 0.05, 2.0, 5.0),   # PI3Ki dose = 1.0 µM
            ('IC50_ras',  0.05, 2.0, 5.0),   # RasInh dose = 1.0 µM
            # braf_ic50: must be > 0.1 (otherwise saturated at experimental dose)
            # and < 2.0 (otherwise Vemurafenib has no BRAF effect at dose=0.5)
            ('braf_ic50', 0.1,  2.0, 5.0),
        ]

        for param_key, lo, hi, hard_max in ic50_bounds:
            eff = F.softplus(self._model.k_params[param_key])

            # Soft penalty: quadratic outside [lo, hi]
            loss = loss + torch.clamp(eff - hi,       min=0.0) ** 2 * 5.0
            loss = loss + torch.clamp(lo - eff,       min=0.0) ** 2 * 5.0

            # Hard penalty: steep quadratic beyond hard_max
            loss = loss + torch.clamp(eff - hard_max, min=0.0) ** 2 * 50.0

        return loss.to(self._model.ode_species_weights.device)

    def drug_contrast_loss(
        self,
        solver,
        n_times: int = 50,
        t_range: tuple = (0.05, 0.25),   # default: early suppression window only
    ) -> torch.Tensor:
        """
        Enforce that pERK and pMEK are lower at the experimental Vemurafenib dose
        than at zero dose, at matched time points across the training window.

        Samples n_times time points uniformly in t_range (normalised).
        For each time t, computes:
            contrast_loss = ReLU(pERK(t, vem=dose) - pERK(t, vem=0) + margin)

        A positive loss means the drug-treated prediction is >= baseline minus margin,
        i.e. the drug is not suppressing pERK enough.

        This is a physics-consistent constraint: Vemurafenib inhibits BRAF, which
        reduces MEK activation, which reduces ERK phosphorylation. Under the ODE
        structure, lower RAS_GTP → lower pCRAF → lower pMEK → lower pERK.

        Parameters
        ----------
        solver      : the PINN solver (used to call forward)
        n_times     : number of time points to sample
        t_range     : (t_min_norm, t_max_norm) — use 0.05–0.7 to focus on t=2.4–33.6h
                      where the drug effect should be established but before late rebound
        """
        eps_dose = torch.tensor(1e-5, dtype=torch.float32)  # "zero drug" baseline

        # Sample time points
        t_vals = torch.FloatTensor(n_times).uniform_(t_range[0], t_range[1]).unsqueeze(1)

        # Build input pairs: (t, vem=0) and (t, vem=dose)
        # Note: VEM_CONCENTRATION is the module-level constant
        vem_dose = float(VEM_CONCENTRATION)

        zero_drug = torch.zeros(n_times, 4, dtype=torch.float32)
        zero_drug[:, 0] = eps_dose   # near-zero Vem, others zero

        dose_drug = torch.zeros(n_times, 4, dtype=torch.float32)
        dose_drug[:, 0] = vem_dose

        X_zero = LabelTensor(
            torch.cat([t_vals, zero_drug], dim=1),
            self.input_variables
        )
        X_dose = LabelTensor(
            torch.cat([t_vals, dose_drug], dim=1),
            self.input_variables
        )

        # Forward pass (no grad for inputs — we only want output values here)
        pred_zero = solver.forward(X_zero).as_subclass(torch.Tensor)
        pred_dose = solver.forward(X_dose).as_subclass(torch.Tensor)

        # Un-normalise to biological scale for interpretable margin
        y_range = self._y_range_buf
        y_min   = self._y_min_buf
        y_zero = pred_zero * y_range + y_min
        y_dose = pred_dose * y_range + y_min

        pERK_idx = SPECIES_ORDER.index("pERK")
        pMEK_idx = SPECIES_ORDER.index("pMEK")

        pERK_zero = y_zero[:, pERK_idx]
        pERK_dose = y_dose[:, pERK_idx]
        pMEK_zero = y_zero[:, pMEK_idx]
        pMEK_dose = y_dose[:, pMEK_idx]

        # Margin: absolute floor ensures the contrast loss cannot be satisfied by
        # collapsing pERK to near-zero. The margin is the LARGER of:
        #   - 20% of the baseline prediction (relative)
        #   - 0.15 biological units (absolute floor)
        # When pERK_zero=0.10, relative=0.02 but floor=0.15 dominates.
        # When pERK_zero=1.0,  relative=0.20 and floor=0.15, relative dominates.
        pERK_margin = torch.clamp(0.20 * pERK_zero.detach(), min=0.15)
        pMEK_margin = torch.clamp(0.15 * pMEK_zero.detach(), min=0.05)

        # Hinge loss: penalise when drug-treated >= baseline - margin
        contrast_pERK = F.relu(pERK_dose - (pERK_zero.detach() - pERK_margin))
        contrast_pMEK = F.relu(pMEK_dose - (pMEK_zero.detach() - pMEK_margin))

        # Remove amplitude_floor penalty entirely — the IC condition (ic_weight=30)
        # is now responsible for anchoring pERK amplitude. The floor was creating a
        # ceiling by causing the optimizer to satisfy it minimally at ~0.5.
        loss = contrast_pERK.mean() + 0.5 * contrast_pMEK.mean()

        return loss

    def contrast_equation(self, input_: LabelTensor, output_: LabelTensor) -> LabelTensor:
        """
        Wrapper that computes drug contrast loss as a per-sample residual.
        Returns a LabelTensor of shape (N, 1) — all entries equal to the
        scalar contrast loss, broadcast so PINA treats it as a condition.
        """
        if self._solver_ref is None:
            # Solver not yet attached — return zeros (no-op during problem construction)
            n = output_.shape[0]
            out = torch.zeros(n, 1, dtype=torch.float32)
            out_lt = out.as_subclass(LabelTensor)
            out_lt.labels = ["contrast_res"]
            return out_lt

        contrast = self.drug_contrast_loss(
            solver=self._solver_ref,
            n_times=40,
            t_range=(0.05, 0.25),   # 0.25 normalised = 12h — early suppression window only
        )
        n = output_.shape[0]
        # Broadcast scalar loss to shape (N, 1)
        out = contrast.expand(n, 1)
        out_lt = out.as_subclass(LabelTensor)
        out_lt.labels = ["contrast_res"]
        return out_lt

if __name__ == "__main__":
    seed = 42
    normalization_mode = "train_only"
    split_mode = "cutoff"
    train_until_hour = 48.0
    max_epochs = 2000   # extra 2000 epochs for late-time rebound convergence
    learning_rate = 2e-4 # slightly lower: reduced contrast weight means less noise
                         # from that loss term; finer steps needed for late dynamics
    condition_tag = TARGET_CONDITION.replace(" ", "_").replace("/", "_")
    set_seed(seed)

    LOGGER.info("=" * 60)
    LOGGER.info("PINN — Single Condition: %s", TARGET_CONDITION)
    LOGGER.info("=" * 60)
    LOGGER.info("Seed=%d | lr=%.5f | epochs=%d", seed, learning_rate, max_epochs)

    # ── 1. Load full dataset ──────────────────────────────────────────────────
    all_train_data, all_test_data, scalers = prepare_training_tensors(
        split_mode=split_mode,
        train_until_hour=train_until_hour,
        normalization_mode=normalization_mode,
    )
    LOGGER.info("Normalization mode=%s", scalers["normalization_mode"])
    LOGGER.info(
        "Full dataset — Train: %d | Test: %d",
        len(all_train_data["t"]), len(all_test_data["t"]),
    )

    # ── 2. Filter to Vemurafenib-only ─────────────────────────────────────────
    train_data = filter_to_condition(all_train_data, TARGET_CONDITION)
    test_data  = filter_to_condition(all_test_data,  TARGET_CONDITION) \
                 if len(all_test_data["t"]) > 0 else all_test_data

    train_data, val_data = temporal_train_val_split(train_data, val_fraction=0.2)
    LOGGER.info("Condition train: %d | Condition val: %d", len(train_data["t"]), len(val_data["t"]))

    # ── Data Derivations ──────────────────────────────────────────────────────
    drugs_arr = np.array(train_data["drugs"])
    vem_col = drugs_arr[:, 0]
    LOGGER.info(
        "Vem drug column stats — min: %.4f | max: %.4f | mean: %.4f | unique: %s",
        vem_col.min(), vem_col.max(), vem_col.mean(),
        np.unique(vem_col).round(4).tolist(),
    )
    ACTUAL_VEM_CONCENTRATION = float(np.unique(vem_col[vem_col > 1e-6]).mean())
    LOGGER.info("Using ACTUAL_VEM_CONCENTRATION=%.4f from data", ACTUAL_VEM_CONCENTRATION)

    t_min_hours = float(scalers.get("t_min", 0.0))
    t_range     = float(scalers["t_range"])
    t_norm_at_zero = (0.0 - t_min_hours) / t_range
    LOGGER.info(
        "t_norm at t=0h: %.4f (t_min=%.2fh, t_range=%.2fh)",
        t_norm_at_zero, t_min_hours, t_range
    )

    # Sanity check: verify network outputs are non-negative after un-normalisation
    with torch.no_grad():
        model_init = SignalingModel()
        sample_input = torch.cat([
            torch.tensor(train_data['t_norm'], dtype=torch.float32)[:5],
            torch.tensor(train_data['drugs'], dtype=torch.float32)[:5]
        ], dim=1)
        sample_output = model_init(sample_input).as_subclass(torch.Tensor)
        sample_unorm = sample_output * scalers['y_range'] + scalers['y_min']
        n_negative = (sample_unorm < 0).sum().item()
        if n_negative > 0:
            LOGGER.warning(
                "%d / %d un-normalised species values are negative at init. "
                "Check y_min in scalers — y_min should be >= 0 for concentrations.",
                n_negative, sample_unorm.numel()
            )
        else:
            LOGGER.info("Un-normalised species values all non-negative at init. OK.")

    # ── 3. Verify collocation normalisation matches scalers ───────────────────
    expected_t_range = float(scalers["t_range"])
    if abs(expected_t_range - 48.0) > 1.0:
        LOGGER.warning(
            "scalers['t_range']=%.2f differs from assumed 48.0h. "
            "Update get_vem_collocation_points divisor to match.",
            expected_t_range,
        )

    # ── 4. Diagnostics: drug column check in collocation ─────────────────────
    t_col_check, drug_col_check = get_vem_collocation_points(n_points=100, vem_concentration=ACTUAL_VEM_CONCENTRATION)
    LOGGER.info(
        "Collocation drug columns — mean: %s | max: %s",
        drug_col_check.mean(dim=0).tolist(),
        drug_col_check.max(dim=0).values.tolist(),
    )
    assert drug_col_check[:, 0].mean().item() > 0.1, \
        "Vemurafenib column in collocation points is near zero — check get_vem_collocation_points"
    assert drug_col_check[:, 1:].abs().max().item() < 1e-6, \
        "Non-vem drug columns in collocation are non-zero — check get_vem_collocation_points"

    # ── 5. Build model and initial conditions ─────────────────────────────────
    model = SignalingModel()

    # ── IC50 parameter diagnostic (runs at init and after training) ───────────────
    def log_ic50_values(model: nn.Module, label: str = "") -> dict:
        """Log effective IC50 values and return them as a dict for run_summary."""
        prefix = f"[{label}] " if label else ""
        LOGGER.info("%sEffective IC50 values:", prefix)
        ic50_summary = {}
        for key in ['IC50_vem', 'IC50_tram', 'IC50_pi3k', 'IC50_ras', 'braf_ic50']:
            raw_val = float(model.k_params[key].detach().cpu())
            eff_val = float(F.softplus(model.k_params[key].detach().cpu()))
            # Compute implied drug effect at the experimental dose
            # In this context, ACTUAL_VEM_CONCENTRATION is the dose for IC50_vem
            vem_dose = ACTUAL_VEM_CONCENTRATION if key == 'IC50_vem' else 0.5
            n_hill   = float(torch.clamp(model.k_params['hill_coeff'].detach(), 1.0, 4.0))
            eps      = 1e-7
            drug_effect = (vem_dose + eps)**n_hill / (eff_val**n_hill + (vem_dose + eps)**n_hill + 1e-8)
            flag = ""
            if key == 'IC50_vem' and drug_effect < 0.2:
                flag = "  *** WARNING: drug effect < 0.20 — IC50 has drifted, drug ineffective ***"
            elif key == 'IC50_vem' and drug_effect > 0.95:
                flag = "  *** WARNING: drug effect > 0.95 — IC50 unrealistically low ***"
            LOGGER.info(
                "  %-15s raw=%+.4f  effective=%.4f  drug_effect@dose=%.3f%s",
                key, raw_val, eff_val, drug_effect, flag
            )
            ic50_summary[key] = {
                "raw": round(raw_val, 4),
                "effective": round(eff_val, 4),
                "drug_effect_at_dose": round(float(drug_effect), 4),
            }
        return ic50_summary

    # Log IC50 values at initialisation — before any training
    _ic50_init = log_ic50_values(model, label="INIT")

    # ── IC50 curriculum constants ─────────────────────────────────────────────────
    IC50_KEYS = ['IC50_vem', 'IC50_tram', 'IC50_pi3k', 'IC50_ras', 'braf_ic50']
    IC50_FREEZE_EPOCHS = 1000   # epochs before IC50 params are allowed to move

    # ── Phase 1: freeze IC50 parameters ──────────────────────────────────────────
    # Rationale: During early training the network learns the trajectory shape and
    # the other kinetic parameters find their correct scale. If IC50 params are free
    # from epoch 0, the optimizer exploits them immediately to make the drug term
    # vanish (trivial solution). Freezing them forces the network to account for the
    # drug effect through the other parameters first.
    for key in IC50_KEYS:
        model.k_params[key].requires_grad_(False)
    LOGGER.info(
        "Phase 1: IC50 parameters FROZEN for first %d epochs: %s",
        IC50_FREEZE_EPOCHS, IC50_KEYS
    )

    class UnfreezeIC50Callback(Callback):
        """
        Unfreeze IC50 parameters at a specified epoch (phase 2 of curriculum training).

        After unfreezing, IC50 parameters are free to move but are constrained by the
        ic50_prior_loss added to the physics condition. This prevents them from drifting
        back to the trivial solution while still allowing refinement.
        """
        def __init__(self, model: nn.Module, unfreeze_epoch: int = IC50_FREEZE_EPOCHS):
            super().__init__()
            self.model = model
            self.unfreeze_epoch = unfreeze_epoch
            self._unfrozen = False

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if self._unfrozen:
                return
            if trainer.current_epoch + 1 >= self.unfreeze_epoch:
                for key in IC50_KEYS:
                    self.model.k_params[key].requires_grad_(True)
                self._unfrozen = True
                # Log effective values at the moment of unfreezing
                LOGGER.info(
                    "Epoch %d: IC50 parameters UNFROZEN (phase 2). Current effective values:",
                    trainer.current_epoch + 1,
                )
                for key in IC50_KEYS:
                    eff = float(F.softplus(self.model.k_params[key].detach()))
                    LOGGER.info("  %-15s effective=%.4f", key, eff)

    # ── pMEK steady-state sanity check ───────────────────────────────────────────
    # Verify that the initial kinetic parameters produce a plausible pMEK SS value
    # before committing to training. Uses a simplified analytical approximation.
    with torch.no_grad():
        _k = model.k_params
        _k_mek      = float(F.softplus(_k['k_mek']))
        _k_mek_deg  = float(F.softplus(_k['k_mek_deg']))
        _k_mek_braf = float(F.softplus(_k['k_mek_braf']))
        _Ks_mek     = float(F.softplus(_k['K_sat_mek']))
        # Approximate: assume pCRAF~0.3 under Vem, Vem_braf~1.0, Tram_eff~0
        _pCRAF_approx = 0.3
        _production   = _k_mek * _pCRAF_approx + _k_mek_braf * 0.0  # BRAF fully blocked
        # Solve quadratic for pMEK_ss: pMEK_ss^2 * k_deg + pMEK_ss * (k_deg * Ks - prod * Ks) - prod * Ks * Ks/(Ks) = 0
        # Simplified: pMEK_ss ≈ prod * Ks_mek / (k_deg * Ks_mek + k_deg * pMEK_approx)
        # Iterative estimate
        _pMEK_est = _Ks_mek * _production / (_k_mek_deg * _Ks_mek)
        LOGGER.info(
            "pMEK SS estimate at init: %.4f "
            "(k_mek=%.3f k_mek_deg=%.3f k_mek_braf=%.3f Ks_mek=%.3f pCRAF_approx=%.2f)",
            _pMEK_est, _k_mek, _k_mek_deg, _k_mek_braf, _Ks_mek, _pCRAF_approx
        )
        # Data range for pMEK is approximately 0.0–0.2 under Vemurafenib
        if _pMEK_est > 0.5:
            LOGGER.warning(
                "pMEK SS estimate (%.4f) is significantly above the expected data range "
                "(0.0–0.2). Consider lowering k_mek or raising k_mek_deg in INITIAL_K_PARAMS.",
                _pMEK_est
            )
        elif _pMEK_est < 0.05:
            LOGGER.warning(
                "pMEK SS estimate (%.4f) is below 0.05 — may be too low. "
                "Check k_mek initialisation.", _pMEK_est
            )
        else:
            LOGGER.info("pMEK SS estimate is within plausible range. OK.")

    # ── BRAF IC50 saturation check ────────────────────────────────────────────
    with torch.no_grad():
        _braf_ic50_eff = float(F.softplus(model.k_params['braf_ic50'].detach()))
        _n_hill = float(torch.clamp(model.k_params['hill_coeff'].detach(), 1.0, 4.0))
        _eps = 1e-7
        LOGGER.info("BRAF IC50 saturation at experimental doses (braf_ic50_eff=%.4f):",
                    _braf_ic50_eff)
        for _dose in [0.0, 0.1, 0.3, 0.5, 1.0]:
            _vb = (_dose + _eps)**_n_hill / (_braf_ic50_eff**_n_hill + (_dose + _eps)**_n_hill)
            _sensitivity = "LOW SENSITIVITY" if _vb > 0.85 else "OK"
            LOGGER.info("  Vem=%.2f  Vem_braf=%.4f  %s", _dose, _vb, _sensitivity)
        if _braf_ic50_eff < 0.1:
            LOGGER.warning(
                "braf_ic50_eff=%.4f is below 0.1 — BRAF channel is saturated "
                "at all experimental doses. Increase braf_ic50 initial value.",
                _braf_ic50_eff
            )

    input_variables  = ["t", "vem", "tram", "pi3k", "ras"]
    output_variables = SPECIES_ORDER

    # Per-species IC weights for Vemurafenib condition.
    # pMEK=0.1: under Vem, pMEK is already suppressed at early times — the early
    #           measurement may reflect pre-drug-equilibration. Do not anchor to it.
    # pERK=1.0: strong anchor — early pERK value establishes the initial suppression.
    # pCRAF=0.3: Vem paradox activation makes early pCRAF variable — weak anchor.
    # DUSP6=0.5: moderate anchor — DUSP6 drops rapidly under pERK suppression.
    # All others=1.0: species not directly in the BRAF→MEK→ERK cascade are reliable.
    VEM_IC_SPECIES_WEIGHTS = {
        "pEGFR":  1.0,
        "HER2":   1.0,
        "HER3":   1.0,
        "IGF1R":  1.0,
        "pCRAF":  0.3,
        "pMEK":   0.1,   # nearly excluded: early pMEK unreliable under Vem
        "pERK":   1.0,
        "DUSP6":  0.5,
        "pAKT":   1.0,
        "p4EBP1": 1.0,
    }

    ic_tensors = build_initial_condition_tensors(
        train_data=train_data,
        scalers=scalers,
        input_variables=input_variables,
        output_variables=output_variables,
        t_threshold_hours=4.0,
        n_replicate=128,
        vem_concentration=ACTUAL_VEM_CONCENTRATION,
        species_ic_weights=VEM_IC_SPECIES_WEIGHTS,
    )

    # ── IC anchor scale verification ──────────────────────────────────────────
    X_ic_check, Y_ic_check = ic_tensors
    y_ic_unorm = Y_ic_check.as_subclass(torch.Tensor)[:1] * scalers["y_range"] + scalers["y_min"]
    LOGGER.info("IC anchor (first replicate, biological scale):")
    for j, sp in enumerate(SPECIES_ORDER):
        val = float(y_ic_unorm[0, j])
        LOGGER.info("  %-12s  %.4f", sp, val)

    pERK_ic_bio = float(y_ic_unorm[0, SPECIES_ORDER.index("pERK")])
    if pERK_ic_bio < 0.3:
        LOGGER.warning(
            "IC anchor pERK=%.4f is very low in biological units. "
            "With IC weight=50, this will strongly pull pERK predictions toward ~%.2f. "
            "Check that t_threshold_hours is capturing the correct early samples.",
            pERK_ic_bio, pERK_ic_bio
        )
    else:
        LOGGER.info(
            "IC anchor pERK=%.4f appears biologically plausible. OK.", pERK_ic_bio
        )

    # ── 6. Build problem ──────────────────────────────────────────────────────
    problem = SignalingProblem(
        train_data=train_data,
        scalers=scalers,
        model=model,
        ic_tensors=ic_tensors,
    )

    # ── 7. Build solver ───────────────────────────────────────────────────────
    # Weight rationale (updated for phase 2 — ODE satisfaction confirmed):
    #   data         (5.0)  — raised from 2.0: model proved it can satisfy ODEs,
    #                          now needs stronger data pull to select correct solution
    #   physics      (8.0)  — lowered from 10.0: ODEs are satisfied, relax slightly
    #                          to give data more relative influence
    #   physics_late (12.0) — lowered from 15.0: consistent with physics reduction
    #   ic           (20.0) — unchanged: t=0 anchor is still critical
    #   steady_state (0.5)  — unchanged: weak basal prior
    #   contrast     (8.0)  — equal to physics: drug sensitivity is as important as ODE satisfaction.
    #                        Without this, the model satisfies ODEs at a drug-insensitive fixed point.
    solver = PINN(
        problem=problem,
        model=model,
        optimizer=TorchOptimizer(torch.optim.Adam, lr=learning_rate),
        # Weight rationale (rebound-enabling regime):
        #   physics_late (35.0) — raised from 25.0: the late-time ODE window (t=24–48h)
        #                          is where the rebound must occur. Higher weight forces
        #                          the model to satisfy RTK→pCRAF→pMEK→pERK at those times.
        #   contrast (8.0)      — reduced from 12.0: restricted to t<12h, so total
        #                          contrast gradient signal is smaller; reduce weight
        #                          to prevent it from dominating early-time training.
        #   mek_vem_ss (5.0)    — slightly reduced: pMEK fixing is now partially handled
        #                          by the corrected k_craf/k_mek_deg ratio.
        #   All others unchanged.
        weighting=ScalarWeighting(weights={
            'data':         20.0,
            'physics':      20.0,
            'physics_late': 35.0,
            'steady_state': 1.0,
            'mek_vem_ss':   5.0,
            'ic':           30.0,
            'contrast':     8.0,
        }),
    )

    # Wire solver reference into problem so drug_contrast_loss can call forward()
    problem._solver_ref = solver
    LOGGER.info("Solver reference wired into problem for drug contrast loss.")

    # ── 8. Callbacks ──────────────────────────────────────────────────────────
    snapshot_dir = f"optimization_snapshots_{condition_tag}"
    optimization_snapshot_cb = OptimizationSnapshotCallback(
        model=model,
        solver=solver,
        problem=problem,
        train_data=train_data,
        scalers=scalers,
        every_n_epochs=50,
        max_snapshots=100,
        output_dir=snapshot_dir,
    )

    val_cb = ValidationCallback(solver, val_data, scalers)
    resample_cb = CollocationResampleCallback(problem)
    lr_decay_cb = LRDecayCallback(solver)

    unfreeze_ic50_cb = UnfreezeIC50Callback(model=model, unfreeze_epoch=IC50_FREEZE_EPOCHS)

    trainer = Trainer(
        solver=solver,
        max_epochs=max_epochs,
        accelerator="cpu",
        callbacks=[
            MetricTracker(),
            optimization_snapshot_cb,
            unfreeze_ic50_cb,
            val_cb,
            resample_cb,
            lr_decay_cb
        ],
        enable_model_summary=False,
        gradient_clip_val=1.0,
    )

    LOGGER.info("Training for %d epochs on condition: %s", max_epochs, TARGET_CONDITION)
    trainer.train()

    # Log IC50 values after training — check for drift
    LOGGER.info("=" * 50)
    ic50_after_training = log_ic50_values(model, label="POST-TRAIN")
    LOGGER.info("=" * 50)

    # ── 13. Initialize Run summary ─────────────────────────────────────────────
    # Moved earlier to avoid NameError when logging diagnostics
    run_summary = {
        "condition":                TARGET_CONDITION,
        "vem_concentration":        VEM_CONCENTRATION,
        "seed":                     seed,
        "split_mode":               split_mode,
        "train_until_hour":         train_until_hour,
        "normalization_mode":       normalization_mode,
        "optimizer":                "Adam",
        "learning_rate":            learning_rate,
        "max_epochs":               max_epochs,
        "train_samples":            int(len(train_data["t"])),
        "ic50_freeze_epochs":       IC50_FREEZE_EPOCHS,
        "loss_weights": {
            "data":         20.0,
            "physics":      20.0,
            "physics_late": 25.0,
            "steady_state": 1.0,
            "mek_vem_ss":   8.0,
            "ic":           30.0,
            "contrast":     12.0,
        },
    }

    # ── Contrast loss diagnostic after training ───────────────────────────────
    with torch.no_grad():
        contrast_final = problem.drug_contrast_loss(
            solver=solver, n_times=100, t_range=(0.05, 0.70)
        )
    LOGGER.info(
        "Drug contrast loss after training: %.6f "
        "(0.0 = perfect drug sensitivity, >0.1 = pERK still drug-insensitive)",
        contrast_final.item()
    )
    run_summary["contrast_loss_final"] = round(float(contrast_final.item()), 6)
    run_summary["ic50_after_training"] = ic50_after_training

    # ── 9. Physics residual diagnostic ───────────────────────────────────────
    LOGGER.info("Computing physics residual on Vem collocation points...")
    phys_input   = problem._conditions['physics'].input
    phys_input_g = phys_input.clone().detach().requires_grad_(True)
    phys_pred_g  = solver.forward(phys_input_g)
    phys_residual = problem.signaling_odes(phys_input_g, phys_pred_g)
    phys_res_tensor = phys_residual.as_subclass(torch.Tensor)
    phys_mse_per_species = (phys_res_tensor ** 2).mean(dim=0).detach().cpu()
    phys_mse = phys_mse_per_species.mean()
    phys_mse = torch.nan_to_num(phys_mse, nan=0.0)

    LOGGER.info("Physics residual MSE (mean across species): %.6f", phys_mse.item())
    LOGGER.info("Per-species physics residual MSE:")
    for species, mse_val in zip(SPECIES_ORDER, phys_mse_per_species.tolist()):
        LOGGER.info("  %-12s  %.6f", species, mse_val)

    del phys_input_g, phys_pred_g, phys_residual, phys_res_tensor

    # ── 10. Evaluation ────────────────────────────────────────────────────────
    def _count(data, key):
        v = data.get(key)
        if v is None: return 0
        if isinstance(v, torch.Tensor): return v.shape[0]
        return len(v)

    has_test = _count(test_data, "t") > 0
    eval_data   = test_data if has_test else train_data
    eval_prefix = "Test" if has_test else "Train (Fit)"
    LOGGER.info("Evaluating on %s set (%d samples)", eval_prefix, len(eval_data["t"]))

    t_eval = torch.tensor(eval_data["t_norm"], dtype=torch.float32)
    d_eval = torch.tensor(eval_data["drugs"],  dtype=torch.float32)
    X_eval = LabelTensor(
        torch.cat([t_eval, d_eval], dim=1),
        ["t", "vem", "tram", "pi3k", "ras"],
    )

    with torch.no_grad():
        predictions = solver.forward(X_eval)

    y_pred_norm = predictions.as_subclass(torch.Tensor)
    y_true_norm = torch.tensor(eval_data["y_norm"], dtype=torch.float32)

    eval_mse = nn.MSELoss()(y_pred_norm, y_true_norm)
    assert not torch.isnan(eval_mse), f"NaN in normalised {eval_prefix} MSE"
    LOGGER.info("%s MSE (normalised):         %.6f", eval_prefix, eval_mse.item())

    y_pred = y_pred_norm * scalers["y_range"] + scalers["y_min"]
    y_true = y_true_norm * scalers["y_range"] + scalers["y_min"]

    true_mse = nn.MSELoss()(y_pred, y_true)
    assert not torch.isnan(true_mse), f"NaN in un-normalised {eval_prefix} MSE"
    LOGGER.info("%s MSE (biological scale):   %.6f", eval_prefix, true_mse.item())

    # ── 11. Per-species eval MSE ──────────────────────────────────────────────
    per_species_mse = ((y_pred - y_true) ** 2).mean(dim=0)
    LOGGER.info("Per-species eval MSE (biological scale):")
    for species, mse_val in zip(SPECIES_ORDER, per_species_mse.tolist()):
        LOGGER.info("  %-12s  %.6f", species, mse_val)

    # ── 12. Detailed metrics CSV ──────────────────────────────────────────────
    metrics_rows = compute_detailed_metrics(
        y_true.detach().cpu().numpy(),
        y_pred.detach().cpu().numpy(),
        eval_data["t"],
        eval_data["condition"],
    )
    metrics_path = f"detailed_metrics_{condition_tag}.csv"
    save_metrics_csv(metrics_rows, metrics_path)
    LOGGER.info("Saved detailed metrics → %s", metrics_path)

    # ── ODE satisfaction at unseen time points ────────────────────────────────────
    LOGGER.info("ODE satisfaction check at unseen time points...")
    t_dense = torch.linspace(0.01, 0.99, 500).unsqueeze(1)
    drugs_dense = torch.full((500, 4), 0.0)
    drugs_dense[:, 0] = ACTUAL_VEM_CONCENTRATION
    X_dense_raw = torch.cat([t_dense, drugs_dense], dim=1)
    X_dense = LabelTensor(X_dense_raw, ["t", "vem", "tram", "pi3k", "ras"])
    X_dense_g = X_dense.clone().detach().requires_grad_(True)

    pred_dense = solver.forward(X_dense_g)
    res_dense = problem.signaling_odes(X_dense_g, pred_dense)
    res_tensor = res_dense.as_subclass(torch.Tensor)
    ode_mse_unseen = (res_tensor ** 2).mean(dim=0).detach().cpu()

    LOGGER.info("ODE residual MSE at 500 UNSEEN time points (per species):")
    for species, mse_val in zip(SPECIES_ORDER, ode_mse_unseen.tolist()):
        flag = " *** HIGH ***" if mse_val > 0.1 else ""
        LOGGER.info("  %-12s  %.6f%s", species, mse_val, flag)

    # ── Drug sensitivity direction check ─────────────────────────────────────────
    LOGGER.info("Drug sensitivity direction check...")
    t_mid = torch.tensor([[0.25]], dtype=torch.float32)
    doses = [0.0, 0.1, 0.3, 0.5, 1.0]
    pERK_idx  = SPECIES_ORDER.index("pERK")
    pCRAF_idx = SPECIES_ORDER.index("pCRAF")

    pERK_at_doses  = []
    pCRAF_at_doses = []

    with torch.no_grad():
        for dose in doses:
            x_test = LabelTensor(
                torch.tensor([[0.25, dose, 0.0, 0.0, 0.0]], dtype=torch.float32),
                ["t", "vem", "tram", "pi3k", "ras"]
            )
            pred = solver.forward(x_test).as_subclass(torch.Tensor)
            pred_unorm = pred * scalers["y_range"] + scalers["y_min"]
            pERK_at_doses.append(float(pred_unorm[0, pERK_idx]))
            pCRAF_at_doses.append(float(pred_unorm[0, pCRAF_idx]))

    LOGGER.info("Drug sensitivity at t=12h:")
    LOGGER.info("  Dose (Vem) | pERK  | pCRAF")
    for dose, erk, craf in zip(doses, pERK_at_doses, pCRAF_at_doses):
        LOGGER.info("  %.2f       | %.3f | %.3f", dose, erk, craf)

    perk_suppression_pct = (pERK_at_doses[0] - pERK_at_doses[-1]) / (pERK_at_doses[0] + 1e-8) * 100

    if perk_suppression_pct >= 40.0:
        LOGGER.info(
            "PASS: pERK suppressed by %.1f%% at max dose (expected >= 40%%).",
            perk_suppression_pct
        )
    elif perk_suppression_pct >= 10.0:
        LOGGER.warning(
            "PARTIAL: pERK suppressed by only %.1f%% at max dose (expected >= 40%%). "
            "IC50_vem may still be elevated. Check ic50_after_training in run_summary.",
            perk_suppression_pct
        )
    else:
        LOGGER.warning(
            "FAIL: pERK suppressed by only %.1f%% at max dose (expected >= 40%%). "
            "Trivial solution likely persists. IC50_vem has probably drifted above 2.0. "
            "Consider increasing IC50_FREEZE_EPOCHS to 1500 or raising ic50_prior penalty.",
            perk_suppression_pct
        )

    run_summary.update({
        "test_samples":             int(len(test_data["t"])) if has_test else 0,
        "eval_mse_normalized":      float(eval_mse.item()),
        "eval_mse_biological_scale":float(true_mse.item()),
        "phys_residual_mse":        float(phys_mse.item()),
        "phys_residual_per_species": dict(zip(
            SPECIES_ORDER, [round(v, 6) for v in phys_mse_per_species.tolist()]
        )),
        "ode_mse_unseen_time_points": dict(zip(
            SPECIES_ORDER, [round(v, 6) for v in ode_mse_unseen.tolist()]
        )),
        "drug_sensitivity_check": {
            "doses":                    doses,
            "pERK_at_t12h":             [round(v, 4) for v in pERK_at_doses],
            "pCRAF_at_t12h":            [round(v, 4) for v in pCRAF_at_doses],
            "pERK_suppression_pct":     round(float(perk_suppression_pct), 2),
            "pERK_suppression_pass":    bool(perk_suppression_pct >= 40.0),
        },
    })

    # ── Absolute pERK amplitude check ─────────────────────────────────────────
    # pERK at t=1h (normalised t ≈ 0.021) under Vemurafenib should be in the range
    # [0.5, 4.0] biological units based on the training data.
    pERK_idx_s = SPECIES_ORDER.index("pERK")
    with torch.no_grad():
        x_early = LabelTensor(
            torch.tensor([[0.021, ACTUAL_VEM_CONCENTRATION, 0.0, 0.0, 0.0]],
                         dtype=torch.float32),
            ["t", "vem", "tram", "pi3k", "ras"]
        )
        pred_early = solver.forward(x_early).as_subclass(torch.Tensor)
        pred_early_unorm = pred_early * scalers["y_range"] + scalers["y_min"]
        pERK_at_t1h = float(pred_early_unorm[0, pERK_idx_s])

    LOGGER.info("pERK absolute amplitude at t=1h (biological units): %.4f", pERK_at_t1h)
    if pERK_at_t1h < 0.5:
        LOGGER.warning(
            "pERK amplitude at t=1h is %.4f — FAR BELOW expected range [0.5, 4.0]. "
            "Model has collapsed to near-zero. Increase data/ic weights or "
            "reduce physics weight. Current weights: data=50 ic=50 physics=5.",
            pERK_at_t1h
        )
    elif pERK_at_t1h > 4.0:
        LOGGER.warning(
            "pERK amplitude at t=1h is %.4f — ABOVE expected range [0.5, 4.0]. "
            "Model is overpredicting early pERK.", pERK_at_t1h
        )
    else:
        LOGGER.info(
            "PASS: pERK amplitude at t=1h (%.4f) is within expected range [0.5, 4.0].",
            pERK_at_t1h
        )
    run_summary["pERK_amplitude_t1h"] = round(pERK_at_t1h, 4)
    run_summary["pERK_amplitude_pass"] = bool(0.5 <= pERK_at_t1h <= 4.0)

    # ── Late-time pERK rebound check ─────────────────────────────────────────
    # Under Vemurafenib, adaptive resistance should cause pERK to recover
    # to at least 50% of its early value by t=48h.
    # Normalised t=48h = 1.0, t=1h ≈ 0.021
    with torch.no_grad():
        x_late_check = LabelTensor(
            torch.tensor([[1.0, ACTUAL_VEM_CONCENTRATION, 0.0, 0.0, 0.0]],
                         dtype=torch.float32),
            ["t", "vem", "tram", "pi3k", "ras"]
        )
        pred_late = solver.forward(x_late_check).as_subclass(torch.Tensor)
        pred_late_unorm = pred_late * scalers["y_range"] + scalers["y_min"]
        pERK_at_t48h = float(pred_late_unorm[0, SPECIES_ORDER.index("pERK")])

    rebound_ratio = pERK_at_t48h / (pERK_at_t1h + 1e-8)
    LOGGER.info(
        "pERK late-time rebound: t=1h=%.4f → t=48h=%.4f (ratio=%.2f)",
        pERK_at_t1h, pERK_at_t48h, rebound_ratio
    )
    if rebound_ratio >= 0.8:
        LOGGER.info(
            "PASS: pERK at t=48h (%.4f) is >= 80%% of t=1h value (%.4f). "
            "Rebound is present.", pERK_at_t48h, pERK_at_t1h
        )
    elif rebound_ratio >= 0.4:
        LOGGER.warning(
            "PARTIAL: pERK rebound ratio=%.2f (target >= 0.8). "
            "Rebound is partial — late-time physics may need more epochs or "
            "higher physics_late weight.", rebound_ratio
        )
    else:
        LOGGER.warning(
            "FAIL: pERK rebound ratio=%.2f (target >= 0.8). "
            "pERK at t=48h (%.4f) is far below early value (%.4f). "
            "Contrast loss time range or physics_late weight may need adjustment.",
            rebound_ratio, pERK_at_t48h, pERK_at_t1h
        )
    run_summary["pERK_t48h"]       = round(pERK_at_t48h, 4)
    run_summary["pERK_rebound_ratio"] = round(float(rebound_ratio), 4)
    run_summary["pERK_rebound_pass"]  = bool(rebound_ratio >= 0.8)
    summary_path = f"run_summary_{condition_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    LOGGER.info("Saved run summary → %s", summary_path)

    # ── 14. Save model ────────────────────────────────────────────────────────
    model_path = f"pina_signaling_model_{condition_tag}.pth"
    torch.save(model.state_dict(), model_path)
    LOGGER.info("Saved model → %s", model_path)
    LOGGER.info("Training complete for condition: %s", TARGET_CONDITION)
