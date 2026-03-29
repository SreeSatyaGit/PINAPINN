import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
from data_utils import prepare_training_tensors, SPECIES_ORDER
import re

def softplus(x: float) -> float:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.log1p(np.exp(-abs(x))) + max(x, 0)

def effective_params(raw: dict) -> dict:
    """Convert raw parameter values to effective values matching PINN initialisation."""
    out = {}
    for key, v in raw.items():
        if key in {"hill_coeff", "n_dusp"}:
            out[key] = float(np.clip(v, 1.0, 4.0))
        else:
            out[key] = softplus(v)
    return out

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

def signaling_odes(t: float, y: np.ndarray, k: dict, drugs: dict) -> np.ndarray:
    eps = 1e-7

    y = np.maximum(y, 0.0)
    pEGFR, HER2, HER3, IGF1R, pCRAF, pMEK, pERK, DUSP6, pAKT, p4EBP1 = y

    Vem = abs(drugs['vem'])
    Tram = abs(drugs['tram'])
    PI3Ki = abs(drugs['pi3k'])
    RasInh = abs(drugs['ras'])

    n = np.clip(k['hill_coeff'], 1.0, 4.0)

    Vem_inh = (Vem + eps)**n / (k['IC50_vem']**n + (Vem + eps)**n + 1e-8)
    Tram_eff = (Tram + eps)**n / (k['IC50_tram']**n + (Tram + eps)**n + 1e-8)
    PI3Ki_eff = (PI3Ki + eps)**n / (k['IC50_pi3k']**n + (PI3Ki + eps)**n + 1e-8)
    Ras_eff = (RasInh + eps)**n / (k['IC50_ras']**n + (RasInh + eps)**n + 1e-8)

    Ks_egfr = k['K_sat_egfr']
    Ks_her2 = k['K_sat_her2']
    Ks_her3 = k['K_sat_her3']
    Ks_igfr = k['K_sat_igfr']
    Ks_craf = k['K_sat_craf']
    Ks_mek = k['K_sat_mek']
    Ks_erk = k['K_sat_erk']
    Ks_akt = k['K_sat_akt']
    Ks_4ebp1 = k['K_sat_4ebp1']

    k_erk_rtk = k['k_erk_rtk']
    Km_rtk = k['Km_rtk']
    ERK_feedback = k_erk_rtk * pERK / (Km_rtk + pERK + 1e-8)

    k_up = k['k_up']
    drug_relief = k_up * (Vem_inh + Tram_eff + PI3Ki_eff)

    k_erk_sos = k['k_erk_sos']
    k_akt_rtk = k['k_akt_rtk']
    Km_sos = k['Km_sos']
    Km_artk = k['Km_artk']
    ERK_to_SOS = k_erk_sos * pERK / (Km_sos + pERK + 1e-8)
    AKT_to_RTK = k_akt_rtk * pAKT / (Km_artk + pAKT + 1e-8)

    RTK_total = pEGFR + HER2 + 1.5 * HER3 + IGF1R
    RAS_GTP = RTK_total * (1.0 - ERK_to_SOS) * (1.0 - AKT_to_RTK) * (1.0 - Ras_eff)

    k_raf_pi3k = k['k_raf_pi3k']
    Km_raf_pi3k = k['Km_raf_pi3k']
    k_erk_pi3k = k['k_erk_pi3k']
    Km_erk_pi3k = k['Km_erk_pi3k']
    RAF_to_PI3K = k_raf_pi3k * pCRAF / (Km_raf_pi3k + pCRAF + 1e-8)
    ERK_to_PI3K = k_erk_pi3k * pERK / (Km_erk_pi3k + pERK + 1e-8)

    k_ras_frac = k['k_ras_pi3k_frac']
    PI3K_input = RTK_total * (1.0 - ERK_to_PI3K) * (1.0 - k_ras_frac * Ras_eff) + RAF_to_PI3K

    k_akt_raf = k['k_akt_raf']
    Km_akt_raf = k['Km_akt_raf']
    AKT_RAF_inhib = k_akt_raf * pAKT / (Km_akt_raf + pAKT + 1e-8)

    k_paradox = k['k_paradox']
    pCRAF_fl = max(pCRAF, 0.05)
    pi3ki_att = 1.0 - 0.7 * PI3Ki_eff
    Vem_paradox = k_paradox * Vem * Ks_craf / (Ks_craf + pCRAF_fl + 1e-8) * pi3ki_att

    ke = k['k_egfr']
    ked = k['k_egfr_deg']
    dydt0 = ke * (1.0 + drug_relief) * Ks_egfr / (Ks_egfr + pEGFR + 1e-8) - (ked + ERK_feedback) * pEGFR

    kh2 = k['k_her2']
    kh2d = k['k_her2_deg']
    kh2tx = k['k_her2_tx']
    dydt1 = kh2 * Ks_her2 / (Ks_her2 + HER2 + 1e-8) + kh2tx * (1.0 - pERK / (Ks_erk + pERK + 1e-8)) - kh2d * HER2

    kh3 = k['k_her3']
    kh3d = k['k_her3_deg']
    kh3tx = k['k_her3_tx']
    dydt2 = kh3 * Ks_her3 / (Ks_her3 + HER3 + 1e-8) + kh3tx * (1.0 - pERK / (Ks_erk + pERK + 1e-8)) - kh3d * HER3

    ki = k['k_igf']
    kid = k['k_igf_deg']
    dydt3 = ki * Ks_igfr / (Ks_igfr + IGF1R + 1e-8) - (kid + ERK_feedback + AKT_to_RTK) * IGF1R

    kc = k['k_craf']
    kcd = k['k_craf_deg']
    dydt4 = kc * RAS_GTP * Ks_craf / (Ks_craf + pCRAF + 1e-8) + Vem_paradox - (kcd + AKT_RAF_inhib) * pCRAF

    km = k['k_mek']
    kmd = k['k_mek_deg']
    kmb = k['k_mek_braf']
    braf_ic50 = 0.04
    Vem_braf = (Vem + eps)**n / (braf_ic50**n + (Vem + eps)**n + 1e-8)
    dydt5 = (km * pCRAF + kmb * (1.0 - Vem_braf)) * (1.0 - Tram_eff) * Ks_mek / (Ks_mek + pMEK + 1e-8) - kmd * pMEK

    kdusp_cat = k['k_dusp_cat']
    Km_dusp = k['Km_dusp']
    DUSP_act = kdusp_cat * DUSP6 / (Km_dusp + DUSP6 + 1e-8)
    ker = k['k_erk']
    kerd = k['k_erk_deg']
    dydt6 = ker * pMEK * Ks_erk / (Ks_erk + pERK + 1e-8) - (kerd + DUSP_act) * pERK

    n_dusp = np.clip(k['n_dusp'], 1.5, 3.5)
    kds = k['k_dusp_synth']
    Km_ds = k['Km_dusp_s']
    kdd = k['k_dusp_deg']
    DUSP_ind = kds * (pERK + eps)**n_dusp / (Km_ds**n_dusp + (pERK + eps)**n_dusp + 1e-8)
    dydt7 = DUSP_ind - kdd * DUSP6

    ka = k['k_akt']
    kad = k['k_akt_deg']
    dydt8 = ka * PI3K_input * (1.0 - PI3Ki_eff) * Ks_akt / (Ks_akt + pAKT + 1e-8) - kad * pAKT

    k4 = k['k_4ebp1']
    k4d = k['k_4ebp1_deg']
    k4b = k['k_4ebp1_comp']
    dydt9 = k4 * pAKT * Ks_4ebp1 / (Ks_4ebp1 + p4EBP1 + 1e-8) + k4b - k4d * p4EBP1

    dydt = np.array([dydt0, dydt1, dydt2, dydt3, dydt4, dydt5, dydt6, dydt7, dydt8, dydt9])
    return dydt

CONDITIONS = [
    {"name": "No Drug (Basal)",         "vem": 0.0, "tram": 0.0, "pi3k": 0.0, "ras": 0.0},
    {"name": "Vemurafenib Only (0.5)",  "vem": 0.5, "tram": 0.0, "pi3k": 0.0, "ras": 0.0},
    {"name": "Trametinib Only (0.3)",   "vem": 0.0, "tram": 0.3, "pi3k": 0.0, "ras": 0.0},
    {"name": "Vem + Tram Combo",        "vem": 0.5, "tram": 0.3, "pi3k": 0.0, "ras": 0.0},
    {"name": "Vem + PI3Ki Combo",       "vem": 0.5, "tram": 0.0, "pi3k": 1.0, "ras": 0.0},
    {"name": "Vem + panRAS Combo",      "vem": 0.5, "tram": 0.0, "pi3k": 0.0, "ras": 1.0},
]

def get_initial_conditions(train_data: dict, t_threshold: float = 2.0) -> np.ndarray:
    """
    Compute mean species values at the earliest time points as initial conditions.
    Returns array of shape (10,) in SPECIES_ORDER.
    """
    t = np.array(train_data["t"]).squeeze()
    y_raw = np.array(train_data["y_raw"])   # shape (N, 10), un-normalised
    mask = t <= t_threshold
    if mask.sum() == 0:
        raise ValueError(f"No samples found at t <= {t_threshold}h")
    return y_raw[mask].mean(axis=0)

def integrate_condition(condition: dict, y0: np.ndarray, k: dict) -> tuple:
    """Returns (t_array, y_array) or (None, None) on failure."""
    drugs = {key: condition[key] for key in ("vem", "tram", "pi3k", "ras")}
    try:
        sol = solve_ivp(
            fun=lambda t, y: signaling_odes(t, y, k, drugs),
            t_span=(0.0, 50.0),
            y0=y0,
            method="LSODA",
            t_eval=np.linspace(0, 50, 300),
            rtol=1e-6,
            atol=1e-9,
        )
        if sol.success:
            return sol.t, sol.y.T   # shape (300, 10)
        else:
            print(f"WARNING: Integration failed for {condition['name']}: {sol.message}")
            return None, None
    except Exception as e:
        print(f"WARNING: Exception for {condition['name']}: {e}")
        return None, None

def match_condition(data_condition_str: str, condition_name: str) -> bool:
    """Loose match: check if key drug tokens appear in the data condition string."""
    dc = data_condition_str.lower()
    name = condition_name.lower()
    if "no drug" in name or "basal" in name:
        return "no drug" in dc or "basal" in dc or "dmso" in dc
    tokens = [t.strip() for t in name.replace("+", " ").replace("(", " ").replace(")", " ").split() if len(t) > 2]
    return any(tok in dc for tok in tokens)

if __name__ == "__main__":
    # 1. Load data
    train_data, test_data, scalers = prepare_training_tensors(
        split_mode="cutoff", train_until_hour=48.0, normalization_mode="train_only"
    )

    if "y_raw" in train_data:
        y_raw_train = train_data["y_raw"]
    else:
        y_raw_train = train_data["y_norm"] * scalers["y_range"].numpy() + scalers["y_min"].numpy()

    # 2. Compute effective parameters
    k_eff = effective_params(INITIAL_K_PARAMS)

    # 3. Compute initial conditions
    y0 = get_initial_conditions(train_data)
    print(f"Initial conditions (t<=2h mean): {dict(zip(SPECIES_ORDER, y0.round(3)))}")

    # 4. Integrate all conditions and plot
    Path("ode_baseline_plots").mkdir(exist_ok=True)

    results_summary = []

    for cond in CONDITIONS:
        t_arr, y_arr = integrate_condition(cond, y0, k_eff)
        
        status = "SUCCESS" if y_arr is not None else "FAILED"
        if status == "SUCCESS":
            results_summary.append({
                "name": cond["name"],
                "status": status,
                "t_final": t_arr[-1],
                "y_final": y_arr[-1],
                "pred": {sp: y_arr[-1, i] for i, sp in enumerate(SPECIES_ORDER)}
            })
        else:
            results_summary.append({
                "name": cond["name"],
                "status": status,
                "t_final": 0.0,
                "y_final": None,
                "pred": None
            })

        if y_arr is not None:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()

            # Find matching data points
            data_t = []
            data_y = []
            if "condition" in train_data:
                for dc, dt, dy in zip(train_data["condition"], train_data["t"], y_raw_train):
                    if match_condition(dc, cond["name"]):
                        data_t.append(dt)
                        data_y.append(dy)
            
            data_t = np.array(data_t)
            if len(data_y) > 0:
                data_y = np.stack(data_y)
            else:
                data_y = np.array([])
                
            for i, sp in enumerate(SPECIES_ORDER):
                ax = axes[i]
                ax.plot(t_arr, y_arr[:, i], 'b-', label='ODE')
                if data_t.size > 0:
                    ax.plot(data_t, data_y[:, i], 'go', label='Train Data')
                ax.set_title(sp)
                ax.set_xlabel("Time (hours)")
                ax.set_ylabel("Expression Level")
                if i == 0:
                    ax.legend()
            
            plt.suptitle(cond["name"])
            plt.tight_layout()
            
            slug = re.sub(r'[^a-zA-Z0-9]', '_', cond["name"]).strip('_')
            slug = re.sub(r'_+', '_', slug)
            fig.savefig(f"ode_baseline_plots/baseline_{slug}.png", dpi=150)
            plt.close(fig)

    # 5. Print summary table
    print("\n=== ODE Baseline Diagnostic Summary ===\n")
    print("Effective parameter values (post-softplus):")
    for key, val in k_eff.items():
        print(f"  {key:<12} = {val:.4f}")
    
    print("\nIntegration results:")
    for res in results_summary:
        if res["status"] == "SUCCESS":
            y_final_str = np.array2string(res["y_final"][:5], precision=2, separator=', ')
            y_final_str = y_final_str.replace("[", "[...").replace("]", "...]")
            print(f"  {res['name']:<25} {res['status']}  t_final={res['t_final']}  y_final={y_final_str}")
        else:
            print(f"  {res['name']:<25} {res['status']}")
            
    # Baseline comparison at t=48h for No Drug
    print("\nSpecies steady-state comparison (no drug, t=48h):")
    print(f"  {'Species':<11} {'ODE_pred':<10} {'Data_mean':<10} {'Ratio'}")
    
    no_drug_res = next((r for r in results_summary if "no drug" in r["name"].lower() and r["status"] == "SUCCESS"), None)
    
    if no_drug_res is not None:
        data_t = train_data["t"].flatten()
        # Find 48h data
        mask_48h = (data_t >= 47.0) & (data_t <= 50.0)
        
        # Need to specifically get No Drug data at 48h or closest to steady state
        data_y_48h = None
        for r_name in np.unique(train_data["condition"]):
            if match_condition(r_name, "No Drug (Basal)"):
                mask_cond = train_data["condition"] == r_name
                if (mask_cond & mask_48h).sum() > 0:
                    data_y_48h = y_raw_train[mask_cond & mask_48h].mean(axis=0)
                elif mask_cond.sum() > 0:
                    data_y_48h = y_raw_train[mask_cond].mean(axis=0)
                break
                
        if data_y_48h is not None:
            for i, sp in enumerate(SPECIES_ORDER):
                pred_val = no_drug_res["pred"][sp]
                data_val = data_y_48h[i]
                ratio = pred_val / data_val if data_val > 1e-6 else np.inf
                print(f"  {sp:<11} {pred_val:<10.3f} {data_val:<10.3f} {ratio:<10.3f}")
        else:
            print("  [Could not extract 48h baseline data]")
