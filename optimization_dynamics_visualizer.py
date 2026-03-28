import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_utils import SPECIES_ORDER


SPECIES_PARAMETER_MAP = {
    "pEGFR": ["k_egfr", "k_egfr_deg", "K_sat_egfr", "w_egfr"],
    "HER2": ["k_her2", "k_her2_deg", "k_her2_tx", "K_sat_her2", "w_her2"],
    "HER3": ["k_her3", "k_her3_deg", "k_her3_tx", "K_sat_her3", "w_her3"],
    "IGF1R": ["k_igf", "k_igf_deg", "K_sat_igfr", "w_igf1r"],
    "pCRAF": ["k_craf", "k_craf_deg", "k_paradox", "K_sat_craf", "w_craf"],
    "pMEK": ["k_mek", "k_mek_deg", "k_mek_braf", "K_sat_mek", "w_mek"],
    "pERK": ["k_erk", "k_erk_deg", "K_sat_erk", "w_erk"],
    "DUSP6": ["k_dusp_synth", "k_dusp_deg", "k_dusp_cat", "Km_dusp", "w_dusp6"],
    "pAKT": ["k_akt", "k_akt_deg", "K_sat_akt", "w_akt"],
    "p4EBP1": ["k_4ebp1", "k_4ebp1_deg", "k_4ebp1_comp", "K_sat_4ebp1", "w_4ebp1"],
}


def _load_snapshots(history_path: Path):
    snapshots = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))
    snapshots.sort(key=lambda x: x["epoch"])
    return snapshots


def _parameter_summary_per_species(effective_parameters):
    values = []
    for species in SPECIES_ORDER:
        names = SPECIES_PARAMETER_MAP[species]
        vals = [effective_parameters[n] for n in names if n in effective_parameters]
        values.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(values, dtype=np.float32)


def create_dynamic_optimization_figures(
    history_path: str = "optimization_snapshots/snapshot_history.jsonl",
    output_dir: str = "optimization_figures",
    max_figures: int = 10,
):
    history_file = Path(history_path)
    if not history_file.exists():
        raise FileNotFoundError(
            f"Snapshot history not found at '{history_path}'. "
            "Run training first: python3 run_pina_model.py"
        )

    snapshots = _load_snapshots(history_file)[:max_figures]
    if not snapshots:
        raise RuntimeError("No snapshots found to visualize.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, snap in enumerate(snapshots, start=1):
        epoch = snap["epoch"]
        data_mse = np.array(snap["data_mse_per_species"], dtype=np.float32)
        physics_mse = np.array(snap["physics_mse_per_species"], dtype=np.float32)
        param_summary = _parameter_summary_per_species(snap["effective_parameters"])

        heat_data = np.stack([data_mse, physics_mse], axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.25, 1]})
        fig.suptitle(
            f"Optimization Dynamics Snapshot {idx}/10 (Epoch {epoch})",
            fontsize=14,
            fontweight="bold",
        )

        ax0 = axes[0]
        im = ax0.imshow(np.log10(heat_data + 1e-12), cmap="viridis", aspect="auto")
        ax0.set_title("Protein Fit + ODE Residual per Species")
        ax0.set_xticks([0, 1])
        ax0.set_xticklabels(["Data MSE", "Physics MSE"])
        ax0.set_yticks(np.arange(len(SPECIES_ORDER)))
        ax0.set_yticklabels(SPECIES_ORDER)
        cbar = fig.colorbar(im, ax=ax0)
        cbar.set_label("log10(MSE)")

        ax1 = axes[1]
        y_pos = np.arange(len(SPECIES_ORDER))
        ax1.barh(y_pos, param_summary, color="#1f77b4", alpha=0.85)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(SPECIES_ORDER)
        ax1.set_title("Mean Optimized Parameters per Protein")
        ax1.set_xlabel("Mean effective parameter value")
        ax1.invert_yaxis()

        plt.tight_layout()
        fig_path = out_dir / f"optimization_snapshot_{idx:02d}_epoch_{epoch:04d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")

    print(
        f"Created {len(snapshots)} optimization figures in '{out_dir}'. "
        "Each figure corresponds to one 10-epoch snapshot."
    )


if __name__ == "__main__":
    create_dynamic_optimization_figures()
