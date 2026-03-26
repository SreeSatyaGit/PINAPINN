import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pina import LabelTensor
from run_pina_model import SignalingModel, SPECIES_ORDER
from data_utils import prepare_training_tensors

# Configure logging
logging.basicConfig(level=logging.ERROR)

# 1. SETTINGS
# ---------------------------------
# Choose which experimental condition to compare against as a baseline
REFERENCE_CONDITION = "Vem + PI3Ki Combo" 

# Define your custom dosage to see how it differs
CUSTOM_DOSES = {
    'vem':  0.25,   # Half-dose Vemurafenib
    'tram': 0.15,   # Moderate Tram
    'pi3k': 0.10,   # Standard dose PI3Ki
    'ras':  0.00
}

def simulate():
    print(f"Loading data and model for comparison...")
    # Load scaling parameters and data
    train_data, test_data, scalers = prepare_training_tensors(
        split_mode="partial_condition_holdout",
        holdout_condition=REFERENCE_CONDITION,
        partial_condition_train_timepoints=[0.0, 1.0, 4.0],
        normalization_mode="train_only"
    )
    
    # 2. EXTRACT REFERENCE DATA
    # ---------------------------------
    # Check both train/test to find the reference condition samples
    ref_mask_test = test_data['condition'] == REFERENCE_CONDITION
    ref_mask_train = train_data['condition'] == REFERENCE_CONDITION
    
    # Experimental points for plotting dots (Convert scalers to numpy)
    y_range_np = scalers['y_range'].detach().cpu().numpy()
    y_min_np = scalers['y_min'].detach().cpu().numpy()
    
    ref_times = []
    ref_y = []
    
    if any(ref_mask_train):
        ref_times.extend(train_data['t'][ref_mask_train])
        ref_y.extend(train_data['y_norm'][ref_mask_train] * y_range_np + y_min_np)
    if any(ref_mask_test):
        ref_times.extend(test_data['t'][ref_mask_test])
        ref_y.extend(test_data['y_norm'][ref_mask_test] * y_range_np + y_min_np)
    
    ref_times = np.array(ref_times)
    ref_y = np.stack(ref_y) if len(ref_y) > 0 else np.zeros((0, 10))
    
    # Extract reference drug concentration (use the one from train if possible)
    if any(ref_mask_train):
        idx = np.where(ref_mask_train)[0][0]
        ref_drugs = train_data['drugs'][idx]
    else:
        idx = np.where(ref_mask_test)[0][0]
        ref_drugs = test_data['drugs'][idx]
    
    # 3. PREDICT TRAJECTORIES
    # ---------------------------------
    model = SignalingModel()
    model.load_state_dict(torch.load("pina_signaling_model.pth"))
    model.eval()

    t_hours = np.linspace(0, 48, 200)
    t_norm = t_hours / scalers['t_range']
    
    def get_traj(d):
        inputs = np.zeros((len(t_hours), 5))
        inputs[:, 0] = t_norm
        inputs[:, 1:] = d
        X = LabelTensor(torch.tensor(inputs, dtype=torch.float32), 
                         ['t', 'vem', 'tram', 'pi3k', 'ras'])
        with torch.no_grad():
            out = model(X).as_subclass(torch.Tensor)
        return out * scalers['y_range'] + scalers['y_min']

    y_ref_traj = get_traj(ref_drugs)
    y_custom_traj = get_traj([CUSTOM_DOSES['vem'], CUSTOM_DOSES['tram'], CUSTOM_DOSES['pi3k'], CUSTOM_DOSES['ras']])

    # 4. PLOTTING
    # ---------------------------------
    print(f"Generating comparison plot...")
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()
    
    ref_label = f"REF: {REFERENCE_CONDITION}"
    cust_label = f"CUSTOM: {CUSTOM_DOSES}"
    
    for i, species in enumerate(SPECIES_ORDER):
        ax = axes[i]
        
        # Experimental Dots
        if len(ref_times) > 0:
            ax.scatter(ref_times, ref_y[:, i], color='black', marker='o', s=80, 
                       label='Exp. (Standard)', zorder=5, alpha=0.8)
        
        # Standard Prediction Trajectory
        ax.plot(t_hours, y_ref_traj[:, i], color='black', ls='--', lw=2, 
                label='PINN (Standard)', alpha=0.5)
        
        # Custom Prediction Trajectory
        ax.plot(t_hours, y_custom_traj[:, i], color='#27ae60', lw=4, 
                label='PINN (Custom Dose)', zorder=4)
        
        ax.set_title(species, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (h)", fontsize=10)
        ax.set_ylim(bottom=0)
        
        # Auto-scale y-axis
        y_max = max(y_ref_traj[:, i].max(), y_custom_traj[:, i].max(), 
                    ref_y[:, i].max() if len(ref_y) > 0 else 0) * 1.5
        ax.set_ylim(top=max(y_max, 0.5))
        
        ax.grid(alpha=0.2)
        if i == 0 or i == 5:
            ax.set_ylabel("Expression Level")
        
        if i == 4: # Top right
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)

    plt.suptitle(f"In-Silico Drug Response Comparison\nCUSTOM ({CUSTOM_DOSES}) vs {REFERENCE_CONDITION}", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    save_path = "simulated_drug_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"SUCCESS! Comparison plot saved to -> {save_path}")

if __name__ == "__main__":
    simulate()
