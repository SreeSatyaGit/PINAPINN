# A Hybrid Physics-Informed Neural Network for Cancer Signaling Dynamics

This repository contains the source code and data processing pipelines for a Physics-Informed Neural Network (PINN) designed to model the coupled MAPK and PI3K/AKT signaling pathways in cancer cells. By integrating mechanistic ordinary differential equations (ODEs) with deep learning parameter inference, this model serves as a digital twin capable of predicting temporally extended cellular responses to multi-drug perturbations.

## Repository Contents

*   **`run_pina_model.py`**: The core training script. It defines the multi-layer perceptron (MLP) architecture, the mechanistic ODE residuals (physics loss), and the penalty constraints for basal homeostasis.
*   **`data_utils.py`**: Functions for processing raw time-series biological data (e.g., RPPA/Western Blot), applying min-max normalization, establishing baseline non-negativity constraints, and generating dense collocation grids.
*   **`simulate_drug_combination.py`**: An inference pipeline for the continuous prediction of signaling trajectories under arbitrary, continuous drug concentrations and combinations.
*   **`visualize_predictions.py`**: Evaluation script for plotting model predictions against held-out experimental ground truth and rendering biological dynamics.

## System Requirements

### Hardware Requirements
The training and inference pipelines require standard compute hardware. While a GPU (CUDA-compatible) accelerates the collocation-point evaluation during training, the model can be fully trained and evaluated on a standard multi-core CPU within minutes. 

### Software Requirements
The project is built on Python 3.9+ and relies heavily on PyTorch and the PINA (Physics-Informed Neural networks for Advanced modeling) framework.

*   `torch` >= 2.0.0
*   `pina` (Latest release supporting automatic differentiation operators)
*   `numpy` >= 1.23.0
*   `matplotlib` >= 3.6.0

## Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/SreeSatyaGit/PINAPINN.git
   cd PINAPINN
   ```

2. Establish a Python virtual environment and install the required dependencies:
   ```bash
   python3 -m venv pina_env
   source pina_env/bin/activate
   pip install torch numpy matplotlib
   pip install pina-math
   ```

## Usage and Execution

### 1. Training the Model
To train the PINN from scratch and optimize the coupled kinetic parameters (binding affinities, degradation rates, and drug IC50s):
```bash
python3 run_pina_model.py
```
This script will produce a converged model state dictionary (`pina_signaling_model.pth`), alongside runtime statistics (`run_summary.json`) and specific evaluation metrics across species (`detailed_metrics.csv`).

### 2. Experimental Data Visualization
To compare the predicted signaling profiles against the empirical measurements for predefined combinations (including zero-drug controls and holdout conditions):
```bash
python3 visualize_predictions.py
```
Output figures will trace both normalized input data points and continuous lines representing the integrated differential predictions.

### 3. In-Silico Drug Simulation
To evaluate the model's performance as a digital twin under novel drug concentrations not present in the primary dataset:
```bash
python3 simulate_drug_combination.py
```
Users may modify the `CUSTOM_DOSES` hashmap within this script to test any permutation of BRAF, MEK, PI3K, or pan-RAS inhibitor concentrations. The system will plot side-by-side trajectories of the chosen experimental baseline and the simulated perturbation.

## Data Availability
The synthetic or processed experimental inputs leveraged for the construction of this model are handled internally via the data utilities. Re-parameterization bounds and initialized kinetics are defined explicitly within the network scaffolding in the training script. Detailed datasets corresponding to publications must be requested or accessed via the supplementary materials of the affiliated manuscript.

## License
This source code is made available for academic research and peer review. Any reuse, modification, or redistribution must conform to standard academic citation practices and relevant institutional licensing agreements.
