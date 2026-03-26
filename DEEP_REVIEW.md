# Deep Repository Review

## Scope & Method
- Reviewed all Python source files in the repository (`run_pina_model.py`, `data_utils.py`, `physics_utils.py`, `visualize_predictions.py`).
- Focused on: correctness, reproducibility, maintainability, evaluation hygiene, and operational risk.
- Did **not** execute long model training (1000 epochs) due runtime cost; review is static + lightweight checks.

## Executive Summary
This repository has a strong core model implementation, but there are several medium/high-impact issues that may affect scientific reliability and maintainability:
1. A stale/unused physics module with a model API mismatch.
2. Split-mode validation that silently accepts invalid values.
3. Evaluation normalization computed from full dataset (documented as intentional, but risky for strict holdout benchmarking).
4. Reproducibility gaps (no fixed seeds while random sampling is central).
5. Signal suppression patterns and hygiene issues (global warning suppression, duplicate imports, unused locals).

---

## Findings

### 1) Stale physics module with incompatible model signature (**High**)
- `physics_utils.compute_physics_loss` expects `model(t_physics, drugs)` with two arguments.
- The active model in `run_pina_model.py` defines `forward(self, x)` (single concatenated input tensor).
- `physics_utils.py` is not referenced by the training script.

**Why it matters:**
- Any future attempt to reuse `physics_utils.py` will fail at runtime.
- This creates parallel logic paths and encourages drift between “real” and “assumed” model contracts.

**Evidence:**
- `physics_utils.py` call-site expectation: `y_pred_norm = model(t_physics, drugs)`.
- `run_pina_model.py` model contract: `def forward(self, x)`.

**Recommendation:**
- Either delete/archive `physics_utils.py` if unused, or refactor to the active model interface and wire it into training/tests.

### 2) Invalid `split_mode` values are silently accepted (**High**)
- In `prepare_training_tensors`, any unrecognized `split_mode` falls through to a default branch (`t <= train_until_hour`) instead of raising.

**Why it matters:**
- Typos (e.g., `split_mode="holdot"`) silently produce a different split than intended, creating hard-to-detect experimental errors.

**Evidence:**
- `data_utils.py` uses an `else:` fallback branch after explicit mode checks.

**Recommendation:**
- Replace fallback with explicit `elif split_mode == "cutoff"` and final `raise ValueError` for unknown modes.

### 3) Full-dataset normalization for holdout evaluation (**Medium/High**)
- `y_min`/`y_max` are computed over **all** rows (`y_data`) before applying train/test masks.

**Why it matters:**
- The code comments justify this as not being label leakage, but for strict benchmarking this is still information sharing from test distribution.
- Can inflate evaluation comparability versus train-only fit transforms.

**Evidence:**
- `data_utils.py` computes `y_min = np.min(y_data, axis=0)` and `y_max = np.max(y_data, axis=0)` before `package_data(train_mask/test_mask)`.

**Recommendation:**
- Add a parameter to choose scaler policy (`"train_only"` vs `"global"`) and log policy in outputs.

### 4) No deterministic seeding despite stochastic sampling (**Medium**)
- Collocation generation relies on random draws (`uniform`, `choice`, `normal`, `permutation`).
- No seeds are set in training entrypoint.

**Why it matters:**
- Training outcomes may vary significantly between runs, limiting scientific reproducibility and debugability.

**Evidence:**
- `data_utils.py` random calls in `get_collocation_points`.
- `run_pina_model.py` main block does not set NumPy/Torch seeds.

**Recommendation:**
- Add deterministic seed controls (NumPy + PyTorch + optional deterministic backend flags).

### 5) Global warning suppression masks diagnostics (**Medium**)
- `warnings.filterwarnings("ignore")` suppresses all warnings process-wide.

**Why it matters:**
- Can hide numerical instability, deprecations, and framework warnings needed during model development.

**Evidence:**
- Global warning suppression at import-time in `run_pina_model.py`.

**Recommendation:**
- Remove global suppression; if needed, target specific warning categories/messages locally.

### 6) Duplicate import in visualization script (**Low**)
- `matplotlib.pyplot as plt` is imported twice.

**Why it matters:**
- Low risk functionally, but indicates reduced code hygiene and review rigor.

**Evidence:**
- Duplicate consecutive imports in `visualize_predictions.py`.

**Recommendation:**
- Remove duplicate import and run formatter/linter.

### 7) Unused local variable in holdout split branch (**Low**)
- `holdout_set = set(holdout_timepoints)` is defined but never used.

**Why it matters:**
- Minor, but suggests incomplete refactors and cognitive noise.

**Evidence:**
- `data_utils.py` in `split_mode == "holdout"` branch.

**Recommendation:**
- Delete unused variable or use it directly in mask calculation.

### 8) Repository contains large training artifacts in source tree (**Medium**)
- `lightning_logs/version_*/checkpoints/*.ckpt` and metrics CSVs are present in repo tree.

**Why it matters:**
- Usually leads to repository bloat, noisy diffs, and harder collaboration unless intentionally versioned.

**Evidence:**
- Multiple checkpoint files under `lightning_logs/`.

**Recommendation:**
- Add or validate `.gitignore` policy for generated artifacts; store model artifacts in release storage if needed.

---

## Suggested Priority Order
1. Fix split-mode validation (Finding 2).
2. Resolve/remove stale `physics_utils.py` path (Finding 1).
3. Add deterministic seeding controls (Finding 4).
4. Decide and expose normalization policy (Finding 3).
5. Cleanup hygiene items (Findings 5–7).
6. Define artifact tracking strategy (`lightning_logs`) (Finding 8).
