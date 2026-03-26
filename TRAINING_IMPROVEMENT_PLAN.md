# PINAPINN Model-Improvement Playbook

This is the exact approach I would use to improve the current fits you shared.

## 1) Diagnose the error pattern first (before touching architecture)
From your plots, the model currently:
- **Misses early-time sharp transients** (0–8h) for pERK/pMEK in several conditions.
- **Over-smooths trajectories** (curves are too “global/sigmoidal”, not local enough around t=1–8h).
- Has **systematic baseline mismatch** in no-drug and low-drug behavior.
- In holdout condition (Vem+PI3Ki), captures trends but misses magnitude at held-out late points.

### Action
Compute and track per-species/per-condition metrics:
- MAE, RMSE, MAPE at each timepoint.
- Separate score buckets: **early** (0,1,4,8h) vs **late** (24,48h).
- Rank worst 3 species/conditions each run.

---

## 2) Stabilize experiment protocol (so changes are comparable)
For every run, lock:
- seed,
- split definition,
- normalization mode,
- collocation generation recipe,
- total training epochs.

### Action
Create a single “run config” table in logs/artifacts:
- model width/depth,
- loss weights,
- collocation counts,
- optimizer/lr schedule,
- best validation epoch.

---

## 3) Rebalance objective terms (most likely highest ROI)
PINNs often fail here due to data-vs-physics imbalance.

### Action
Use a staged schedule:
1. **Warm-start data only** (e.g., 200–400 epochs) to fit observed points.
2. Turn on physics term with small weight.
3. Increase physics weight gradually (curriculum).

Recommended initial sweep:
- `lambda_data`: fixed 1.0
- `lambda_physics`: [1e-4, 3e-4, 1e-3, 3e-3]
- `lambda_steady_state`: [1e-4, 1e-3, 1e-2]

Pick by holdout RMSE + biology plausibility.

---

## 4) Improve temporal representation (current bottleneck)
Your model is likely under-representing fast early dynamics.

### Action
Test richer time features while keeping drug inputs unchanged:
- Add basis terms: `[t, t^2, log(1+t), exp(-t/tau)]` with small set of tau values.
- Or use Fourier features on normalized time for better local curvature.

Then compare **early-window RMSE** specifically.

---

## 5) Increase collocation quality where needed
You already bias to early-time, which is good. Next step: make it adaptive.

### Action
Adaptive collocation loop every N epochs:
1. Evaluate physics residual on a dense candidate pool.
2. Resample top-k highest residual points.
3. Mix with baseline random points to avoid mode collapse.

This usually improves sharp transient fit without huge model-size growth.

---

## 6) Constrain parameters to biologically valid ranges
Unconstrained trainable kinetics can produce compensatory but unrealistic dynamics.

### Action
For key kinetic params:
- Optimize unconstrained raw variable `u` and map with `softplus`/sigmoid to bounded ranges.
- Example: hill coefficient in [1,4], degradation rates > 0, saturation constants > 0.

Track learned parameter trajectories during training and stop runs that diverge to extremes.

---

## 7) Add targeted penalties for known failure modes
The plots suggest repeated behavior errors (late rebounds too strong/weak, baseline drift).

### Action
Add small auxiliary penalties:
- no-drug drift penalty at all times (not only average derivative near zero),
- early-time anchor loss at t=1 and t=4 for most sensitive species (pERK/pMEK/DUSP6),
- monotonicity window constraints where biologically expected under specific treatments.

Keep these penalties small; use as priors, not hard clamps.

---

## 8) Optimize training dynamics
### Action
Use:
- AdamW (or Adam) warm phase,
- cosine/plateau scheduler,
- optional LBFGS short refinement at end (PINNs often benefit),
- gradient norm logging and clip only when needed.

Add early stopping on holdout condition metric, not just train loss.

---

## 9) Model capacity sweep (after loss balancing)
Only after steps 3–8:
- Try widths [64, 96, 128], depth [2, 3, 4],
- Keep best 3 candidates by holdout score,
- Select the smallest model meeting error target.

Bigger model alone usually will not fix the current mismatch if losses are misbalanced.

---

## 10) Concrete 2-week execution plan
### Day 1–2
- Add metric dashboard (per species/condition/time bucket).
- Baseline rerun x3 seeds.

### Day 3–5
- Loss-weight curriculum sweep (Step 3).
- Choose best setting by holdout + visual fit.

### Day 6–8
- Time-feature upgrade + adaptive collocation (Steps 4–5).

### Day 9–10
- Parameter constraints + auxiliary penalties (Steps 6–7).

### Day 11–12
- Optimizer/scheduler + LBFGS refinement tests.

### Day 13–14
- Capacity sweep + final model card (metrics + plots + learned params).

---

## Acceptance criteria for “improved”
A run is accepted only if all are true:
1. Holdout (Vem+PI3Ki) RMSE improves by >=20% vs current baseline.
2. Early-window RMSE (0–8h) improves for pERK/pMEK/DUSP6 in at least 4/5 conditions.
3. No-drug condition remains near steady state without unrealistic drift.
4. Learned kinetic parameters stay within predefined biological bounds.
