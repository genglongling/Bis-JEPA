# Matched baselines: DINO-WM vs Ours (PushT / planning)

To compare **DINO-WM** and **Ours (Bis-JEPA)** in a paper, the runs must share the same **data**, **randomness**, and **compute budget** (at least for the training stage you report).

## Train both methods here

| Method | Config | Latent space | Loss |
|--------|--------|--------------|------|
| **Ours (DINO-Bisim)** | `train_local` / `train.yaml` | 196×32 bisim (+ aux 196×384 DINO) | JEPA_bisim + `dinowm_coef`×JEPA_DINO-WM + bisim |
| **DINO-WM only** | `train_dinowm_local` / `train_dinowm` | 196×384 DINOv2 patches | JEPA MSE only (`has_bisim: false`) |

```bash
export DATASET_DIR=/path/to/datasets/data
python train.py --config-name train_dinowm_local training.seed=0
python train.py --config-name train_local training.seed=0
```

Use the same `training.epochs`, `batch_size`, `num_hist`, and env overrides for both. Checkpoints and `hydra.yaml` land under `outputs/YYYY-MM-DD/HH-MM-SS/`.

## What to keep identical

1. **Dataset**  
   - Same `DATASET_DIR` parent so `pusht_noise/` (and the same train/val split on disk) is read.  
   - Do not change env config between baselines without stating it (e.g. `conf/env/pusht.yaml`).

2. **Seed**  
   - Use the same `training.seed` (e.g. `0`) for all runs in a table row, *or* report **mean ± std over multiple seeds** with the **same** seed list for each method.  
   - DINO-WM: set their run seed to the same value if their launcher exposes it; note it in the paper.

3. **Budget (epochs, batch, architecture flags)**  
   - For **DINO-WM / PCA-style bisim** comparisons, align with `conf/train.yaml` or `conf/train_local.yaml`: e.g. `training.epochs=50`, `batch_size=20`, `num_hist: 3`, `regularization: pca` (Table 2). For **VICReg** auxiliary, set the same budget but state `regularization: vicreg` in the table footnote.  
   - DINO-WM: use **their** published training recipe *or* the **same** epoch and batch size as you use for Ours, and state any residual differences (e.g. their default vs your `num_hist`).

4. **Planning eval (after training)**  
   - Same `plan_pusht_local` (or a shared `plan_*.yaml`), same `n_evals`, `goal_H`, CEM/MPC settings, and **same** `eval_state` success definition.  
   - For headline numbers, compare under the **same** `pusht_env.visual_condition` (often **NC** first; then a second table for NC…D if needed).

5. **Short vs full training**  
   - `training.epochs=2` is a **smoke test** only; it is not comparable to DINO-WM’s reported PushT success (~0.9) without stating “2-epoch ablation”.

## Learning curve (2 / 10 / 50 / 100)

- **Recommended:** one run to the **maximum** epoch budget (e.g. 50 per Table 2), then plot `val_loss` vs epoch and optionally mark 2, 10, 25, 50 on the x-axis.  
- **Script:** from repo root,
  ```bash
  python scripts/plot_learning_curve.py \
    --csv outputs/YYYY-MM-DD/HH-MM-SS/training_loss_log.csv \
    --out figures/pusht_val_loss.png \
    --print-checkpoints
  ```
- **Multi-line comparison (two CSVs on one plot):**
  ```bash
  python scripts/plot_learning_curve.py --metric val_loss --out both.png \
    --run "Ours=outputs/.../run_ours/training_loss_log.csv" \
    --run "DINO-WM=outputs/.../run_dinowm/training_loss_log.csv"
  ```

DINO-WM is trained **either** as part of Ours (`train_dinowm_aux: true` → `predictor_dinowm` in the same checkpoint) **or** standalone via `train_dinowm_local` / `train_dinowm`. You can also use the upstream [DINO-WM repository](https://github.com/gaoyuezhou/dino_wm) with the same `DATASET_DIR` and eval settings.

## Bisimulation loss (Ours only)

DINO-WM runs do **not** use bisim. For Ours, the paper objective is \(\mathcal{L}_{\mathrm{jepa\text{-}bisim} = \mathcal{L}_{\mathrm{dyn}} + \lambda_{\mathrm{bisim}}\mathcal{L}_{\mathrm{bisim}}\) with \(\mathcal{L}_{\mathrm{dyn}} = \|h(f(o_{t+1})) - T_\phi(h(f(o_t)), a_t)\|^2\) and \(\mathcal{L}_{\mathrm{bisim}} = (\|w_t - w'_t\|^2 - \gamma\|T_\phi(w_t,a_t) - T_\phi(w'_t,a'_t)\|^2)^2\) (independent \(a_t, a'_t\) from paired trajectories). See README **Architecture → Bisimulation loss** for notation ↔ code. Keys: `bisim_coef`, `bisim_transition_target: predictor`, `bisim_latent_metric: l2`, `regularization`, `bisim_memory_buffer_size`.

## `train_local` vs `train`

`conf/train_local.yaml` is **single-GPU, local** only (`hydra/launcher: basic`). It matches `conf/train.yaml` on epochs, LRs, bisim, `num_hist`, **`regularization: pca`**, and most flags. Slurm and GPU type in `train.yaml` are for the cluster. Use **`regularization=vicreg`** locally for VicReg auxiliary.
