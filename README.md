# Learning Invariant Visual Representations for Planning with Joint-Embedding Predictive World Models

[[Paper]](https://arxiv.org/abs/2602.18639)

Leonardo F. Toso\*, Davit Shadunts\*, Yunyang Lu\*, Nihal Sharma, Donglin Zhan, Nam H. Nguyen, James Anderson

Columbia University, Capital One

\* Equal contribution

<p align="center">
  <img src="assets/encoding_motivation.png" width="600">
</p>

## Overview

JEPA-based world models, including [DINO-WM](https://arxiv.org/abs/2411.04983), are sensitive to *slow features* — task-irrelevant visual variations such as background changes, lighting, and distractors that change slowly over time. The predictive objective in JEPAs can be minimized by encoding only such temporally consistent information, leading to degenerate representations that fail under test-time visual shifts.

We address this by augmenting the latent dynamics with a **bisimulation encoder** that enforces control-relevant state equivalence. States with similar transition dynamics are mapped to nearby latent embeddings, while task-irrelevant visual features are discarded. The bisimulation encoder is trained jointly with the transition model, without relying on reward prediction.

Our model operates in a latent space up to **10x smaller** than that of DINO-WM and is agnostic to the choice of pretrained visual encoder (DINOv2, SimDINOv2, iBOT).

<p align="center">
  <img src="assets/motivating_pointmaze.png" width="600">
</p>

## Results

We evaluate on PointMaze navigation under six test-time **sim** visual conditions: **NC** (no change, neutral background), **SC** (slight background change), **C** (tinted background), **LC** (large color shift), **LCG** (large color gradient), and **D** (distractors, including a moving highlight). The same six codes are used for PushT-style planning eval; rendering is set at **environment** time (e.g. `wall_env.visual_condition` / `pusht_env.visual_condition` in the planning config).

<p align="center">
  <img src="assets/backgrounds_pm.png" width="600">
</p>

| Model | NC | SC | C | LC | LCG | D |
|-------|------|------|------|------|------|------|
| DINO-WM | 0.80 | 0.72 | 0.60 | 0.56 | 0.48 | 0.78 |
| DINO-WM w/ DR | 0.82 | 0.82 | 0.82 | 0.68 | 0.64 | 0.82 |
| **Ours (DINO-Bisim)** | **0.78** | **0.80** | **0.76** | **0.86** | **0.78** | **0.82** |

DINO-WM degrades under background changes (0.80 → 0.48 from NC to LCG). Domain randomization helps when test backgrounds resemble training augmentations but fails under larger shifts. Our model maintains consistent performance across all conditions.

We further validate with different pretrained visual encoders:

| Model | NC | SC | C | LC | LCG | D |
|-------|------|------|------|------|------|------|
| No Encoder | 0.68 | 0.44 | 0.70 | 0.26 | 0.36 | 0.64 |
| **DINOv2** | **0.78** | **0.80** | **0.76** | **0.86** | **0.78** | **0.82** |
| SimDINOv2 | 0.40 | 0.38 | 0.36 | 0.42 | 0.42 | 0.36 |
| iBOT | 0.72 | 0.70 | 0.74 | 0.72 | 0.72 | 0.72 |

## Getting Started

### Installation

```bash
git clone https://github.com/jd-anderson/dino_bsmpc.git
cd dino_bsmpc
conda env create -f environment.yaml
conda activate dino_wm
```

#### MuJoCo

Create the `.mujoco` directory and download MuJoCo210:

```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz
```

Add to `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

### Datasets

Datasets are provided by [DINO-WM](https://github.com/gaoyuezhou/dino_wm) and can be downloaded [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28).

Set the dataset path:
```bash
export DATASET_DIR=/path/to/data
```

Expected structure:
```
data
├── deformable
│   ├── granular
│   └── rope
├── point_maze
├── pusht_noise
└── wall_single
```

## How to run

End-to-end flow for a typical **local PushT** workflow (see `conf/train_local.yaml`); for Slurm / other envs, the same idea applies with `train.yaml` and the `env=*` override.

1. **Activate the environment** (after [Installation](#installation)):

   ```bash
   conda activate dino_wm
   cd <path/to/this-repo>
   ```

2. **Point to datasets** (or rely on the default in `train.py`):

   ```bash
   export DATASET_DIR=/path/to/parent/of/pusht_noise
   # If unset, training defaults DATASET_DIR to <repo>/datasets/data
   ```

3. **Train** (checkpoints, CSV loss log, and image mosaics go under the Hydra run dir—see below):

   ```bash
   python train.py --config-name train_local
   ```

   For **paper / cluster**-style training (Slurm, other tasks):

   ```bash
   python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
   ```

4. **Find outputs** (default `ckpt_base_path=./` in the configs):  
   - Run directory: `outputs/YYYY-MM-DD/HH-MM-SS/` (Hydra [working directory](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/)).  
   - Checkpoints: `.../checkpoints/model_*.pth`.  
   - Loss: `.../training_loss_log.csv`, terminal + [Weights & Biases](https://wandb.ai) scalars.  
   - Mosaics: `.../train/`, `.../valid/` (PNGs; loss is **not** drawn on them).

5. **Plan / evaluate in sim** (optional): set `model_name` to the run folder name under `outputs/`, e.g. `2026-04-21/18-44-53`, and set `ckpt_base_path` so `ckpt_base_path/outputs/<model_name>/` contains checkpoints.

   - **All six visual conditions (NC…D)** in one go:

     ```bash
     python eval_pusht_six_conditions.py --model-name YYYY-MM-DD/HH-MM-SS --ckpt-base-path ./
     ```

   - **Single** planning run (see [Planning](#planning) for more):

     ```bash
     python plan.py --config-name plan_pusht_local.yaml model_name=YYYY-MM-DD/HH-MM-SS
     ```

6. **Sweeps** (optional): `python train_sweep.py ...` and `python evaluate_visual_grid.py --config ...` for multi–checkpoint tables.

**Hydra tips:** any config value can be overridden on the command line, e.g. `training.epochs=2`, `regularization=pca`. Use `python train.py --config-name train_local --help` for the composed config (including nested keys).

## Training

Train a world model with the bisimulation encoder:

```bash
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
```

Key bisimulation hyperparameters can be set via Hydra overrides:
```bash
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3 \
    bisim_latent_dim=32 \
    training.bisim_lr=5e-7 var_loss_coef=1
```

Hyperparameter sweeps:
```bash
python train_sweep.py --config-file train_sweep_config.json --gpus 0 1 2 3
```

Model checkpoints are saved to `<ckpt_base_path>/outputs/`. Set `ckpt_base_path` in `conf/train.yaml`.

### Local training (PushT, `train_local`)

For single-GPU, non-Slurm runs the repo includes `conf/train_local.yaml` (default env: `pusht`, Hydra `basic` launcher). Example:

```bash
export DATASET_DIR=/path/to/datasets/data   # parent of `pusht_noise/`; see "Datasets" above
python train.py --config-name train_local
```

`train.py` also defaults `DATASET_DIR` to `<cwd>/datasets/data` if the variable is unset. Training logs and `training_loss_log.csv` are written under the Hydra run directory, e.g. `outputs/YYYY-MM-DD/HH-MM-SS/`.

**Image mosaics (train + val):** On the first batch of each phase per epoch, the trainer writes PNGs under `train/` and `valid/`. If `has_decoder: true`, the existing grid is [ground truth \| predicted future \| reconstructed]. If `has_decoder: false` (common in `train_local` with a frozen encoder and no VQ-VAE), `training.log_ground_truth_mosaic: true` (default) still saves a **ground-truth** frame grid so you can verify windows and data. W&B can log the same mosaics when `training.log_image_mosaics_to_wandb: true` (keys like `train/gt_frames`, `valid/gt_frames`, and `*/recon_pred_mosaic` with a decoder).

### Bisimulation regularization: PCA (default) vs VICReg

Bisim can use either the **PCA / hinge** schedule plus **per-patch covariance** regularization (`regularization: pca`, or the legacy path when `regularization` is not `vicreg`), or a **VICReg**-style block on **mean-pooled** bisim features (`regularization: vicreg`). Coefficients for the latter are set with `vicreg_inv_coef`, `vicreg_var_coef`, `vicreg_cov_coef`, and `vicreg_std_min` in `conf/train_local.yaml` / `conf/train.yaml`. Use **PCA** for the original paper setup unless you are explicitly comparing to VICReg.

**Logging (apples-to-apples with PCA columns):** for VICReg, `train_bisim_var_loss` / `train_bisim_cov_reg` log the **VICReg variance-hinge** and **off-diagonal covariance** terms (weighted); `bisim_vicreg_inv` and `bisim_vicreg_total` report invariance and the full VIC block. For the PCA path, the variance and covariance columns are unchanged; the `vicreg_*` fields are zero. See `loss_history/loss_csv.py` and `models/bisim.py` for details.

**Preliminary local metrics (PushT, `pusht_noise` full train/val, 2 epochs, comparable batch setup):** total loss in `training_loss_log.csv` (not planning success). These are for quick comparison only; the PCA run used an earlier config snapshot, while the VICReg run used `regularization: vicreg` in `train_local`.

| Regime | Epoch | train_loss | val_loss |
|--------|--------|------------|----------|
| PCA (hinge/PCA + per-patch cov) | 1 | 0.304 | 0.252 |
| PCA | 2 | 0.214 | 0.221 |
| VICReg | 1 | 0.203 | 0.322 |
| VICReg | 2 | 0.164 | 0.309 |

VICReg yields **lower training** loss in this table but **higher validation** total loss than this PCA run; use planning / downstream metrics to judge the regime you care about, not a single scalar.

### Encoder Selection

The pretrained visual encoder is specified via the `encoder` config group:
```bash
# DINOv2 (default, ViT-S/14, d_z=384)
python train.py --config-name train.yaml encoder=dino ...

# SimDINOv2 (ViT-B/16, d_z=768)
python train.py --config-name train.yaml encoder=simdino ...

# iBOT (ViT-S/16, d_z=384)
python train.py --config-name train.yaml encoder=ibot ...
```

To train the bisimulation encoder directly from pixels (bypassing the pretrained encoder):
```bash
python train.py --config-name train.yaml model.bypass_dinov2=True ...
```

## Planning

Plan with a trained model using MPC with CEM:

```bash
python plan.py model_name=<model_name> n_evals=5 planner=cem goal_H=5 \
    goal_source='random_state' planner.opt_steps=30
```

Environment-specific planning configs:
```bash
python plan.py --config-name plan_point_maze.yaml model_name=point_maze
python plan.py --config-name plan_pusht.yaml model_name=pusht
python plan.py --config-name plan_wall.yaml model_name=wall
```

**Training/validation** uses **fixed** trajectories on disk; **closed-loop** evaluation under a given appearance is done in **sim** by setting the visual condition, e.g. for PushT:

```bash
python plan.py --config-name plan_pusht_local.yaml model_name=YOUR_OUTPUT_DIR \
  pusht_env.visual_condition=NC
# Repeat with SC, C, LC, LCG, D to measure robustness (same checkpoint).
```

**One model, all six conditions** (typical after training): use `eval_pusht_six_conditions.py`. It runs `plan.py` with `plan_pusht_local` once per condition and writes `plan_outputs/sixcond_<model>_<timestamp>.json` plus a Markdown row to stdout.

```bash
export DATASET_DIR=/path/to/parent/datasets/data
python eval_pusht_six_conditions.py --model-name 2026-04-21/18-44-53 --ckpt-base-path ./
# Optional: faster evals or extra Hydra overrides for each run
python eval_pusht_six_conditions.py --model-name 2026-04-21/18-44-53 --ckpt-base-path ./ \
  --extra n_evals=20
```

`model_name` is the folder under `outputs/` (often `YYYY-MM-DD/HH-MM-SS` from Hydra). See `python eval_pusht_six_conditions.py --help`.

A separate helper sweeps **multiple** checkpoints (e.g. ablation rows) via JSON config (`evaluate_visual_grid.py` and `evaluate_visual_grid_config_pusht.json`):

```bash
python evaluate_visual_grid.py --config evaluate_visual_grid_config_pusht.json
```

**Note:** `val` during **training** is one dataloader (your `train/val` split in `DATASET_DIR/.../pusht_noise`); it does not automatically run six val passes. OOD sim eval is the intended way to get NC–D success curves with a model trained (often on near-NC) data.

Set `ckpt_base_path` in `conf/plan.yaml` to point to the checkpoint directory. Planning logs and visualizations are saved to `./plan_outputs/`.

## Citation

```
@misc{toso2026learninginvariantvisualrepresentations,
      title={Learning Invariant Visual Representations for Planning with Joint-Embedding Predictive World Models}, 
      author={Leonardo F. Toso and Davit Shadunts and Yunyang Lu and Nihal Sharma and Donglin Zhan and Nam H. Nguyen and James Anderson},
      year={2026},
      eprint={2602.18639},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.18639}, 
}
```

## Acknowledgements

This codebase builds on [DINO-WM](https://github.com/gaoyuezhou/dino_wm) by [Zhou et al.](https://arxiv.org/abs/2411.04983).
