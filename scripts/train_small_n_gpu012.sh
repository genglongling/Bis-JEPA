#!/usr/bin/env bash
# Train + eval small-n PushT baselines (live DINO), WM ∥ Bisim per GPU:
#   GPU 0: n=25  — dinowm ∥ dinobisim   steps_per_epoch=141
#   GPU 1: n=50  — dinowm ∥ dinobisim   steps_per_epoch=282
#   GPU 2: n=100 — dinowm ∥ dinobisim   steps_per_epoch=564
# Then serial legacy_six eval (WM then Bisim) on that GPU.
#
# Usage (Anderson):
#   bash scripts/train_small_n_gpu012.sh
# From Mac:
#   bash scripts/train_small_n_gpu012.sh --remote
set -euo pipefail

REPO="${REPO:-$HOME/Bis-JEPA}"
REMOTE="${REMOTE:-anderson-accelerator}"

if [[ "${1:-}" == "--remote" ]]; then
  LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  RHOST="${REMOTE:-anderson-accelerator}"
  RREPO="${REMOTE_REPO:-~/Bis-JEPA}"
  scp "$LOCAL_REPO/scripts/train_small_n_gpu012.sh" \
    "$LOCAL_REPO/scripts/parallel_rollout_pipeline.sh" \
    "$LOCAL_REPO/scripts/eval_readme_tables_all.sh" \
    "$LOCAL_REPO/scripts/update_readme_pusht_results.py" \
    "$RHOST:${RREPO}/scripts/"
  scp "$LOCAL_REPO/env/visual_conditions.py" "$RHOST:${RREPO}/env/"
  ssh -o BatchMode=yes "$RHOST" "bash ${RREPO}/scripts/train_small_n_gpu012.sh"
  exit 0
fi

CKPT_BASE="${CKPT_BASE:-/mnt/t7shield/gloria/bis-jepa}"
T7_SCRATCH="${T7_SCRATCH:-$CKPT_BASE/scratch}"
LOG_DIR="${LOG_DIR:-$CKPT_BASE/logs/rollout_sweep}"
MANIFEST="${MANIFEST:-$LOG_DIR/manifest_small_n.json}"
RESULTS_JSON="${RESULTS_JSON:-$T7_SCRATCH/plan_outputs/rollout_sweep_results.json}"
PLAN_OUTPUTS="${PLAN_OUTPUTS:-$T7_SCRATCH/plan_outputs}"

GPU_N25="${GPU_N25:-0}"
GPU_N50="${GPU_N50:-1}"
GPU_N100="${GPU_N100:-2}"
STOP_EXISTING="${STOP_EXISTING:-1}"
# Resume partial WM by_tag ckpts by default
FRESH_TRAIN="${FRESH_TRAIN:-0}"

export CKPT_BASE LOG_DIR MANIFEST RESULTS_JSON PLAN_OUTPUTS
export SAVE_EVERY="${SAVE_EVERY:-1}" TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export USE_PRECOMPUTED_DINO=false
export USE_PRECOMPUTED_DINO_EVAL=false
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-20}"
export SKIP_OPENLOOP_ROLLOUT="${SKIP_OPENLOOP_ROLLOUT:-1}"
export BISIM_POS_ENCODING="${BISIM_POS_ENCODING:-grid_mlp}"
# Two ~11GiB trainers share a 48GiB card; allow start when ≥10GiB free
export MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-10000}"
export PARALLEL_TRAIN_STAGGER_SEC="${PARALLEL_TRAIN_STAGGER_SEC:-15}"
export EPOCHS="${EPOCHS:-50}" PURGE_AFTER_PULL=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DATASET_DIR="${DATASET_DIR:-$REPO/datasets/data}"
export EVAL_CATEGORY="${EVAL_CATEGORY:-legacy_six}"
export CEM_EPISODE_BATCH_SIZE="${CEM_EPISODE_BATCH_SIZE:-1}"
export CEM_NUM_SAMPLES="${CEM_NUM_SAMPLES:-300}"
export SKIP_ARCHIVE="${SKIP_ARCHIVE:-1}"
export TMPDIR="${TMPDIR:-/dev/shm/gloria-bisjepa}"
export WANDB_DIR="${WANDB_DIR:-$CKPT_BASE/wandb}"
export PYTHONUNBUFFERED=1
export FRESH_TRAIN

STEPS_N25="${STEPS_N25:-141}"
STEPS_N50="${STEPS_N50:-282}"
STEPS_N100="${STEPS_N100:-564}"
VAL_N25="${VAL_N25:-14}"
VAL_N50="${VAL_N50:-28}"
VAL_N100="${VAL_N100:-56}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/small_n_gpu012.log"; }

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-dino_wm310}"
mkdir -p "$LOG_DIR" "$PLAN_OUTPUTS" "$TMPDIR" "$REPO/plan_outputs" "$CKPT_BASE/outputs/by_tag"
cd "$REPO"

if [[ "$STOP_EXISTING" == "1" ]]; then
  log "STOP_EXISTING=1 — killing prior small-n workers/trains (resume ckpts if present)"
  pkill -f "parallel_rollout_pipeline.sh worker_multi .*dinowm_n25" 2>/dev/null || true
  pkill -f "parallel_rollout_pipeline.sh worker_multi .*dinowm_n50" 2>/dev/null || true
  pkill -f "parallel_rollout_pipeline.sh worker_multi .*dinowm_n100" 2>/dev/null || true
  pkill -f "parallel_rollout_pipeline.sh worker_wm_bisim_parallel" 2>/dev/null || true
  pkill -f "wandb_display_name=dinowm_n25" 2>/dev/null || true
  pkill -f "wandb_display_name=dinowm_n50" 2>/dev/null || true
  pkill -f "wandb_display_name=dinowm_n100" 2>/dev/null || true
  pkill -f "wandb_display_name=dinobisim_n25" 2>/dev/null || true
  pkill -f "wandb_display_name=dinobisim_n50" 2>/dev/null || true
  pkill -f "wandb_display_name=dinobisim_n100" 2>/dev/null || true
  pkill -f "by_tag/dinowm_n25" 2>/dev/null || true
  pkill -f "by_tag/dinowm_n50" 2>/dev/null || true
  pkill -f "by_tag/dinowm_n100" 2>/dev/null || true
  pkill -f "by_tag/dinobisim_n25" 2>/dev/null || true
  pkill -f "by_tag/dinobisim_n50" 2>/dev/null || true
  pkill -f "by_tag/dinobisim_n100" 2>/dev/null || true
  sleep 5
  rm -f "$LOG_DIR/parallel_gpu0.lock" "$LOG_DIR/parallel_gpu1.lock" "$LOG_DIR/parallel_gpu2.lock"
fi

if [[ "$FRESH_TRAIN" == "1" ]]; then
  log "FRESH_TRAIN=1 — wiping by_tag small-n checkpoints"
  for t in dinowm_n25 dinowm_n50 dinowm_n100 dinobisim_n25 dinobisim_n50 dinobisim_n100; do
    rm -rf "$CKPT_BASE/outputs/by_tag/$t"
  done
else
  log "FRESH_TRAIN=0 — resume partial by_tag small-n if present"
  for t in dinowm_n25 dinowm_n50 dinowm_n100 dinobisim_n25 dinobisim_n50 dinobisim_n100; do
    d="$CKPT_BASE/outputs/by_tag/$t/checkpoints"
    if [[ -d "$d" ]]; then
      n=$(ls -1 "$d"/model_[0-9]*.pth 2>/dev/null | wc -l)
      log "  $t: $n ckpts"
    else
      log "  $t: (none)"
    fi
  done
fi

launch_parallel() {
  local gpu="$1"
  local steps="$2"
  local val_steps="$3"
  local wm_spec="$4"
  local bisim_spec="$5"
  local logfile="$LOG_DIR/small_n_gpu${gpu}.log"
  : > "$logfile"
  log "LAUNCH GPU $gpu PARALLEL steps=$steps val=$val_steps wm=$wm_spec bisim=$bisim_spec"
  nohup env \
    CKPT_BASE="$CKPT_BASE" MANIFEST="$MANIFEST" RESULTS_JSON="$RESULTS_JSON" \
    PLAN_OUTPUTS="$PLAN_OUTPUTS" LOG_DIR="$LOG_DIR" \
    USE_PRECOMPUTED_DINO=false USE_PRECOMPUTED_DINO_EVAL=false \
    FORCE_TRAIN="$FORCE_TRAIN" FRESH_TRAIN=0 \
    SKIP_OPENLOOP_ROLLOUT="$SKIP_OPENLOOP_ROLLOUT" \
    EPOCHS="$EPOCHS" SAVE_EVERY="$SAVE_EVERY" \
    STEPS_PER_EPOCH="$steps" \
    VAL_STEPS_PER_EPOCH="$val_steps" \
    TRAIN_NUM_WORKERS="$TRAIN_NUM_WORKERS" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    BISIM_POS_ENCODING="$BISIM_POS_ENCODING" \
    MIN_GPU_FREE_MIB="$MIN_GPU_FREE_MIB" \
    PARALLEL_TRAIN_STAGGER_SEC="$PARALLEL_TRAIN_STAGGER_SEC" \
    EVAL_CATEGORY="$EVAL_CATEGORY" \
    CEM_EPISODE_BATCH_SIZE="$CEM_EPISODE_BATCH_SIZE" \
    CEM_NUM_SAMPLES="$CEM_NUM_SAMPLES" \
    SKIP_ARCHIVE="$SKIP_ARCHIVE" \
    DATASET_DIR="$DATASET_DIR" TMPDIR="$TMPDIR" \
    WANDB_MODE="$WANDB_MODE" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    bash "$REPO/scripts/parallel_rollout_pipeline.sh" worker_wm_bisim_parallel "$gpu" \
      "$wm_spec" "$bisim_spec" \
    >> "$logfile" 2>&1 &
  log "  PID=$! log=$logfile"
}

log "=== train_small_n_gpu012 PARALLEL WM∥Bisim start ==="
log "S(n): n25=$STEPS_N25 n50=$STEPS_N50 n100=$STEPS_N100  epochs=$EPOCHS"

launch_parallel "$GPU_N25" "$STEPS_N25" "$VAL_N25" \
  "dinowm_n25:dinowm:25" "dinobisim_n25:bisim:25"

launch_parallel "$GPU_N50" "$STEPS_N50" "$VAL_N50" \
  "dinowm_n50:dinowm:50" "dinobisim_n50:bisim:50"

launch_parallel "$GPU_N100" "$STEPS_N100" "$VAL_N100" \
  "dinowm_n100:dinowm:100" "dinobisim_n100:bisim:100"

log "=== launched parallel small-n chains ==="
log "Monitor: tail -f $LOG_DIR/small_n_gpu*.log"
