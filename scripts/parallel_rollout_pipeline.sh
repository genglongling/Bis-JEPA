#!/usr/bin/env bash
# Train -> six-condition eval -> archive on Anderson T7 storage.
# Default: sequential on GPU 0. Per-GPU workers: worker <gpu> <tag> <method> <n_rollout>
#
# On Anderson:
#   nohup bash scripts/parallel_rollout_pipeline.sh >> logs/rollout_sweep/parallel.log 2>&1 &
#   bash scripts/start_gpu12_rollout_workers.sh   # dinowm_n5000 @ GPU1, dinowm_nfull @ GPU2
#
# On Mac (after archives appear):
#   bash scripts/pull_anderson_checkpoints.sh
set -uo pipefail

REPO="${REPO:-$HOME/Bis-JEPA}"
LOG_DIR="${LOG_DIR:-$REPO/logs/rollout_sweep}"
mkdir -p "$LOG_DIR"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dino_wm310}"
DATASET_DIR="${DATASET_DIR:-$REPO/datasets/data}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
# n1000-sized = TrajSlicer optimizer-step budget @ batch_size=20.
# Old TrajSlicer progress showed 5267 batches/epoch → 5267 * 20 = 105340 samples
# for RandomTrajSamplerDataset (DataLoader batches = ceil(len / batch_size)).
STEPS_PER_EPOCH_N1000="${STEPS_PER_EPOCH_N1000:-105340}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
VAL_STEPS_PER_EPOCH="${VAL_STEPS_PER_EPOCH:-200}"
USE_PRECOMPUTED_DINO="${USE_PRECOMPUTED_DINO:-false}"
USE_PRECOMPUTED_DINO_EVAL="${USE_PRECOMPUTED_DINO_EVAL:-$USE_PRECOMPUTED_DINO}"
DINO_FEATURES_ROOT="${DINO_FEATURES_ROOT:-$CKPT_BASE/dino_features}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
if [[ -z "${SYNC_DIR:-}" ]]; then
  if [[ "$CKPT_BASE" == *t7shield* ]]; then
    SYNC_DIR="$CKPT_BASE/scratch/to_sync"
  else
    SYNC_DIR="$REPO/to_sync"
  fi
fi
MANIFEST="${MANIFEST:-$REPO/logs/rollout_sweep/manifest.json}"
RESULTS_JSON="${RESULTS_JSON:-${PLAN_OUTPUTS:-$REPO/plan_outputs}/rollout_sweep_results.json}"
CKPT_BASE="${CKPT_BASE:-/mnt/t7shield/gloria/bis-jepa}"
if [[ ! -d "${CKPT_BASE}/outputs" && -d "$REPO/Checkpoints/outputs" ]]; then
  CKPT_BASE="$REPO/Checkpoints"
fi
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$CKPT_BASE/outputs}"
BISIM_NFULL_CKPT_BASE="${BISIM_NFULL_CKPT_BASE:-$REPO/checkpoints_captialone}"
BISIM_NFULL_MODEL="${BISIM_NFULL_MODEL:-push-T randomized/new_pushT90}"
BISIM_NFULL_EPOCH="${BISIM_NFULL_EPOCH:-90}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-50}"
SAVE_EVERY="${SAVE_EVERY:-5}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-35000}"
GPU_WAIT_SEC="${GPU_WAIT_SEC:-60}"
PURGE_AFTER_PULL="${PURGE_AFTER_PULL:-0}"
WAIT_PULL_SEC="${WAIT_PULL_SEC:-3600}"
PULL_POLL_SEC="${PULL_POLL_SEC:-120}"
# Match plan_pusht_local.yaml / July baseline (CEM 300 samples).
CEM_NUM_SAMPLES="${CEM_NUM_SAMPLES:-300}"
# Skip heavy archive rsync when home/T7 scratch is tight (still writes sixcond + manifest).
SKIP_ARCHIVE="${SKIP_ARCHIVE:-0}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

verify_dataset() {
  if [[ ! -d "$DATASET_DIR/pusht_noise" ]]; then
    log "ERROR missing dataset: $DATASET_DIR/pusht_noise (set DATASET_DIR to parent of pusht_noise)"
    return 1
  fi
  return 0
}

resolve_eval_category() {
  # Live override so long-running workers pick up category changes without restart.
  if [[ -n "${LOG_DIR:-}" && -f "$LOG_DIR/EVAL_CATEGORY" ]]; then
    tr -d '[:space:]' < "$LOG_DIR/EVAL_CATEGORY"
    return 0
  fi
  echo "${EVAL_CATEGORY:-legacy_six}"
}

expected_eval_cond_count() {
  local cat
  cat="$(resolve_eval_category)"
  case "$cat" in
    all) echo 17 ;;
    color) echo 7 ;;
    color_d) echo 8 ;;
    legacy_six) echo 6 ;;
    texture) echo 5 ;;
    viewpoint) echo 4 ;;
    mild_tv) echo 4 ;;
    dynamic) echo 1 ;;
    *) echo 6 ;;
  esac
}

sixcond_has_rates() {
  local tag="$1"
  local j="$SYNC_DIR/$tag/sixcond.json"
  [[ -f "$j" ]] || return 1
  TAG="$tag" JSON="$j" python3 - <<'PY'
import json, os, sys
from pathlib import Path
r = json.loads(Path(os.environ["JSON"]).read_text()).get("rates", {})
sys.exit(0 if any(v is not None for v in r.values()) else 1)
PY
}

results_has_rates() {
  local tag="$1"
  local min_count="${2:-1}"
  [[ -f "$RESULTS_JSON" ]] || return 1
  TAG="$tag" RESULTS_JSON="$RESULTS_JSON" MIN_COUNT="$min_count" python3 - <<'PY'
import json, os, sys
from pathlib import Path
tag = os.environ["TAG"]
min_count = int(os.environ["MIN_COUNT"])
data = json.loads(Path(os.environ["RESULTS_JSON"]).read_text())
rates = {}
for e in data.get("entries", []):
    if e.get("tag") == tag:
        rates = e.get("rates") or {}
        break
n = sum(1 for v in rates.values() if v is not None)
sys.exit(0 if n >= min_count else 1)
PY
}

tag_is_training() {
  pgrep -f "wandb_display_name=$1" >/dev/null 2>&1
}

should_skip_entry() {
  local tag="$1"
  if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
    return 1
  fi
  # Default: any prior rates ⇒ skip (gap-fill script covers missing README conditions).
  # Set REQUIRE_FULL_EVAL=1 to re-run when rates < category size (e.g. 6/17).
  local need=1
  if [[ "${REQUIRE_FULL_EVAL:-0}" == "1" ]]; then
    need="$(expected_eval_cond_count)"
  fi
  if sixcond_has_rates "$tag" || results_has_rates "$tag" "$need"; then
    log "SKIP $tag (eval rates already present; need>=$need)"
    return 0
  fi
  if tag_is_training "$tag"; then
    log "SKIP $tag (already training — likely on another GPU worker)"
    return 0
  fi
  return 1
}

stop_old_rollout_jobs() {
  log "Stopping old rollout workers (sequential mode)..."
  local self=$$ p
  for p in $(pgrep -f "parallel_rollout_pipeline.sh" 2>/dev/null); do
    [[ "$p" == "$self" || "$p" == "$PPID" ]] && continue
    kill "$p" 2>/dev/null || true
  done
  pkill -f "sequential_rollout_pipeline.sh" 2>/dev/null || true
  pkill -f "launch_anderson_rollout_sweep.sh" 2>/dev/null || true
  pkill -f "post_train_sixcond_pipeline.sh" 2>/dev/null || true
  pkill -f "python train.py.*wandb_display_name=dinowm_n" 2>/dev/null || true
  pkill -f "python train.py.*wandb_display_name=dinovicreg_n" 2>/dev/null || true
  pkill -f "python train.py.*wandb_display_name=dinobisim_n" 2>/dev/null || true
  pkill -f "python train.py.*wandb_display_name=bisim_n" 2>/dev/null || true
  pkill -f "eval_pusht_six_conditions.py" 2>/dev/null || true
  sleep 2
}

wait_for_gpu() {
  local gpu="${1:-$GPU}"
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 0)
    if (( free >= MIN_GPU_FREE_MIB )); then
      log "GPU $gpu ready (${free} MiB free)"
      return 0
    fi
    log "GPU $gpu busy (${free} MiB free, need ${MIN_GPU_FREE_MIB}) — waiting ${GPU_WAIT_SEC}s"
    sleep "$GPU_WAIT_SEC"
  done
}

resume_info() {
  case "$1" in
    dinowm_n5000) echo "2026-08-08/01-30-34|50" ;;
    dinovicreg_n5000) echo "2026-08-08/01-36-19|37" ;;
    bisim_n5000|dinobisim_n5000) echo "2026-08-08/01-36-12|38" ;;
    dinowm_nfull) echo "2026-08-08/21-46-38|50" ;;
    dinovicreg_nfull) echo "2026-08-08/22-03-21|50" ;;
    dinobisim_nfull) echo "2026-08-08/22-03-17|50" ;;
    # Aug 9+ runs under outputs/by_tag/<tag> (retrain from scratch when FRESH_TRAIN=1)
    dinowm_n1000|dinovicreg_n1000|bisim_n1000|dinobisim_n1000) echo "|0" ;;
    *) echo "|0" ;;
  esac
}

resolve_steps_per_epoch() {
  case "${STEPS_PER_EPOCH:-}" in
    n1000|N1000) echo "${STEPS_PER_EPOCH_N1000}" ;;
    *) echo "${STEPS_PER_EPOCH:-}" ;;
  esac
}

# Extra Hydra overrides keyed by wandb tag (e.g. VicReg baselines).
train_extra_args() {
  local args=()
  local steps
  steps="$(resolve_steps_per_epoch)"
  # null/full/none = exhaustive TrajSlicerDataset (June 2026 paper runs).
  case "${steps}" in
    ""|null|none|full)
      args+=( "training.steps_per_epoch=null" )
      ;;
    *)
      args+=( "training.steps_per_epoch=${steps}" )
      args+=( "training.val_steps_per_epoch=${VAL_STEPS_PER_EPOCH:-200}" )
      ;;
  esac
  if [[ "${USE_PRECOMPUTED_DINO:-false}" == "true" || "${USE_PRECOMPUTED_DINO:-false}" == "1" ]]; then
    args+=( "use_precomputed_dino=true" )
    if [[ -n "${DINO_FEATURES_ROOT:-}" ]]; then
      args+=( "env.dataset.dino_features_root=${DINO_FEATURES_ROOT}/pusht_noise" )
    fi
  else
    args+=( "use_precomputed_dino=false" )
  fi
  if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
    args+=( "training.batch_size=${TRAIN_BATCH_SIZE}" )
  fi
  if [[ "${SKIP_OPENLOOP_ROLLOUT:-1}" == "1" ]]; then
    args+=( "training.skip_openloop_rollout=true" )
  fi
  case "$1" in
    dinovicreg_*|*_vicreg) args+=( "regularization=vicreg" ) ;;
  esac
  case "$1" in
    dinobisim_*|dinovicreg_*|bisim_*)
      args+=( "bisim_pos_encoding=${BISIM_POS_ENCODING:-grid_mlp}" )
      ;;
  esac
  echo "${args[*]}"
}

update_manifest_entry() {
  local tag="$1"
  local model_name="$2"
  local sixcond_json="$3"
  local ckpt_base="$4"
  local eval_note="${5:-}"
  TAG="$tag" MODEL_NAME="$model_name" SIXCOND_JSON="$sixcond_json" EVAL_NOTE="$eval_note" \
    MANIFEST="$MANIFEST" RESULTS_JSON="$RESULTS_JSON" CKPT_BASE="$ckpt_base" python3 - <<'PY'
import json
import os
from pathlib import Path
manifest = Path(os.environ["MANIFEST"])
results = Path(os.environ["RESULTS_JSON"])
sixcond = os.environ.get("SIXCOND_JSON", "")
tag = os.environ["TAG"]
model_name = os.environ["MODEL_NAME"]
ckpt_base = os.environ["CKPT_BASE"]
eval_note = os.environ.get("EVAL_NOTE", "")
data = json.loads(manifest.read_text()) if manifest.is_file() else {"entries": []}
rates = {}
if sixcond and Path(sixcond).is_file():
    rates = json.loads(Path(sixcond).read_text()).get("rates", {})
entry = None
for e in data.get("entries", []):
    if e.get("tag") == tag:
        entry = e
        break
if entry is None:
    entry = {"tag": tag}
    data.setdefault("entries", []).append(entry)
entry["model_name"] = model_name
entry["ckpt_base_path"] = ckpt_base
entry["rates"] = rates
vals = [v for v in rates.values() if v is not None]
entry["eval_status"] = "done" if vals else "failed"
if eval_note:
    entry["eval_note"] = eval_note
vals = [v for v in rates.values() if v is not None]
entry["mean_rate"] = sum(vals)/len(vals) if vals else None
manifest.write_text(json.dumps(data, indent=2) + "\n")
results.write_text(json.dumps(data, indent=2) + "\n")
PY
  python3 scripts/update_readme_pusht_results.py --results-json "$RESULTS_JSON" 2>/dev/null || true
}

eval_and_archive() {
  local tag="$1"
  local model_name="$2"
  local ckpt_base="$3"
  local eval_note="${4:-}"
  local model_epoch="${5:-latest}"
  local gpu="${6:-$GPU}"

  wait_for_gpu "$gpu"
  local out_json="${PLAN_OUTPUTS:-$REPO/plan_outputs}/sixcond_${tag}_$(date +%Y%m%d%H%M%S).json"
  local eval_log="$LOG_DIR/eval_${tag}_gpu${gpu}.log"
  mkdir -p "$(dirname "$out_json")"
  if [[ -n "$eval_note" ]]; then
    log "EVAL sixcond tag=$tag model=$model_name epoch=$model_epoch gpu=$gpu ($eval_note)"
  else
    log "EVAL sixcond tag=$tag model=$model_name epoch=$model_epoch gpu=$gpu"
  fi
  log "EVAL progress: tail -f $eval_log"
  set +e
  EVAL_CATEGORY="$(resolve_eval_category)"
  local n_expect
  n_expect="$(expected_eval_cond_count)"
  log "EVAL category=$EVAL_CATEGORY ($n_expect conds × ~20–60+ min; ~${n_expect}–$((n_expect * 2))h/model is normal)"
  CEM_BS="${CEM_EPISODE_BATCH_SIZE:-1}"
  CEM_NS="${CEM_NUM_SAMPLES:-300}"
  local eval_extra=(
    model_epoch="$model_epoch" n_evals=50 goal_H=5 goal_source=dset planner.max_iter=5
    planner.sub_planner.episode_batch_size="$CEM_BS"
    planner.sub_planner.num_samples="$CEM_NS"
  )
  if [[ "${USE_PRECOMPUTED_DINO_EVAL}" == "true" || "${USE_PRECOMPUTED_DINO_EVAL}" == "1" ]]; then
    eval_extra+=( use_precomputed_dino_eval=true )
    if [[ -n "${DINO_FEATURES_ROOT:-}" ]]; then
      eval_extra+=( "dino_features_root=${DINO_FEATURES_ROOT}/pusht_noise" )
    fi
  fi
  DATASET_DIR="$DATASET_DIR" EVAL_CATEGORY="$EVAL_CATEGORY" CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONUNBUFFERED=1 python eval_pusht_six_conditions.py \
    --model-name "$model_name" \
    --ckpt-base-path "$ckpt_base" \
    --config-name plan_pusht_local \
    --category "$EVAL_CATEGORY" \
    --no-wandb \
    --output-json "$out_json" \
    --extra "${eval_extra[@]}" \
    >> "$eval_log" 2>&1
  local erc=$?
  set -e
  local sixcond_json=""
  if [[ $erc -eq 0 && -f "$out_json" ]]; then
    if TAG="$tag" JSON="$out_json" python3 - <<'PY'
import json, os, sys
from pathlib import Path
r = json.loads(Path(os.environ["JSON"]).read_text()).get("rates", {})
sys.exit(0 if any(v is not None for v in r.values()) else 1)
PY
    then
      sixcond_json="$out_json"
    else
      log "EVAL FAILED tag=$tag (no success rates in $out_json)"
    fi
  else
    log "EVAL FAILED tag=$tag rc=$erc"
  fi

  update_manifest_entry "$tag" "$model_name" "$sixcond_json" "$ckpt_base" "$eval_note"

  if [[ "${SKIP_ARCHIVE:-0}" == "1" ]]; then
    log "SKIP archive tag=$tag (SKIP_ARCHIVE=1)"
    return 0
  fi

  local dest="$SYNC_DIR/$tag"
  rm -rf "$dest"
  mkdir -p "$dest/run/checkpoints"
  echo "$model_name" > "$dest/model_path.txt"
  echo "$ckpt_base" > "$dest/ckpt_base_path.txt"
  [[ -n "$eval_note" ]] && echo "$eval_note" > "$dest/eval_note.txt"
  [[ -n "$sixcond_json" && -f "$sixcond_json" ]] && cp "$sixcond_json" "$dest/sixcond.json"
  rsync -a "${ckpt_base}/outputs/$model_name/hydra.yaml" "$dest/run/" 2>/dev/null || \
    rsync -a "${ckpt_base}/outputs/$model_name/.hydra" "$dest/run/" 2>/dev/null || true
  local latest_ckpt="${ckpt_base}/outputs/$model_name/checkpoints/model_latest.pth"
  if [[ -f "$latest_ckpt" ]]; then
    cp -a "$latest_ckpt" "$dest/run/checkpoints/model_latest.pth" 2>/dev/null || \
      rsync -a "$latest_ckpt" "$dest/run/checkpoints/" 2>/dev/null || true
  fi
  rsync -a "${ckpt_base}/outputs/$model_name/training_loss_log.csv" "$dest/run/" 2>/dev/null || true
  rm -f "$SYNC_DIR/${tag}.PULLED"
  log "ARCHIVED tag=$tag -> $dest ($(du -sh "$dest" 2>/dev/null | cut -f1))"
}

ensure_bisim_nfull_layout() {
  local src="$BISIM_NFULL_CKPT_BASE/$BISIM_NFULL_MODEL"
  local dest="$BISIM_NFULL_CKPT_BASE/outputs/$BISIM_NFULL_MODEL"
  local ckpt="$src/checkpoints/model_${BISIM_NFULL_EPOCH}.pth"
  if [[ ! -f "$ckpt" ]]; then
    log "SKIP bisim_nfull — missing $ckpt"
    return 1
  fi
  mkdir -p "$(dirname "$dest")"
  if [[ ! -e "$dest" ]]; then
    ln -sfn "$src" "$dest"
    log "Linked $dest -> $src"
  fi
  return 0
}

eval_bisim_full_from_checkpoints() {
  local gpu="${1:-$GPU}"
  if sixcond_has_rates "bisim_nfull"; then
    log "SKIP bisim_nfull (valid sixcond already archived)"
    return 0
  fi
  rm -rf "$SYNC_DIR/bisim_nfull"
  ensure_bisim_nfull_layout || return 0
  log "EVAL bisim_nfull (DINO-Bisim full) from $BISIM_NFULL_CKPT_BASE model=$BISIM_NFULL_MODEL epoch=$BISIM_NFULL_EPOCH gpu=$gpu"
  eval_and_archive "bisim_nfull" "$BISIM_NFULL_MODEL" "$BISIM_NFULL_CKPT_BASE" \
    "DINO-Bisim full from checkpoints_captialone/" "$BISIM_NFULL_EPOCH" "$gpu"
  log "DONE bisim_nfull checkpoints_captialone eval"
}

by_tag_saved_epoch() {
  local tag="$1"
  local ckpt_dir="$OUTPUTS_ROOT/by_tag/$tag/checkpoints"
  local latest=0 f base n
  shopt -s nullglob
  for f in "$ckpt_dir"/model_[0-9]*.pth; do
    base=$(basename "$f" .pth)   # model_5
    n=${base#model_}
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    (( n > latest )) && latest=$n
  done
  shopt -u nullglob
  echo "$latest"
}

# Train only. Sets LAST_MODEL_NAME and writes $LOG_DIR/${tag}.model_name on success.
train_entry() {
  local tag="$1"
  local method="$2"
  local n_rollout="$3"
  local gpu="${4:-$GPU}"
  GPU="$gpu"
  LAST_MODEL_NAME=""

  wait_for_gpu "$gpu"

  local config=train_local
  [[ "$method" == "dinowm" ]] && config=train_dinowm_local
  local rollout_arg=""
  [[ "$n_rollout" != "full" ]] && rollout_arg="env.dataset.n_rollout=$n_rollout"

  local resume_model="" resume_epoch=0
  if [[ "${FRESH_TRAIN:-0}" != "1" ]]; then
    IFS='|' read -r resume_model resume_epoch <<< "$(resume_info "$tag")"
    resume_epoch="${resume_epoch:-0}"
  fi

  local logfile="$LOG_DIR/${tag}_gpu${gpu}.log"
  local train_epochs="$EPOCHS"
  local hydra_args=( "ckpt_base_path=$CKPT_BASE" )
  local model_name=""
  local by_tag_ckpt="$OUTPUTS_ROOT/by_tag/$tag/checkpoints/model_latest.pth"
  local needs_retrain=0
  [[ "${FORCE_TRAIN:-0}" == "1" ]] && needs_retrain=1
  [[ " ${RETRAIN_TAGS:-} " == *" $tag "* ]] && needs_retrain=1

  local extra_arg
  extra_arg="$(train_extra_args "$tag")"
  local extra_train_args=()
  if [[ -n "$extra_arg" ]]; then
    # shellcheck disable=SC2206
    extra_train_args=( $extra_arg )
  fi

  # Prefer resuming partial by_tag runs (model_latest exists but epoch < EPOCHS).
  local by_tag_epoch=0
  if [[ -f "$by_tag_ckpt" && "$needs_retrain" -eq 0 && "${FRESH_TRAIN:-0}" != "1" ]]; then
    by_tag_epoch="$(by_tag_saved_epoch "$tag")"
    if (( by_tag_epoch >= EPOCHS )); then
      log "SKIP train tag=$tag gpu=$gpu (by_tag/$tag already at epoch $by_tag_epoch/$EPOCHS)"
      model_name="by_tag/$tag"
    else
      resume_model="by_tag/$tag"
      resume_epoch="$by_tag_epoch"
      log "RESUME by_tag tag=$tag from epoch $by_tag_epoch (target $EPOCHS)"
    fi
  fi

  if [[ -n "$model_name" ]]; then
    : # already skipped
  elif [[ -n "$resume_model" && -f "$OUTPUTS_ROOT/$resume_model/checkpoints/model_latest.pth" ]]; then
    # train.py treats training.epochs as absolute end epoch (resume-safe: epoch 36 + epochs=50 → 37..50).
    if (( resume_epoch >= EPOCHS )); then
      log "SKIP train tag=$tag already at epoch $resume_epoch (target $EPOCHS)"
      model_name="$resume_model"
    elif tag_is_training "$tag"; then
      log "WAIT tag=$tag already training on another process"
      while tag_is_training "$tag"; do sleep 60; done
      model_name="$resume_model"
    else
      hydra_args+=( "hydra.run.dir=$OUTPUTS_ROOT/$resume_model" )
      log "RESUME tag=$tag dir=$resume_model epoch=$resume_epoch target_epochs=$EPOCHS gpu=$gpu"
      log "TRAIN tag=$tag config=$config rollout=${rollout_arg:-full} epochs=$EPOCHS gpu=$gpu"
      set +e
      CUDA_VISIBLE_DEVICES="$gpu" python train.py --config-name "$config" \
        training.epochs="$EPOCHS" \
        training.save_every_x_epoch="$SAVE_EVERY" \
        training.log_image_mosaics_to_wandb=false \
        env.num_workers="$TRAIN_NUM_WORKERS" \
        ${rollout_arg:+"$rollout_arg"} \
        wandb_display_name="$tag" \
        "${extra_train_args[@]}" \
        "${hydra_args[@]}" \
        > "$logfile" 2>&1
      local rc=$?
      set -e
      if [[ $rc -ne 0 ]]; then
        log "TRAIN FAILED tag=$tag rc=$rc log=$logfile"
        return "$rc"
      fi
    fi
  elif tag_is_training "$tag"; then
    log "WAIT tag=$tag already training"
    while tag_is_training "$tag"; do sleep 60; done
    model_name=$(grep -m1 "Model saved dir:" "$logfile" 2>/dev/null | sed "s|.*/outputs/||" || true)
    [[ -z "$model_name" && -n "$resume_model" ]] && model_name="$resume_model"
  else
    [[ -n "$resume_model" ]] && log "WARN resume dir missing for tag=$tag ($resume_model) — fresh train on T7"
    # Unique run dir per tag avoids Hydra timestamp collisions when multiple jobs start together.
    hydra_args+=( "hydra.run.dir=$OUTPUTS_ROOT/by_tag/$tag" )
    log "TRAIN tag=$tag config=$config rollout=${rollout_arg:-full} epochs=$train_epochs gpu=$gpu dir=by_tag/$tag extra=${extra_train_args[*]:-none}"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python train.py --config-name "$config" \
      training.epochs="$train_epochs" \
      training.save_every_x_epoch="$SAVE_EVERY" \
      training.log_image_mosaics_to_wandb=false \
      env.num_workers="$TRAIN_NUM_WORKERS" \
      ${rollout_arg:+"$rollout_arg"} \
      wandb_display_name="$tag" \
        "${extra_train_args[@]}" \
        "${hydra_args[@]}" \
        > "$logfile" 2>&1
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      log "TRAIN FAILED tag=$tag rc=$rc log=$logfile"
      return "$rc"
    fi
  fi

  if [[ -z "$model_name" ]]; then
    model_name=$(grep -m1 "Model saved dir:" "$logfile" 2>/dev/null | sed "s|.*/outputs/||" || true)
  fi
  if [[ -f "$OUTPUTS_ROOT/by_tag/$tag/checkpoints/model_latest.pth" ]]; then
    model_name="by_tag/$tag"
  fi
  if [[ -z "$model_name" && -n "$resume_model" ]]; then
    model_name="$resume_model"
  fi
  if [[ -z "$model_name" ]]; then
    model_name=$(find "$OUTPUTS_ROOT" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %P\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
  fi
  if [[ -z "$model_name" ]]; then
    log "TRAIN FAILED tag=$tag could not resolve model_name"
    return 1
  fi

  LAST_MODEL_NAME="$model_name"
  printf '%s\n' "$model_name" > "$LOG_DIR/${tag}.model_name"
  log "TRAIN DONE tag=$tag model=$model_name gpu=$gpu"
  return 0
}

eval_after_train() {
  local tag="$1"
  local model_name="$2"
  local gpu="${3:-$GPU}"
  GPU="$gpu"

  wait_for_gpu "$gpu"
  eval_and_archive "$tag" "$model_name" "$CKPT_BASE" "" "latest" "$gpu"

  if [[ "$PURGE_AFTER_PULL" == "1" ]]; then
    local deadline=$((SECONDS + WAIT_PULL_SEC))
    log "WAIT pull tag=$tag (max ${WAIT_PULL_SEC}s)"
    while [[ ! -f "$SYNC_DIR/${tag}.PULLED" ]]; do
      if (( SECONDS > deadline )); then
        log "WARN pull timeout tag=$tag"
        break
      fi
      sleep "$PULL_POLL_SEC"
    done
    if [[ -f "$SYNC_DIR/${tag}.PULLED" && -d "$OUTPUTS_ROOT/$model_name" ]]; then
      rm -rf "$OUTPUTS_ROOT/$model_name"
      rm -rf "$SYNC_DIR/$tag"
      rm -f "$SYNC_DIR/${tag}.PULLED"
      log "PURGED tag=$tag model=$model_name"
    fi
  fi
  log "DONE tag=$tag gpu=$gpu"
  df -h "$CKPT_BASE" / | tail -2 >> "$LOG_DIR/parallel.log"
}

process_entry() {
  local tag="$1"
  local method="$2"
  local n_rollout="$3"
  local gpu="${4:-$GPU}"
  train_entry "$tag" "$method" "$n_rollout" "$gpu" || return $?
  eval_after_train "$tag" "$LAST_MODEL_NAME" "$gpu"
}

setup_env() {
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  export DATASET_DIR
  export TMPDIR="${TMPDIR:-/dev/shm/gloria-bisjepa}"
  export WANDB_DIR="${WANDB_DIR:-$REPO/wandb}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  mkdir -p "$TMPDIR" "$WANDB_DIR"
  cd "$REPO"
  verify_dataset || exit 1
  mkdir -p "$LOG_DIR" "$SYNC_DIR" "$OUTPUTS_ROOT" "$(dirname "$RESULTS_JSON")"
}

acquire_lock() {
  local lock_path="$1"
  exec 8>"$lock_path"
  if ! flock -n 8; then
    echo "Another rollout worker is already running ($lock_path)" >&2
    exit 1
  fi
}

run_eval_only() {
  local tag="$1"
  local model_name="$2"
  local gpu="${3:-$GPU}"
  GPU="$gpu"
  acquire_lock "$LOG_DIR/parallel_gpu${gpu}.lock"
  setup_env
  log "=== eval_only tag=$tag model=$model_name gpu=$gpu ==="
  wait_for_gpu "$gpu"
  eval_and_archive "$tag" "$model_name" "$CKPT_BASE" "eval-only" "latest" "$gpu"
  log "=== eval_only done tag=$tag ==="
}

run_gpu_worker() {
  local gpu="$1"
  local tag="$2"
  local method="$3"
  local n_rollout="$4"
  GPU="$gpu"
  acquire_lock "$LOG_DIR/parallel_gpu${gpu}.lock"
  setup_env
  log "=== gpu worker start gpu=$gpu tag=$tag method=$method rollout=$n_rollout ==="
  if should_skip_entry "$tag"; then
    log "=== gpu worker done gpu=$gpu tag=$tag (skipped) ==="
    exit 0
  fi
  process_entry "$tag" "$method" "$n_rollout" "$gpu" || exit 1
  log "=== gpu worker done gpu=$gpu tag=$tag ==="
}

run_gpu_worker_multi() {
  local gpu="$1"
  shift
  GPU="$gpu"
  acquire_lock "$LOG_DIR/parallel_gpu${gpu}.lock"
  setup_env
  log "=== gpu worker multi start gpu=$gpu jobs=$* ==="
  local spec tag method n_rollout
  for spec in "$@"; do
    IFS=':' read -r tag method n_rollout <<< "$spec"
    log "--- job tag=$tag method=$method rollout=$n_rollout ---"
    if should_skip_entry "$tag"; then
      log "SKIP tag=$tag"
      continue
    fi
    process_entry "$tag" "$method" "$n_rollout" "$gpu" || exit 1
  done
  log "=== gpu worker multi done gpu=$gpu ==="
}

# Per GPU: train WM+Bisim in parallel → eval each serially → train+eval VicReg.
# Args: worker_wm_bisim_vicreg <gpu> <wm_spec> <bisim_spec> <vicreg_spec>
# Spec format: tag:method:n_rollout  (e.g. dinowm_n1000:dinowm:1000)
run_gpu_worker_wm_bisim_vicreg() {
  local gpu="$1"
  local wm_spec="${2:?wm_spec}"
  local bisim_spec="${3:?bisim_spec}"
  local vicreg_spec="${4:?vicreg_spec}"
  GPU="$gpu"
  acquire_lock "$LOG_DIR/parallel_gpu${gpu}.lock"
  setup_env
  log "=== gpu worker WM+Bisim|| → VicReg start gpu=$gpu wm=$wm_spec bisim=$bisim_spec vicreg=$vicreg_spec ==="

  local wm_tag wm_method wm_n bisim_tag bisim_method bisim_n vicreg_tag vicreg_method vicreg_n
  IFS=':' read -r wm_tag wm_method wm_n <<< "$wm_spec"
  IFS=':' read -r bisim_tag bisim_method bisim_n <<< "$bisim_spec"
  IFS=':' read -r vicreg_tag vicreg_method vicreg_n <<< "$vicreg_spec"

  local pid_wm="" pid_bisim="" rc_wm=0 rc_bisim=0
  local do_wm=1 do_bisim=1
  if should_skip_entry "$wm_tag"; then
    log "SKIP train+eval $wm_tag"
    do_wm=0
  fi
  if should_skip_entry "$bisim_tag"; then
    log "SKIP train+eval $bisim_tag"
    do_bisim=0
  fi

  if [[ "$do_wm" -eq 1 || "$do_bisim" -eq 1 ]]; then
    log "--- parallel TRAIN phase: WM+Bisim on gpu=$gpu ---"
    if [[ "$do_wm" -eq 1 ]]; then
      train_entry "$wm_tag" "$wm_method" "$wm_n" "$gpu" &
      pid_wm=$!
      log "  background TRAIN $wm_tag pid=$pid_wm"
      # Let the first job allocate VRAM before the second wait_for_gpu.
      sleep "${PARALLEL_TRAIN_STAGGER_SEC:-45}"
    fi
    if [[ "$do_bisim" -eq 1 ]]; then
      train_entry "$bisim_tag" "$bisim_method" "$bisim_n" "$gpu" &
      pid_bisim=$!
      log "  background TRAIN $bisim_tag pid=$pid_bisim"
    fi
    if [[ -n "$pid_wm" ]]; then
      set +e
      wait "$pid_wm"
      rc_wm=$?
      set -e
      [[ $rc_wm -eq 0 ]] || log "TRAIN FAILED $wm_tag rc=$rc_wm"
    fi
    if [[ -n "$pid_bisim" ]]; then
      set +e
      wait "$pid_bisim"
      rc_bisim=$?
      set -e
      [[ $rc_bisim -eq 0 ]] || log "TRAIN FAILED $bisim_tag rc=$rc_bisim"
    fi
    [[ $rc_wm -eq 0 && $rc_bisim -eq 0 ]] || exit 1

    log "--- serial EVAL phase: WM then Bisim on gpu=$gpu ---"
    if [[ "$do_wm" -eq 1 ]]; then
      local wm_model
      wm_model="$(cat "$LOG_DIR/${wm_tag}.model_name" 2>/dev/null || true)"
      [[ -n "$wm_model" ]] || wm_model="by_tag/$wm_tag"
      eval_after_train "$wm_tag" "$wm_model" "$gpu" || exit 1
    fi
    if [[ "$do_bisim" -eq 1 ]]; then
      local bisim_model
      bisim_model="$(cat "$LOG_DIR/${bisim_tag}.model_name" 2>/dev/null || true)"
      [[ -n "$bisim_model" ]] || bisim_model="by_tag/$bisim_tag"
      eval_after_train "$bisim_tag" "$bisim_model" "$gpu" || exit 1
    fi
  fi

  log "--- VicReg phase (train+eval) on gpu=$gpu ---"
  if should_skip_entry "$vicreg_tag"; then
    log "SKIP $vicreg_tag"
  else
    process_entry "$vicreg_tag" "$vicreg_method" "$vicreg_n" "$gpu" || exit 1
  fi

  log "=== gpu worker WM+Bisim|| → VicReg done gpu=$gpu ==="
}

# Per GPU: train WM+Bisim in parallel, then eval WM∥Bisim in parallel (no VicReg).
# Args: worker_wm_bisim_parallel <gpu> <wm_spec> <bisim_spec>
run_gpu_worker_wm_bisim_parallel() {
  local gpu="$1"
  local wm_spec="${2:?wm_spec}"
  local bisim_spec="${3:?bisim_spec}"
  GPU="$gpu"
  acquire_lock "$LOG_DIR/parallel_gpu${gpu}.lock"
  setup_env
  log "=== gpu worker WM+Bisim|| start gpu=$gpu wm=$wm_spec bisim=$bisim_spec ==="

  local wm_tag wm_method wm_n bisim_tag bisim_method bisim_n
  IFS=':' read -r wm_tag wm_method wm_n <<< "$wm_spec"
  IFS=':' read -r bisim_tag bisim_method bisim_n <<< "$bisim_spec"

  local pid_wm="" pid_bisim="" rc_wm=0 rc_bisim=0
  local do_wm=1 do_bisim=1
  if should_skip_entry "$wm_tag"; then
    log "SKIP train+eval $wm_tag"
    do_wm=0
  fi
  if should_skip_entry "$bisim_tag"; then
    log "SKIP train+eval $bisim_tag"
    do_bisim=0
  fi

  if [[ "$do_wm" -eq 1 || "$do_bisim" -eq 1 ]]; then
    log "--- parallel TRAIN phase: WM+Bisim on gpu=$gpu ---"
    if [[ "$do_wm" -eq 1 ]]; then
      train_entry "$wm_tag" "$wm_method" "$wm_n" "$gpu" &
      pid_wm=$!
      log "  background TRAIN $wm_tag pid=$pid_wm"
      sleep "${PARALLEL_TRAIN_STAGGER_SEC:-20}"
    fi
    if [[ "$do_bisim" -eq 1 ]]; then
      train_entry "$bisim_tag" "$bisim_method" "$bisim_n" "$gpu" &
      pid_bisim=$!
      log "  background TRAIN $bisim_tag pid=$pid_bisim"
    fi
    if [[ -n "$pid_wm" ]]; then
      set +e
      wait "$pid_wm"
      rc_wm=$?
      set -e
      [[ $rc_wm -eq 0 ]] || log "TRAIN FAILED $wm_tag rc=$rc_wm"
    fi
    if [[ -n "$pid_bisim" ]]; then
      set +e
      wait "$pid_bisim"
      rc_bisim=$?
      set -e
      [[ $rc_bisim -eq 0 ]] || log "TRAIN FAILED $bisim_tag rc=$rc_bisim"
    fi
    [[ $rc_wm -eq 0 && $rc_bisim -eq 0 ]] || exit 1

    log "--- parallel EVAL phase: WM∥Bisim on gpu=$gpu ---"
    local pid_ewm="" pid_ebisim="" rc_ewm=0 rc_ebisim=0
    if [[ "$do_wm" -eq 1 ]]; then
      local wm_model
      wm_model="$(cat "$LOG_DIR/${wm_tag}.model_name" 2>/dev/null || true)"
      [[ -n "$wm_model" ]] || wm_model="by_tag/$wm_tag"
      eval_after_train "$wm_tag" "$wm_model" "$gpu" &
      pid_ewm=$!
      log "  background EVAL $wm_tag pid=$pid_ewm"
      sleep "${PARALLEL_EVAL_STAGGER_SEC:-30}"
    fi
    if [[ "$do_bisim" -eq 1 ]]; then
      local bisim_model
      bisim_model="$(cat "$LOG_DIR/${bisim_tag}.model_name" 2>/dev/null || true)"
      [[ -n "$bisim_model" ]] || bisim_model="by_tag/$bisim_tag"
      eval_after_train "$bisim_tag" "$bisim_model" "$gpu" &
      pid_ebisim=$!
      log "  background EVAL $bisim_tag pid=$pid_ebisim"
    fi
    if [[ -n "$pid_ewm" ]]; then
      set +e
      wait "$pid_ewm"
      rc_ewm=$?
      set -e
      [[ $rc_ewm -eq 0 ]] || log "EVAL FAILED $wm_tag rc=$rc_ewm"
    fi
    if [[ -n "$pid_ebisim" ]]; then
      set +e
      wait "$pid_ebisim"
      rc_ebisim=$?
      set -e
      [[ $rc_ebisim -eq 0 ]] || log "EVAL FAILED $bisim_tag rc=$rc_ebisim"
    fi
    [[ $rc_ewm -eq 0 && $rc_ebisim -eq 0 ]] || exit 1
  fi

  log "=== gpu worker WM+Bisim|| done gpu=$gpu ==="
}

run_sequential_pipeline() {
  acquire_lock "$LOG_DIR/parallel.lock"
  setup_env
  stop_old_rollout_jobs

  log "=== parallel_rollout_pipeline start GPU=$GPU ckpt=$CKPT_BASE DATASET_DIR=$DATASET_DIR epochs=$EPOCHS save_every=$SAVE_EVERY (sequential) ==="
  df -h "$CKPT_BASE" / >> "$LOG_DIR/parallel.log"

  if should_skip_entry "dinowm_n1000"; then
    :
  else
    process_entry "dinowm_n1000" "dinowm" "1000" "$GPU" || exit 1
  fi
  eval_bisim_full_from_checkpoints "$GPU"
  if should_skip_entry "bisim_n5000"; then
    if tag_is_training "bisim_n5000"; then
      log "WAIT bisim_n5000 training (GPU worker or existing job)"
      while tag_is_training "bisim_n5000"; do sleep 120; done
      if ! sixcond_has_rates "bisim_n5000"; then
        process_entry "bisim_n5000" "bisim" "5000" "$GPU" || exit 1
      fi
    fi
  else
    process_entry "bisim_n5000" "bisim" "5000" "$GPU" || exit 1
  fi
  if should_skip_entry "dinowm_n5000"; then
    :
  else
    process_entry "dinowm_n5000" "dinowm" "5000" "$GPU" || exit 1
  fi
  if should_skip_entry "dinowm_nfull"; then
    :
  else
    process_entry "dinowm_nfull" "dinowm" "full" "$GPU" || exit 1
  fi

  log "=== parallel_rollout_pipeline done ==="
}

if [[ "${1:-}" == "eval_only" ]]; then
  run_eval_only "${2:?tag}" "${3:?model_name}" "${4:-$GPU}"
  exit 0
fi

if [[ "${1:-}" == "worker" ]]; then
  run_gpu_worker "${2:?gpu}" "${3:?tag}" "${4:?method}" "${5:?n_rollout}"
  exit 0
fi

if [[ "${1:-}" == "worker_multi" ]]; then
  run_gpu_worker_multi "${2:?gpu}" "${@:3}"
  exit 0
fi

if [[ "${1:-}" == "worker_wm_bisim_vicreg" ]]; then
  run_gpu_worker_wm_bisim_vicreg "${2:?gpu}" "${3:?wm_spec}" "${4:?bisim_spec}" "${5:?vicreg_spec}"
  exit 0
fi

if [[ "${1:-}" == "worker_wm_bisim_parallel" ]]; then
  run_gpu_worker_wm_bisim_parallel "${2:?gpu}" "${3:?wm_spec}" "${4:?bisim_spec}"
  exit 0
fi

run_sequential_pipeline
