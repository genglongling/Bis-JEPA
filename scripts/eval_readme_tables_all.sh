#!/usr/bin/env bash
# Extra eval: 6 models × 4 mild conditions (T1, T2, V1, V2).
#
# Models (DINO-WM + Bisim × n1000/n5000/nfull):
#   dinowm_n1000 dinowm_n5000 dinowm_nfull
#   dinobisim_n1000 dinobisim_n5000 dinobisim_nfull
#
# Does NOT re-run legacy_six. Merges T1/T2/V1/V2 into rollout_sweep_results.json.
# Skips GPU busy with training (default: avoid GPU 1 while bisim_n1000 trains).
# Waits for a free GPU (≥ MIN_GPU_FREE_MIB).
#
# Usage (Anderson):
#   bash scripts/eval_readme_tables_all.sh
#   bash scripts/eval_readme_tables_all.sh dinowm_n1000 dinobisim_n5000
#
# From Mac:
#   bash scripts/eval_readme_tables_all.sh --remote
set -euo pipefail

REPO="${REPO:-$HOME/Bis-JEPA}"
REMOTE="${REMOTE:-anderson-accelerator}"

if [[ "${1:-}" == "--remote" ]]; then
  shift
  LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  RHOST="${REMOTE:-anderson-accelerator}"
  RREPO="${REMOTE_REPO:-~/Bis-JEPA}"
  scp "$LOCAL_REPO/scripts/eval_readme_tables_all.sh" \
    "$LOCAL_REPO/scripts/update_readme_pusht_results.py" \
    "$RHOST:${RREPO}/scripts/"
  scp "$LOCAL_REPO/eval_pusht_six_conditions.py" "$RHOST:${RREPO}/"
  scp "$LOCAL_REPO/env/visual_conditions.py" "$RHOST:${RREPO}/env/"
  scp "$LOCAL_REPO/env/pusht/pusht_env.py" "$RHOST:${RREPO}/env/pusht/"
  ssh -o BatchMode=yes "$RHOST" "bash ${RREPO}/scripts/eval_readme_tables_all.sh $*"
  exit 0
fi

CKPT_BASE="${CKPT_BASE:-/mnt/t7shield/gloria/bis-jepa}"
T7_SCRATCH="${T7_SCRATCH:-$CKPT_BASE/scratch}"
LOG_DIR="${LOG_DIR:-$CKPT_BASE/logs/rollout_sweep}"
RESULTS_JSON="${RESULTS_JSON:-$T7_SCRATCH/plan_outputs/rollout_sweep_results.json}"
PLAN_OUTPUTS="${PLAN_OUTPUTS:-$T7_SCRATCH/plan_outputs}"
DATASET_DIR="${DATASET_DIR:-$REPO/datasets/data}"
TMPDIR="${TMPDIR:-/dev/shm/gloria-bisjepa}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-20000}"
GPU_WAIT_SEC="${GPU_WAIT_SEC:-120}"
# Prefer free GPUs; leave GPU 1 for bisim_n1000 training unless overridden
GPUS="${GPUS:-0 2 1}"
CEM_EPISODE_BATCH_SIZE="${CEM_EPISODE_BATCH_SIZE:-1}"
CEM_NUM_SAMPLES="${CEM_NUM_SAMPLES:-300}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# Conditions: default mild_tv (T1 T2 V1 V2). Override with CONDS="NC SC C ..."
# For Bisim full sweep use: CONDS="NC SC C LC LCG D1 T1 T2 V1 V2" (legacy_six + mild_tv)
if [[ -n "${CONDS:-}" ]]; then
  # shellcheck disable=SC2206
  CONDS=($CONDS)
else
  CONDS=(T1 T2 V1 V2)
fi
if [[ $# -eq 0 ]]; then
  TAGS=(dinowm_n1000 dinowm_n5000 dinowm_nfull dinobisim_n1000 dinobisim_n5000 dinobisim_nfull)
else
  TAGS=("$@")
fi

mkdir -p "$LOG_DIR" "$PLAN_OUTPUTS" "$TMPDIR"
# Main pipeline stays on legacy_six; this script only adds mild_tv
printf 'legacy_six\n' > "$LOG_DIR/EVAL_CATEGORY"
export TMPDIR WANDB_MODE DATASET_DIR PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" >> "$LOG_DIR/eval_mild_tv.log"
  echo "$msg" >&2
}

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-dino_wm310}"
cd "$REPO"

gpu_has_our_train() {
  local g="$1" pid owner cmd
  while read -r pid; do
    pid="${pid// /}"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [[ "$owner" == "$USER" ]] || continue
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    if [[ "$cmd" == *train.py* ]]; then
      return 0
    fi
  done < <(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  return 1
}

pick_free_gpu() {
  local g free
  for g in $GPUS; do
    if gpu_has_our_train "$g"; then
      continue
    fi
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
    if (( free >= MIN_GPU_FREE_MIB )); then
      echo "$g"
      return 0
    fi
  done
  return 1
}

wait_any_gpu() {
  local g
  while true; do
    if g=$(pick_free_gpu); then
      log "GPU $g ready (${MIN_GPU_FREE_MIB}+ MiB free, no local train.py)"
      echo "$g"
      return 0
    fi
    log "No free GPU — waiting ${GPU_WAIT_SEC}s"
    sleep "$GPU_WAIT_SEC"
  done
}

missing_for_tag() {
  local tag="$1"
  TAG="$tag" RESULTS_JSON="$RESULTS_JSON" CONDS="${CONDS[*]}" python3 - <<'PY'
import json, os
from pathlib import Path
want = os.environ["CONDS"].split()
tag = os.environ["TAG"]
path = Path(os.environ["RESULTS_JSON"])
rates = {}
if path.is_file():
    data = json.loads(path.read_text())
    for e in data.get("entries", []):
        if e.get("tag") == tag:
            rates = e.get("rates") or {}
            break
missing = [c for c in want if rates.get(c) is None]
print(" ".join(missing))
PY
}

tag_ready_ckpt() {
  local tag="$1"
  local ckpt="$CKPT_BASE/outputs/by_tag/$tag/checkpoints/model_latest.pth"
  [[ -f "$ckpt" ]]
}

merge_rates() {
  local tag="$1"
  local new_json="$2"
  local model_name="$3"
  TAG="$tag" NEW_JSON="$new_json" MODEL_NAME="$model_name" \
    RESULTS_JSON="$RESULTS_JSON" CKPT_BASE="$CKPT_BASE" python3 - <<'PY'
import json, os
from pathlib import Path
tag = os.environ["TAG"]
new = json.loads(Path(os.environ["NEW_JSON"]).read_text())
new_rates = new.get("rates") or {}
results = Path(os.environ["RESULTS_JSON"])
data = json.loads(results.read_text()) if results.is_file() else {"entries": []}
entry = None
for e in data.get("entries", []):
    if e.get("tag") == tag:
        entry = e
        break
if entry is None:
    entry = {"tag": tag, "method": (
        "dinowm" if tag.startswith("dinowm") else "bisim"
    )}
    if "n1000" in tag:
        entry["n_rollout"] = 1000
    elif "n5000" in tag:
        entry["n_rollout"] = 5000
    else:
        entry["n_rollout"] = "full"
    data.setdefault("entries", []).append(entry)
rates = dict(entry.get("rates") or {})
if rates.get("D") is not None and rates.get("D1") is None:
    rates["D1"] = rates["D"]
rates.update({k: v for k, v in new_rates.items() if v is not None})
entry["rates"] = rates
entry["model_name"] = os.environ.get("MODEL_NAME") or entry.get("model_name") or f"by_tag/{tag}"
entry["ckpt_base_path"] = os.environ["CKPT_BASE"]
vals = [v for v in rates.values() if v is not None]
entry["eval_status"] = "done" if vals else "failed"
entry["mean_rate"] = sum(vals) / len(vals) if vals else None
results.parent.mkdir(parents=True, exist_ok=True)
results.write_text(json.dumps(data, indent=2) + "\n")
Path(os.environ["NEW_JSON"]).write_text(
    json.dumps({**new, "rates": rates, "tag": tag}, indent=2) + "\n"
)
print(f"merged {len(rates)} rates for {tag} -> {results}")
PY
  python3 "$REPO/scripts/update_readme_pusht_results.py" --results-json "$RESULTS_JSON" \
    --readme "$REPO/README.md" 2>/dev/null || true
}

log "=== eval start conditions=[${CONDS[*]}] tags=${TAGS[*]} ==="
log "MIN_GPU_FREE_MIB=$MIN_GPU_FREE_MIB RESULTS_JSON=$RESULTS_JSON"

for tag in "${TAGS[@]}"; do
  model_name="by_tag/$tag"
  if [[ -f "$LOG_DIR/${tag}.model_name" ]]; then
    model_name="$(tr -d '[:space:]' < "$LOG_DIR/${tag}.model_name")"
  fi
  if ! tag_ready_ckpt "$tag"; then
    log "WAIT $tag — no by_tag checkpoint yet"
    continue
  fi
  # Skip unfinished training tags (still writing checkpoints)
  if pgrep -f "wandb_display_name=${tag}" >/dev/null 2>&1; then
    log "SKIP $tag — currently training; will need re-run later"
    continue
  fi
  missing="$(missing_for_tag "$tag")"
  if [[ -z "$missing" ]]; then
    log "SKIP $tag (${CONDS[*]} already present)"
    continue
  fi
  log "EVAL $tag model=$model_name conditions=[$missing]"
  gpu="$(wait_any_gpu)"
  out_json="$PLAN_OUTPUTS/sixcond_${tag}_mildtv_$(date +%Y%m%d%H%M%S).json"
  eval_log="$LOG_DIR/eval_${tag}_mildtv_gpu${gpu}.log"
  log "  gpu=$gpu log=$eval_log"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" DATASET_DIR="$DATASET_DIR" \
    PYTHONUNBUFFERED=1 python eval_pusht_six_conditions.py \
    --model-name "$model_name" \
    --ckpt-base-path "$CKPT_BASE" \
    --config-name plan_pusht_local \
    --conditions $missing \
    --no-wandb \
    --output-json "$out_json" \
    --extra \
      model_epoch=latest n_evals=50 goal_H=5 goal_source=dset planner.max_iter=5 \
      "planner.sub_planner.episode_batch_size=$CEM_EPISODE_BATCH_SIZE" \
      "planner.sub_planner.num_samples=$CEM_NUM_SAMPLES" \
    >> "$eval_log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 || ! -f "$out_json" ]]; then
    log "FAILED $tag rc=$rc (see $eval_log)"
    continue
  fi
  merge_rates "$tag" "$out_json" "$model_name"
  log "DONE $tag mild_tv"
done

log "=== mild_tv eval pass done ==="
log "Re-run later for any SKIP/WAIT tags (e.g. dinobisim_n1000 still training)"
