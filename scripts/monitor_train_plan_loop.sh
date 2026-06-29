#!/usr/bin/env bash
# Wait for training, run PushT planning (NC SC C LC LCG D), retry until mean success > threshold.
set -uo pipefail

THRESHOLD="${SUCCESS_THRESHOLD:-0.5}"
VISUAL_CONDITIONS=(NC SC C LC LCG D)
REPO="${REPO:-$HOME/Bis-JEPA}"
LOG="${MONITOR_LOG:-$REPO/monitor_train_plan.log}"
STATE="${MONITOR_STATE:-$REPO/monitor_train_plan_state.json}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dino_wm310}"
DATASET_DIR="${DATASET_DIR:-$REPO/datasets/data}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# mode|n_rollout (null=full)|epochs|extra hydra overrides|tag
# Uses conf/train_local.yaml (32-d bisim, grid_mlp, batch 20, paper LRs); epochs passed explicitly.
ATTEMPTS=(
  "wait|1000|50|regularization=pca|paper32_n1000_e50"
  "train|null|50|regularization=pca|paper32_full_e50"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export DATASET_DIR CUDA_VISIBLE_DEVICES="$GPU"
cd "$REPO"

parse_sixcond_json() {
  local json_path="$1"
  python3 - <<'PY' "$json_path"
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
except FileNotFoundError:
    print("|||")
    raise SystemExit(0)
rates = data.get("rates") or {}
vals = [float(v) for v in rates.values() if v is not None]
mean = "" if not vals else sum(vals) / len(vals)
summary = " ".join(f"{k}={rates.get(k)}" for k in ("NC", "SC", "C", "LC", "LCG", "D") if k in rates)
print(f"{mean}|{summary}|{path}")
PY
}

run_planning() {
  local model_name="$1"
  local best_sr=0
  local best_epoch=""
  local best_json=""
  local best_summary=""

  for epoch in latest 50 40 30 20 10 5; do
    local tag
    tag=$(echo "$model_name" | tr '/' '_')
    local out_json="$REPO/plan_outputs/sixcond_monitor_${tag}_e${epoch}_$(date +%Y%m%d%H%M%S).json"
    log "Planning model_name=$model_name model_epoch=$epoch (${VISUAL_CONDITIONS[*]}, n_evals=50)"
    set +e
    python eval_pusht_six_conditions.py \
      --model-name "$model_name" \
      --ckpt-base-path "$REPO" \
      --config-name plan_pusht_local \
      --no-wandb \
      --output-json "$out_json" \
      --extra "model_epoch=$epoch" n_evals=50 goal_H=5 goal_source=dset planner.max_iter=5 \
      >> "$LOG" 2>&1
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      log "Six-condition planning failed epoch=$epoch rc=$rc"
      continue
    fi
    if [[ ! -f "$out_json" ]]; then
      log "No sixcond JSON for epoch=$epoch"
      continue
    fi
    parsed=$(parse_sixcond_json "$out_json")
    sr="${parsed%%|*}"
    rest="${parsed#*|}"
    summary="${rest%%|*}"
    json_path="${rest#*|}"
    log "epoch=$epoch mean_success_rate=${sr:-none} rates: ${summary:-none} json=$json_path"
    if [[ -n "$sr" ]]; then
      awk -v a="$sr" -v b="$best_sr" 'BEGIN{exit !(a>b)}' && best_sr="$sr" && best_epoch="$epoch" && best_json="$json_path" && best_summary="$summary"
      awk -v a="$best_sr" -v t="$THRESHOLD" 'BEGIN{exit !(a>t)}' && break
    fi
  done

  echo "$best_sr|$best_epoch|$best_json|$best_summary"
}

run_training() {
  local n_rollout="$1"
  local epochs="$2"
  local extra="$3"
  local tag="$4"
  local train_log="$REPO/train_${tag}.log"

  local cmd=(python train.py --config-name train_local "training.epochs=$epochs")
  if [[ "$n_rollout" != "null" ]]; then
    cmd+=( "env.dataset.n_rollout=$n_rollout" )
  fi
  # shellcheck disable=SC2206
  extra_parts=( $extra )
  cmd+=( "${extra_parts[@]}" )

  log "Starting training tag=$tag log=$train_log cmd=${cmd[*]}"
  nohup "${cmd[@]}" > "$train_log" 2>&1 &
  local pid=$!
  log "Training PID=$pid"

  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
    tail -1 "$train_log" 2>/dev/null | tee -a "$LOG" || true
  done
  wait "$pid" || true

  local model_dir
  model_dir=$(grep -m1 "Model saved dir:" "$train_log" | sed 's/.*Model saved dir: //')
  if [[ -z "$model_dir" ]]; then
    log "ERROR: could not parse Model saved dir from $train_log"
    return 1
  fi
  # YYYY-MM-DD/HH-MM-SS relative to outputs/
  echo "$model_dir" | sed "s|.*/outputs/||"
}

wait_for_pid() {
  local pid="$1"
  local train_log="${2:-$REPO/train_paper32_n1000_e50.log}"
  log "Waiting for training PID=$pid (log=$train_log)"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
    tail -1 "$train_log" 2>/dev/null | tee -a "$LOG" || true
  done
  log "PID $pid finished"
}

log "=== monitor_train_plan_loop start threshold=$THRESHOLD ==="

if [[ -n "${PLAN_ONLY_MODEL:-}" ]]; then
  model_name="${PLAN_ONLY_MODEL#/}"
  model_name="${model_name#outputs/}"
  log "PLAN_ONLY model_name=$model_name (conditions: ${VISUAL_CONDITIONS[*]})"
  result=$(run_planning "$model_name")
  best_sr="${result%%|*}"
  rest="${result#*|}"
  best_epoch="${rest%%|*}"
  rest="${rest#*|}"
  best_json="${rest%%|*}"
  best_summary="${rest#*|}"
  log "PLAN_ONLY done mean_rate=${best_sr:-0} epoch=$best_epoch ${best_summary:-}"
  if [[ -n "$best_sr" ]] && awk -v a="$best_sr" -v t="$THRESHOLD" 'BEGIN{exit !(a>t)}'; then
    log "SUCCESS: mean_rate=$best_sr > $THRESHOLD (epoch=$best_epoch)"
    exit 0
  fi
  log "Below threshold: mean_rate=${best_sr:-0} <= $THRESHOLD"
  exit 1
fi

attempt_idx=0
for spec in "${ATTEMPTS[@]}"; do
  IFS='|' read -r mode n_rollout epochs extra tag <<< "$spec"
  attempt_idx=$((attempt_idx + 1))
  log "--- Attempt $attempt_idx tag=$tag ---"

  if [[ "$mode" == "wait" ]]; then
    train_log="$REPO/train_${tag}.log"
    wait_pid="${WAIT_PID:-}"
    if [[ -z "$wait_pid" ]] && pgrep -f "python train.py --config-name train_local" >/dev/null 2>&1; then
      wait_pid=$(pgrep -f "python train.py --config-name train_local" | head -1)
      log "Auto-detected training PID=$wait_pid"
    fi
    if [[ -n "$wait_pid" ]] && kill -0 "$wait_pid" 2>/dev/null; then
      wait_for_pid "$wait_pid" "$train_log"
    else
      log "ERROR: wait mode but no live training PID (log=$train_log)"
      continue
    fi
    model_name=$(grep -m1 "Model saved dir:" "$train_log" 2>/dev/null | sed 's|.*/outputs/||' || true)
    if [[ -z "$model_name" ]]; then
      log "ERROR: could not parse model_name from $train_log"
      continue
    fi
  else
    model_name=$(run_training "$n_rollout" "$epochs" "$extra" "$tag")
  fi

  log "Evaluating model_name=$model_name"
  result=$(run_planning "$model_name")
  best_sr="${result%%|*}"
  rest="${result#*|}"
  best_epoch="${rest%%|*}"
  rest="${rest#*|}"
  best_json="${rest%%|*}"
  best_summary="${rest#*|}"

  python3 - <<PY >> "$STATE"
import json, datetime
rec = {
  "time": datetime.datetime.now().isoformat(),
  "attempt": $attempt_idx,
  "tag": "$tag",
  "model_name": "$model_name",
  "mean_success_rate": float("$best_sr") if "$best_sr" else None,
  "best_epoch": "$best_epoch",
  "sixcond_json": "$best_json",
  "rates_summary": "$best_summary",
  "conditions": ["NC", "SC", "C", "LC", "LCG", "D"],
}
print(json.dumps(rec))
try:
    with open("$STATE") as f:
        hist = json.load(f)
except Exception:
    hist = []
if not isinstance(hist, list):
    hist = [hist]
hist.append(rec)
with open("$STATE", "w") as f:
    json.dump(hist, f, indent=2)
PY

  if [[ -n "$best_sr" ]] && awk -v a="$best_sr" -v t="$THRESHOLD" 'BEGIN{exit !(a>t)}'; then
    log "SUCCESS: mean_rate=$best_sr > $THRESHOLD (epoch=$best_epoch) ${best_summary:-}"
    exit 0
  fi
  log "Below threshold: mean_rate=${best_sr:-0} <= $THRESHOLD (${best_summary:-}) — next attempt"
done

log "FAILED: exhausted attempts without reaching $THRESHOLD"
exit 1
