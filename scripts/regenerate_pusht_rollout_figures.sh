#!/usr/bin/env bash
# Scan val episodes, pick WM-fail + Bisim-success, rebuild cache, render figures.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export DATASET_DIR="${DATASET_DIR:-$REPO/datasets/data}"
export DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
export PLAN_DEVICE="${PLAN_DEVICE:-mps}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export PYGAME_HIDE_SUPPORT_PROMPT="${PYGAME_HIDE_SUPPORT_PROMPT:-1}"

PYTHON="${PYTHON:-$REPO/.venv_render_arm/bin/python}"
SCAN_DIR="$REPO/tmp_scan_pusht_ep"
RESULTS_JSON="$SCAN_DIR/results.json"
LOG="$REPO/docs/figures/rollout_examples/regenerate_pusht.log"
CHOSEN_EP_FILE="$SCAN_DIR/chosen_episode.txt"

mkdir -p "$SCAN_DIR"
mkdir -p "$(dirname "$LOG")"

exec > >(tee -a "$LOG") 2>&1
echo "=== PushT rollout regeneration started $(date) ==="

echo "[1/4] Scanning val episodes (NC, WM + Bisim)..."
"$PYTHON" "$REPO/scripts/scan_pusht_rollout_episode.py" \
  --out-json "$RESULTS_JSON"

echo "[2/4] Choosing episode..."
"$PYTHON" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path("tmp_scan_pusht_ep/results.json")
out_path = Path("tmp_scan_pusht_ep/chosen_episode.txt")
payload = json.loads(results_path.read_text())
rows = payload["results"]
candidates = payload.get("candidates") or []

if candidates:
    ep = candidates[0]
    reason = "WM fail + Bisim success"
else:
    wm_fail = [r for r in rows if r["wm_success"] == 0.0 and r["bisim_success"] == 1.0]
    if wm_fail:
        ep = wm_fail[0]["episode"]
        reason = "WM fail + Bisim success (fallback list)"
    else:
        bisim_ok = [r for r in rows if r["bisim_success"] == 1.0]
        if bisim_ok:
            bisim_ok.sort(key=lambda r: (r["wm_success"], r["wm_state_dist"]), reverse=True)
            ep = bisim_ok[0]["episode"]
            reason = "Bisim success (no strict WM-fail pair; best available)"
        else:
            rows.sort(key=lambda r: (r["bisim_state_dist"], r["wm_state_dist"]))
            ep = rows[0]["episode"]
            reason = "No successes found; lowest combined state dist"

out_path.write_text(str(ep))
print(f"Chosen episode {ep}: {reason}")
row = next(r for r in rows if r["episode"] == ep)
print(
    f"  WM success={row['wm_success']} state_dist={row['wm_state_dist']:.1f} | "
    f"Bisim success={row['bisim_success']} state_dist={row['bisim_state_dist']:.1f}"
)
if row["wm_success"] != 0.0 or row["bisim_success"] != 1.0:
    print("WARNING: chosen episode does not strictly match WM-fail + Bisim-success.", file=sys.stderr)
PY

EP="$(cat "$CHOSEN_EP_FILE")"
echo "Using val episode $EP"

echo "[3/4] Rebuilding PushT rollout cache (6 conditions × 2 figures)..."
"$PYTHON" "$REPO/scripts/build_rollout_cache.py" \
  --pusht-only \
  --pusht-episode "$EP"

echo "[3b/4] Backfilling agent paths (if needed)..."
"$PYTHON" "$REPO/scripts/backfill_pusht_rollout_paths.py"

echo "[4/4] Rendering rollout figures..."
"$PYTHON" "$REPO/scripts/render_pusht_visual_conditions.py" --rollout-figures

echo "=== PushT rollout regeneration finished $(date) ==="
echo "Episode: $EP"
echo "Log: $LOG"
