#!/usr/bin/env python3
"""Update README PushT rollout-sweep tables from plan_outputs/rollout_sweep_results.json."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_visual_conditions():
    path = REPO_ROOT / "env" / "visual_conditions.py"
    spec = importlib.util.spec_from_file_location("visual_conditions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vc = _load_visual_conditions()
COLOR_CONDITIONS = _vc.COLOR_CONDITIONS
TEXTURE_CONDITIONS = _vc.TEXTURE_CONDITIONS
VIEWPOINT_CONDITIONS = _vc.VIEWPOINT_CONDITIONS
DYNAMIC_CONDITIONS = _vc.DYNAMIC_CONDITIONS
category_mean = _vc.category_mean

README = REPO_ROOT / "README.md"
START = "<!-- pusht-rollout-results -->"
END = "<!-- /pusht-rollout-results -->"


def fmt_rate(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def rollout_label(n: Any) -> str:
    if n is None or n == "full":
        return "full"
    return str(n)


def method_label(method: str) -> str:
    return {
        "bisim": "DINO-Bisim",
        "vicreg": "DINO-VICReg",
        "dinowm": "DINO-WM",
    }.get(method, method)


def sort_key(e: Dict[str, Any]) -> tuple:
    nr = e.get("n_rollout")
    if nr == "full" or nr is None:
        nr_sort = 10**9
    else:
        nr_sort = int(nr)
    method = e.get("method", "bisim")
    method_order = {"bisim": 0, "vicreg": 1, "dinowm": 2}.get(method, 3)
    return (nr_sort, method_order, e.get("tag", ""))


def ckpt_cell(e: Dict[str, Any], rates: Dict[str, Any]) -> str:
    ckpt = e.get("model_name") or "—"
    status = e.get("eval_status", "")
    if status == "failed":
        return f"{ckpt} (eval failed)"
    if status == "pending" or not rates:
        return f"{ckpt} (pending eval)" if ckpt != "—" else "— (pending eval)"
    return ckpt


def build_category_table(
    entries: List[Dict[str, Any]],
    title: str,
    conditions: Tuple[str, ...],
) -> List[str]:
    header = (
        "| Train rollouts | Method | "
        + " | ".join(conditions)
        + " | Mean | Checkpoint |"
    )
    sep = (
        "|----------------|--------|"
        + "|".join("-----" for _ in conditions)
        + "|------|------------|"
    )
    lines = ["", f"#### {title}", "", header, sep]
    for e in sorted(entries, key=sort_key):
        rates = e.get("rates") or {}
        m = category_mean(rates, conditions)
        cells = [fmt_rate(rates.get(c)) for c in conditions]
        lines.append(
            "| {roll} | {method} | {cells} | {mean} | `{ckpt}` |".format(
                roll=rollout_label(e.get("n_rollout")),
                method=method_label(e.get("method", "bisim")),
                cells=" | ".join(cells),
                mean=fmt_rate(m),
                ckpt=ckpt_cell(e, rates),
            )
        )
    return lines


PREV_START = "#### Previous checkpoints"


def extract_previous_block(readme_text: str) -> Optional[str]:
    """Keep hand-maintained previous-checkpoint tables across auto-updates."""
    if PREV_START not in readme_text:
        return None
    # From #### Previous checkpoints through the blank line before #### Anderson
    # or before #### 1. / END marker.
    m = re.search(
        re.escape(PREV_START) + r".*?(?=\n#### Anderson|\n#### 1\.|\n" + re.escape(END) + r")",
        readme_text,
        flags=re.DOTALL,
    )
    if not m:
        return None
    return m.group(0).rstrip() + "\n"


def build_section(
    data: Dict[str, Any],
    previous_block: Optional[str] = None,
) -> str:
    entries: List[Dict[str, Any]] = data.get("entries", [])
    lines = [
        START,
        "",
        "### PushT rollout sweep (17 visual planning conditions, 4 categories)",
        "",
        "Planning: `plan_pusht_local`, `n_evals=50`, `goal_H=5`, `planner.max_iter=5`. "
        "**Mean** = average SR over conditions in that category.",
        "",
        "| Category | Conditions | Count |",
        "|----------|------------|-------|",
        "| Clean / lighting-color | NC, SC, C, LC, LCR, LCG, LCB | 7 |",
        "| Texture shift | T1–T5 (checker, h-stripes, v-stripes, noise, crosshatch) | 5 |",
        "| Viewpoint shift | V1 zoom-in, V2 zoom-out, V3 rotation, V4 crop | 4 |",
        "| Dynamic nuisance | D1 (moving distractors) | 1 |",
        "",
        "### Legacy six / Anderson current",
        "",
        "Auto-updated from `rollout_sweep_results.json`. "
        "Hand-maintained **Previous checkpoints** rows are preserved across updates.",
        "",
    ]
    if previous_block:
        lines.append(previous_block.rstrip())
        lines.append("")
    lines.extend(
        build_category_table(
            entries,
            "1. Clean / lighting-color shift",
            COLOR_CONDITIONS,
        )
    )
    lines.extend(
        build_category_table(
            entries,
            "2. Texture shift",
            TEXTURE_CONDITIONS,
        )
    )
    lines.extend(
        build_category_table(
            entries,
            "3. Viewpoint shift",
            VIEWPOINT_CONDITIONS,
        )
    )
    lines.extend(
        build_category_table(
            entries,
            "4. Dynamic nuisance",
            DYNAMIC_CONDITIONS,
        )
    )
    lines.extend(["", END, ""])
    return "\n".join(lines)


def patch_readme(readme_text: str, section: str) -> str:
    if START in readme_text and END in readme_text:
        return re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            section.rstrip(),
            readme_text,
            count=1,
            flags=re.DOTALL,
        )
    anchor = "### PushT (`pusht_noise` dataset)"
    idx = readme_text.find(anchor)
    if idx == -1:
        raise SystemExit("Could not find PushT section in README.md")
    insert_at = readme_text.find("\n", idx)
    for _ in range(4):
        insert_at = readme_text.find("\n", insert_at + 1)
    return readme_text[: insert_at + 1] + "\n" + section + readme_text[insert_at + 1 :]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-json",
        type=Path,
        default=REPO_ROOT / "plan_outputs" / "rollout_sweep_results.json",
    )
    p.add_argument("--readme", type=Path, default=README)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.results_json.is_file():
        raise SystemExit(f"Missing {args.results_json}")

    data = json.loads(args.results_json.read_text())
    readme_text = args.readme.read_text()
    previous = extract_previous_block(readme_text)
    section = build_section(data, previous_block=previous)
    new_text = patch_readme(readme_text, section)

    if args.dry_run:
        print(section)
        return

    args.readme.write_text(new_text)
    print(f"Updated {args.readme}")


if __name__ == "__main__":
    main()
