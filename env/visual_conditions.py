"""Test-time visual conditions for PushT / wall sim eval (four categories)."""

from typing import Dict, FrozenSet, Optional, Tuple

# Category 1 — clean / lighting-color shift (7); also referred to as C1–C7
COLOR_CONDITIONS: Tuple[str, ...] = ("NC", "SC", "C", "LC", "LCR", "LCG", "LCB")
C1_C7_CONDITIONS: Tuple[str, ...] = COLOR_CONDITIONS
# C1–C7 + dynamic D (legacy alias D → D1)
C1_C7_D_CONDITIONS: Tuple[str, ...] = COLOR_CONDITIONS + ("D1",)
# Category 2 — texture shift (5)
TEXTURE_CONDITIONS: Tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")
# Category 3 — viewpoint shift (4)
VIEWPOINT_CONDITIONS: Tuple[str, ...] = ("V1", "V2", "V3", "V4")
# Mild subset used for extra eval (texture 1/2 + mild viewpoint 1/2)
MILD_TEXTURE_VIEWPOINT: Tuple[str, ...] = ("T1", "T2", "V1", "V2")
# Category 4 — dynamic nuisance (1)
DYNAMIC_CONDITIONS: Tuple[str, ...] = ("D1",)
# Legacy six-condition set (subset of color + old D code)
LEGACY_SIX_COLUMNS: Tuple[str, ...] = ("NC", "SC", "C", "LC", "LCG", "D")

# Category: legacy six + mild texture/viewpoint (used for Bisim 6+4 eval)
LEGACY_SIX_MILD_TV: Tuple[str, ...] = ("NC", "SC", "C", "LC", "LCG", "D1", "T1", "T2", "V1", "V2")

VISUAL_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "color": COLOR_CONDITIONS,
    "color_d": C1_C7_D_CONDITIONS,
    "legacy_six": LEGACY_SIX_COLUMNS,
    "texture": TEXTURE_CONDITIONS,
    "viewpoint": VIEWPOINT_CONDITIONS,
    "mild_tv": MILD_TEXTURE_VIEWPOINT,
    "legacy_six_mild_tv": LEGACY_SIX_MILD_TV,
    "dynamic": DYNAMIC_CONDITIONS,
}

VISUAL_COLUMNS: Tuple[str, ...] = (
    COLOR_CONDITIONS + TEXTURE_CONDITIONS + VIEWPOINT_CONDITIONS + DYNAMIC_CONDITIONS
)

_ALIASES = {"D": "D1"}
_VALID: FrozenSet[str] = frozenset(VISUAL_COLUMNS) | frozenset(_ALIASES.keys())


def normalize_visual_condition(name: str) -> str:
    u = name.strip().upper()
    u = _ALIASES.get(u, u)
    if u not in VISUAL_COLUMNS:
        raise ValueError(
            f"Unknown visual condition {name!r}; expected one of {sorted(VISUAL_COLUMNS)} "
            f"(legacy alias: D → D1)"
        )
    return u


def category_for_condition(cond: str) -> str:
    c = normalize_visual_condition(cond)
    for cat, conds in VISUAL_CATEGORIES.items():
        if c in conds:
            return cat
    raise KeyError(cond)


def category_mean(rates: dict, conditions: Tuple[str, ...]) -> Optional[float]:
    vals = [float(rates[c]) for c in conditions if rates.get(c) is not None]
    return sum(vals) / len(vals) if vals else None
