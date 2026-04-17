"""Test-time visual conditions (paper Table: NC, SC, C, LC, LCG, D) for non–PointMaze envs."""

from typing import FrozenSet

VISUAL_COLUMNS: tuple[str, ...] = ("NC", "SC", "C", "LC", "LCG", "D")

_VALID: FrozenSet[str] = frozenset(VISUAL_COLUMNS)


def normalize_visual_condition(name: str) -> str:
    u = name.strip().upper()
    if u not in _VALID:
        raise ValueError(
            f"Unknown visual condition {name!r}; expected one of {sorted(_VALID)}"
        )
    return u
