"""Semantics-aligned episode grader for the supply chain environment.

Decomposes the final score into:
  S_fill  – service subscore from fill_rate (higher = better)
  S_cost  – cost efficiency subscore from total_cost (lower cost = higher score)
  S_co2   – carbon subscore from carbon_footprint (hard only; lower = higher score)

Reference constants (C_REF, K_REF) were calibrated from 30-step rollouts
with a range of ordering strategies (zero, moderate, aggressive) so that
a "reasonable" policy lands in the 0.3–0.7 range on each component.
"""

from __future__ import annotations

import math
from typing import Protocol


class _GradableState(Protocol):
    fill_rate: float
    total_cost: float
    carbon_footprint: float


# --- Per-task reference constants for cost normalisation ---
# Chosen so that S_cost ≈ 0.5 at a "typical moderate" episode cost.
C_REF = {"easy": 8_000.0, "medium": 10_000.0, "hard": 15_000.0}

# Carbon reference (hard only): moderate carbon ≈ 15k kgCO2 → S_co2 ≈ 0.37
K_REF = 15_000.0

# --- Fill-rate targets per task (from README §11) ---
FILL_TARGETS = {"easy": 0.70, "medium": 0.80, "hard": 0.85}

# --- Component weights (sum to 1.0 per task) ---
WEIGHTS = {
    "easy":   {"fill": 0.70, "cost": 0.30, "co2": 0.00},
    "medium": {"fill": 0.65, "cost": 0.35, "co2": 0.00},
    "hard":   {"fill": 0.55, "cost": 0.25, "co2": 0.20},
}


def _fill_subscore(fill_rate: float, target: float) -> float:
    """Piecewise-linear: softer credit below target, bonus ramp above."""
    fr = max(0.0, min(1.0, fill_rate))
    if fr >= target:
        return target + (fr - target) * (1.0 - target) / (1.0 - target + 1e-9)
    return fr * (target / max(target, 1e-9))


def _cost_subscore(total_cost: float, c_ref: float) -> float:
    """Monotone decreasing in cost, always in (0, 1]."""
    return 1.0 / (1.0 + max(total_cost, 0.0) / c_ref)


def _co2_subscore(carbon: float, k_ref: float) -> float:
    """Exponential decay – always in (0, 1]."""
    return math.exp(-max(carbon, 0.0) / k_ref)


def grade_episode(state: _GradableState, task_name: str) -> float:
    """Return a deterministic score in [0.0, 1.0] for a completed episode."""
    task = task_name.lower()
    if task not in WEIGHTS:
        raise ValueError(f"unknown task: {task_name!r}")

    w = WEIGHTS[task]
    s_fill = _fill_subscore(state.fill_rate, FILL_TARGETS[task])
    s_cost = _cost_subscore(state.total_cost, C_REF[task])
    s_co2 = _co2_subscore(state.carbon_footprint, K_REF) if w["co2"] > 0 else 0.0

    raw = w["fill"] * s_fill + w["cost"] * s_cost + w["co2"] * s_co2
    return float(max(0.0, min(1.0, raw)))
