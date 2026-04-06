"""Tests for the semantics-aligned episode grader (service.grading)."""

from __future__ import annotations

import pytest

from service.grading import grade_episode
from service.models import SupplyChainState

TASKS = ["easy", "medium", "hard"]


def _state(
    fill_rate: float = 0.5,
    total_cost: float = 5000.0,
    carbon_footprint: float = 3000.0,
) -> SupplyChainState:
    return SupplyChainState(
        fill_rate=fill_rate,
        total_cost=total_cost,
        carbon_footprint=carbon_footprint,
    )


# ---- Bounds ----------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize(
    "fr,cost,co2",
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 100_000.0, 100_000.0),
        (1.0, 100_000.0, 100_000.0),
        (0.5, 8_000.0, 5_000.0),
    ],
)
def test_score_bounded_0_1(task: str, fr: float, cost: float, co2: float):
    score = grade_episode(_state(fr, cost, co2), task)
    assert 0.0 <= score <= 1.0, f"score {score} out of [0,1] for {task}"


# ---- Monotonicity in fill_rate ---------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_higher_fill_rate_not_lower_score(task: str):
    low = grade_episode(_state(fill_rate=0.3), task)
    mid = grade_episode(_state(fill_rate=0.6), task)
    high = grade_episode(_state(fill_rate=0.9), task)
    assert low <= mid <= high, f"fill-rate monotonicity violated for {task}"


# ---- Cost sensitivity (all tasks) ------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_higher_cost_lowers_score(task: str):
    cheap = grade_episode(_state(total_cost=1_000.0), task)
    expensive = grade_episode(_state(total_cost=50_000.0), task)
    assert cheap > expensive, (
        f"cost sensitivity violated for {task}: cheap={cheap}, expensive={expensive}"
    )


# ---- Carbon sensitivity (hard only) ----------------------------------------

def test_higher_carbon_lowers_hard_score():
    low_co2 = grade_episode(_state(carbon_footprint=1_000.0), "hard")
    high_co2 = grade_episode(_state(carbon_footprint=80_000.0), "hard")
    assert low_co2 > high_co2


# ---- Determinism ------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_deterministic(task: str):
    st = _state(fill_rate=0.75, total_cost=12_000.0, carbon_footprint=8_000.0)
    scores = [grade_episode(st, task) for _ in range(5)]
    assert len(set(scores)) == 1, f"non-deterministic scores for {task}: {scores}"


# ---- Unknown task raises ----------------------------------------------------

def test_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown task"):
        grade_episode(_state(), "impossible")
