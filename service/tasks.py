"""Enumerated submission tasks: ids, reset kwargs, objectives, grader references.

``objective`` strings are consumed by ``inference.py`` (``build_system_prompt``)
to inject per-task grading criteria into the LLM system message.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from service.grading import FILL_TARGETS, WEIGHTS

TASK_IDS: Final[tuple[str, ...]] = ("easy", "medium", "hard")

GRADER_MODULE = "service.grading"
GRADER_FUNCTION = "grade_episode"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    seed: int
    horizon: int
    objective: str


def _default_horizon() -> int:
    return int(os.getenv("TASK_HORIZON", "30"))


def task_specs() -> tuple[TaskSpec, ...]:
    h = _default_horizon()
    return (
        TaskSpec(
            task_id="easy",
            difficulty="easy",
            seed=int(os.getenv("TASK_SEED_EASY", "1001")),
            horizon=h,
            objective=(
                "Maximize weighted fill-rate and cost efficiency; "
                f"grader fill target {FILL_TARGETS['easy']:.0%} "
                f"(weights fill/cost = {WEIGHTS['easy']['fill']}/{WEIGHTS['easy']['cost']})."
            ),
        ),
        TaskSpec(
            task_id="medium",
            difficulty="medium",
            seed=int(os.getenv("TASK_SEED_MEDIUM", "2002")),
            horizon=h,
            objective=(
                "Same as easy with higher demand volatility; "
                f"fill target {FILL_TARGETS['medium']:.0%}, "
                f"weights fill/cost = {WEIGHTS['medium']['fill']}/{WEIGHTS['medium']['cost']}."
            ),
        ),
        TaskSpec(
            task_id="hard",
            difficulty="hard",
            seed=int(os.getenv("TASK_SEED_HARD", "3003")),
            horizon=h,
            objective=(
                "Partial visibility, disruptions, seasonality; "
                f"fill target {FILL_TARGETS['hard']:.0%}, "
                f"weights fill/cost/co2 = "
                f"{WEIGHTS['hard']['fill']}/{WEIGHTS['hard']['cost']}/{WEIGHTS['hard']['co2']}."
            ),
        ),
    )


def task_by_id(task_id: str) -> TaskSpec:
    tid = task_id.lower()
    for spec in task_specs():
        if spec.task_id == tid:
            return spec
    raise ValueError(f"unknown task_id: {task_id!r}; expected one of {TASK_IDS}")
