# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic models for the supply chain environment and agent interface."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field, field_validator


DifficultyName = Literal["easy", "medium", "mvp", "hard"]


class AgentAction(Action):
    order_quantities: List[float] = Field(default_factory=lambda: [0.0] * 21, min_length=21, max_length=21)
    shipping_methods: List[int] = Field(default_factory=lambda: [0] * 21, min_length=21, max_length=21)

    @field_validator("order_quantities")
    @classmethod
    def validate_quantities(cls, value: List[float]) -> List[float]:
        if any(quantity < 0 for quantity in value):
            raise ValueError("order quantities must be non-negative")
        return value


class AgentObservation(Observation):
    day: int = 0
    horizon: int = 365
    difficulty: DifficultyName = "mvp"
    echelon_names: List[str] = Field(default_factory=lambda: ["manufacturer", "warehouse", "retailer"])
    active_echelons: List[bool] = Field(default_factory=list)
    visibility_mask: List[bool] = Field(default_factory=list)
    state_vector: List[float] = Field(default_factory=list)
    inventory_levels: List[float] = Field(default_factory=list, min_length=21, max_length=21)
    in_transit_qty: List[float] = Field(default_factory=list, min_length=21, max_length=21)
    demand_forecast: List[float] = Field(default_factory=list, min_length=21, max_length=21)
    days_since_last_order: List[float] = Field(default_factory=list, min_length=21, max_length=21)
    holding_cost_rate: List[float] = Field(default_factory=list, min_length=21, max_length=21)
    lead_time_remaining: List[float] = Field(default_factory=list, min_length=21, max_length=21)
    order_backlogs: List[float] = Field(default_factory=list, min_length=21, max_length=21)
    customer_backlog: List[float] = Field(default_factory=list, min_length=12, max_length=12)
    recent_customer_demand: List[float] = Field(default_factory=list)
    carbon_footprint: float = 0.0
    fill_rate: float = 0.0
    disruption_link: Optional[str] = None
    regime: str = "baseline"
    reward_terms: Dict[str, float] = Field(default_factory=dict)
    action_bounds: List[float] = Field(default_factory=lambda: [0.0] * 21, min_length=21, max_length=21)
    active_events: List[str] = Field(default_factory=list)


class SupplyChainState(State):
    seed: int = 0
    difficulty: DifficultyName = "mvp"
    cumulative_reward: float = 0.0
    total_cost: float = 0.0
    total_demand: float = 0.0
    total_served: float = 0.0
    fill_rate: float = 0.0
    customer_backlog: List[float] = Field(default_factory=list)
    carbon_footprint: float = 0.0
    regime: str = "baseline"
    last_disruption_link: Optional[str] = None
    termination_reason: str = "not_started"
    active_events: List[str] = Field(default_factory=list)


__all__ = [
    "AgentAction",
    "AgentObservation",
    "DifficultyName",
    "SupplyChainState",
]
