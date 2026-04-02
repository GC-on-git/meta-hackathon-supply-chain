# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hackathon Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional
from uuid import uuid4

from hackathon._compat import Environment
from hackathon.models import HackathonAction, HackathonObservation, HackathonState


ECHELON_NAMES = ["manufacturer", "warehouse", "retailer"]
HOLDING_COSTS = [0.015, 0.020, 0.030]
TRANSPORT_COSTS = [0.010, 0.014, 0.018]
FIXED_ORDER_COSTS = [0.05, 0.08, 0.10]

DIFFICULTY_PRESETS = {
    "easy": {
        "active_echelons": [False, False, True],
        "lead_times": [0, 0, 0],
        "base_demand": 18.0,
        "demand_std": 0.0,
        "season_shift_day": None,
        "season_high_demand": 18.0,
        "partial_visibility": False,
        "disruption_probability": 0.0,
        "reward_scale": 6.0,
        "max_order_qty": 40.0,
    },
    "medium": {
        "active_echelons": [False, True, True],
        "lead_times": [0, 3, 3],
        "base_demand": 20.0,
        "demand_std": 4.0,
        "season_shift_day": None,
        "season_high_demand": 20.0,
        "partial_visibility": False,
        "disruption_probability": 0.0,
        "reward_scale": 9.0,
        "max_order_qty": 55.0,
    },
    "mvp": {
        "active_echelons": [True, True, True],
        "lead_times": [4, 3, 2],
        "base_demand": 22.0,
        "demand_std": 4.5,
        "season_shift_day": None,
        "season_high_demand": 22.0,
        "partial_visibility": False,
        "disruption_probability": 0.0,
        "reward_scale": 11.0,
        "max_order_qty": 65.0,
    },
    "hard": {
        "active_echelons": [True, True, True],
        "lead_times": [5, 4, 2],
        "base_demand": 22.0,
        "demand_std": 5.0,
        "season_shift_day": 180,
        "season_high_demand": 32.0,
        "partial_visibility": True,
        "disruption_probability": 0.05,
        "reward_scale": 12.0,
        "max_order_qty": 75.0,
    },
}


@dataclass
class Shipment:
    quantity: float
    eta: int


class HackathonEnvironment(Environment):
    def __init__(self) -> None:
        self._rng = Random()
        self._difficulty = "mvp"
        self._config = DIFFICULTY_PRESETS[self._difficulty]
        self._horizon = 365
        self._max_order_qty = float(self._config["max_order_qty"])
        self._active_echelons = [True, True, True]
        self._lead_times = [4, 3, 2]
        self._inventory = [0.0, 0.0, 0.0]
        self._order_backlogs = [0.0, 0.0, 0.0]
        self._pipelines: list[list[Shipment]] = [[], [], []]
        self._days_since_last_order = [0, 0, 0]
        self._customer_backlog = 0.0
        self._recent_customer_demand: deque[float] = deque(maxlen=7)
        self._latest_forecast = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._last_disruption_link: Optional[str] = None
        self._state = HackathonState(
            episode_id=str(uuid4()),
            step_count=0,
            seed=0,
            difficulty=self._difficulty,
            cumulative_reward=0.0,
            total_cost=0.0,
            total_demand=0.0,
            total_served=0.0,
            fill_rate=0.0,
            customer_backlog=0.0,
            regime="baseline",
            last_disruption_link=None,
            termination_reason="not_started",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: str = "mvp",
        horizon: Optional[int] = None,
        max_order_qty: Optional[float] = None,
        **kwargs: Any,
    ) -> HackathonObservation:
        del kwargs
        if difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(f"unsupported difficulty: {difficulty}")

        if seed is not None:
            self._rng.seed(seed)
        current_seed = int(seed if seed is not None else 0)

        self._difficulty = difficulty
        self._config = DIFFICULTY_PRESETS[difficulty]
        self._horizon = int(horizon or 365)
        self._max_order_qty = float(max_order_qty or self._config["max_order_qty"])
        self._active_echelons = list(self._config["active_echelons"])
        self._lead_times = list(self._config["lead_times"])

        base_demand = float(self._config["base_demand"])
        safety_days = [7.0, 5.0, 3.0]
        self._inventory = [
            round(base_demand * safety_days[index], 2) if active else 0.0
            for index, active in enumerate(self._active_echelons)
        ]
        self._order_backlogs = [0.0, 0.0, 0.0]
        self._pipelines = [[], [], []]
        self._days_since_last_order = [0, 0, 0]
        self._customer_backlog = 0.0
        self._recent_customer_demand = deque([base_demand] * 7, maxlen=7)
        self._latest_forecast = base_demand
        self._last_reward_terms = {
            "holding_cost": 0.0,
            "stockout_penalty": 0.0,
            "transport_cost": 0.0,
            "fill_rate_bonus": 0.0,
            "total": 0.0,
        }
        self._last_disruption_link = None
        self._state = HackathonState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            seed=current_seed,
            difficulty=difficulty,
            cumulative_reward=0.0,
            total_cost=0.0,
            total_demand=0.0,
            total_served=0.0,
            fill_rate=0.0,
            customer_backlog=0.0,
            regime=self._current_regime(day=0),
            last_disruption_link=None,
            termination_reason="in_progress",
        )
        return self._build_observation(done=False, reward=0.0, metadata={"status": "ready"})

    def step(self, action: HackathonAction, timeout_s: Optional[float] = None, **kwargs: Any) -> HackathonObservation:
        del timeout_s, kwargs
        if self._state.termination_reason == "horizon_reached":
            return self._build_observation(
                done=True,
                reward=0.0,
                metadata={"message": "episode already complete", "termination_reason": self._state.termination_reason},
            )

        self._state.step_count += 1
        self._receive_inbound_shipments()
        self._serve_existing_backlog()

        sanitized_action = self._sanitize_action(action.order_quantities)
        self._register_orders(sanitized_action)

        self._last_disruption_link = self._sample_disruption_link()
        shipped_quantities = self._dispatch_replenishment_orders()

        current_demand = self._sample_customer_demand(day=self._state.step_count)
        served_demand = self._serve_customer_demand(current_demand)
        self._latest_forecast = self._update_forecast(current_demand)
        self._recent_customer_demand.append(current_demand)

        reward_terms = self._compute_reward_terms(
            current_demand=current_demand,
            served_demand=served_demand,
            shipped_quantities=shipped_quantities,
        )
        done = self._state.step_count >= self._horizon
        self._state.cumulative_reward += reward_terms["total"]
        self._state.total_cost += (
            reward_terms["holding_cost"] + reward_terms["stockout_penalty"] + reward_terms["transport_cost"]
        )
        self._state.total_demand += current_demand
        self._state.total_served += served_demand
        self._state.fill_rate = self._safe_ratio(self._state.total_served, self._state.total_demand)
        self._state.customer_backlog = self._customer_backlog
        self._state.regime = self._current_regime(day=self._state.step_count)
        self._state.last_disruption_link = self._last_disruption_link
        self._state.termination_reason = "horizon_reached" if done else "in_progress"
        self._last_reward_terms = reward_terms

        return self._build_observation(
            done=done,
            reward=reward_terms["total"],
            metadata={
                "current_demand": round(current_demand, 4),
                "served_demand": round(served_demand, 4),
                "disruption_link": self._last_disruption_link,
            },
        )

    @property
    def state(self) -> HackathonState:
        return self._state

    def _sanitize_action(self, order_quantities: list[float]) -> list[float]:
        sanitized: list[float] = []
        for index, quantity in enumerate(order_quantities):
            if not self._active_echelons[index]:
                sanitized.append(0.0)
                self._days_since_last_order[index] = 0
                continue
            clipped = min(max(float(quantity), 0.0), self._max_order_qty)
            sanitized.append(clipped)
            if clipped > 0:
                self._days_since_last_order[index] = 0
            else:
                self._days_since_last_order[index] += 1
        return sanitized

    def _register_orders(self, order_quantities: list[float]) -> None:
        for index, quantity in enumerate(order_quantities):
            self._order_backlogs[index] += quantity

    def _receive_inbound_shipments(self) -> None:
        for node_index, queue in enumerate(self._pipelines):
            if not queue:
                continue
            remaining_shipments: list[Shipment] = []
            for shipment in queue:
                updated_eta = shipment.eta - 1
                if updated_eta <= 0:
                    self._inventory[node_index] += shipment.quantity
                else:
                    remaining_shipments.append(Shipment(quantity=shipment.quantity, eta=updated_eta))
            self._pipelines[node_index] = remaining_shipments

    def _serve_existing_backlog(self) -> None:
        retailer_index = 2
        backlog_served = min(self._inventory[retailer_index], self._customer_backlog)
        self._inventory[retailer_index] -= backlog_served
        self._customer_backlog -= backlog_served

    def _dispatch_replenishment_orders(self) -> list[float]:
        shipped_quantities = [0.0, 0.0, 0.0]
        for node_index in range(3):
            if not self._active_echelons[node_index]:
                continue

            requested = self._order_backlogs[node_index]
            if requested <= 0:
                continue

            source_index = self._upstream_source(node_index)
            available = requested if source_index is None else min(requested, self._inventory[source_index])
            if self._last_disruption_link == ECHELON_NAMES[node_index]:
                available *= 0.25

            shipped = round(max(available, 0.0), 4)
            if shipped <= 0:
                continue

            if source_index is not None:
                self._inventory[source_index] -= shipped

            self._order_backlogs[node_index] = max(self._order_backlogs[node_index] - shipped, 0.0)
            shipped_quantities[node_index] = shipped

            lead_time = self._lead_times[node_index]
            if lead_time <= 0:
                self._inventory[node_index] += shipped
            else:
                self._pipelines[node_index].append(Shipment(quantity=shipped, eta=lead_time))
        return shipped_quantities

    def _upstream_source(self, node_index: int) -> Optional[int]:
        if node_index == 0:
            return None
        upstream = node_index - 1
        return upstream if self._active_echelons[upstream] else None

    def _sample_customer_demand(self, day: int) -> float:
        mean = float(self._config["base_demand"])
        shift_day = self._config["season_shift_day"]
        if shift_day is not None and day >= shift_day:
            mean = float(self._config["season_high_demand"])

        weekly_wave = 0.0
        if self._difficulty == "hard":
            weekly_wave = 2.5 if day % 7 in {4, 5} else -1.5 if day % 7 == 0 else 0.0

        std = float(self._config["demand_std"])
        sampled = self._rng.gauss(mean + weekly_wave, std) if std > 0 else mean
        return round(max(sampled, 0.0), 4)

    def _serve_customer_demand(self, current_demand: float) -> float:
        retailer_index = 2
        served = min(self._inventory[retailer_index], current_demand)
        self._inventory[retailer_index] -= served
        unmet = current_demand - served
        self._customer_backlog += unmet
        return round(served, 4)

    def _update_forecast(self, observed_demand: float) -> float:
        recent_mean = sum(self._recent_customer_demand) / max(len(self._recent_customer_demand), 1)
        alpha = 0.55 if self._difficulty in {"easy", "medium"} else 0.35
        baseline = alpha * observed_demand + (1 - alpha) * recent_mean
        if self._config["season_shift_day"] and self._state.step_count >= int(self._config["season_shift_day"]):
            baseline += 2.0
        return round(max(baseline, 0.0), 4)

    def _sample_disruption_link(self) -> Optional[str]:
        if self._config["disruption_probability"] <= 0:
            return None
        if self._rng.random() >= float(self._config["disruption_probability"]):
            return None
        active_links = [ECHELON_NAMES[index] for index, active in enumerate(self._active_echelons) if active]
        return self._rng.choice(active_links)

    def _compute_reward_terms(
        self,
        current_demand: float,
        served_demand: float,
        shipped_quantities: list[float],
    ) -> dict[str, float]:
        holding_cost = 0.0
        target_inventory = [120.0, 80.0, 45.0]
        for index, inventory in enumerate(self._inventory):
            if not self._active_echelons[index]:
                continue
            excess = max(inventory - target_inventory[index], 0.0)
            holding_cost += HOLDING_COSTS[index] * inventory
            holding_cost += HOLDING_COSTS[index] * (excess ** 2) / max(target_inventory[index], 1.0)

        stockout_penalty = 0.11 * self._customer_backlog
        stockout_penalty += 0.02 * sum(self._order_backlogs[index] for index in range(3) if self._active_echelons[index])

        transport_cost = 0.0
        for index, shipped in enumerate(shipped_quantities):
            if shipped <= 0:
                continue
            transport_cost += TRANSPORT_COSTS[index] * shipped + FIXED_ORDER_COSTS[index]

        fill_rate_bonus = 0.65 * self._safe_ratio(served_demand, current_demand if current_demand > 0 else 1.0)
        total_cost = holding_cost + stockout_penalty + transport_cost
        raw_reward = fill_rate_bonus - (total_cost / float(self._config["reward_scale"]))
        reward = max(-1.0, min(1.0, raw_reward))
        return {
            "holding_cost": round(holding_cost, 4),
            "stockout_penalty": round(stockout_penalty, 4),
            "transport_cost": round(transport_cost, 4),
            "fill_rate_bonus": round(fill_rate_bonus, 4),
            "total": round(reward, 4),
        }

    def _build_observation(self, done: bool, reward: float, metadata: Optional[dict[str, Any]] = None) -> HackathonObservation:
        inventory_levels = self._visible_inventory_levels()
        in_transit_qty = [round(sum(shipment.quantity for shipment in queue), 4) for queue in self._pipelines]
        lead_time_remaining = [float(min((shipment.eta for shipment in queue), default=0)) for queue in self._pipelines]
        forecasts = self._forecast_per_echelon()
        visibility_mask = [not (self._config["partial_visibility"] and index == 0) for index in range(3)]

        state_vector: list[float] = []
        for index in range(3):
            state_vector.extend(
                [
                    inventory_levels[index],
                    in_transit_qty[index],
                    forecasts[index],
                    float(self._days_since_last_order[index]),
                    HOLDING_COSTS[index],
                    lead_time_remaining[index],
                ]
            )

        return HackathonObservation(
            day=self._state.step_count,
            horizon=self._horizon,
            difficulty=self._difficulty,
            echelon_names=list(ECHELON_NAMES),
            active_echelons=list(self._active_echelons),
            visibility_mask=visibility_mask,
            state_vector=[round(value, 4) for value in state_vector],
            inventory_levels=[round(value, 4) for value in inventory_levels],
            in_transit_qty=in_transit_qty,
            demand_forecast=forecasts,
            days_since_last_order=[float(value) for value in self._days_since_last_order],
            holding_cost_rate=list(HOLDING_COSTS),
            lead_time_remaining=lead_time_remaining,
            order_backlogs=[round(value, 4) for value in self._order_backlogs],
            customer_backlog=round(self._customer_backlog, 4),
            recent_customer_demand=[round(value, 4) for value in self._recent_customer_demand],
            fill_rate=round(self._state.fill_rate, 4),
            disruption_link=self._last_disruption_link,
            regime=self._state.regime,
            reward_terms=self._last_reward_terms,
            action_bounds=[self._max_order_qty if active else 0.0 for active in self._active_echelons],
            done=done,
            reward=reward,
            metadata=metadata or {},
        )
    def _visible_inventory_levels(self) -> list[float]:
        visible = [round(value, 4) for value in self._inventory]
        if self._config["partial_visibility"]:
            visible[0] = -1.0
        return visible

    def _forecast_per_echelon(self) -> list[float]:
        base_forecast = self._latest_forecast
        multipliers = [1.20, 1.10, 1.00]
        forecasts = []
        for index in range(3):
            if not self._active_echelons[index]:
                forecasts.append(0.0)
            else:
                forecasts.append(round(base_forecast * multipliers[index], 4))
        return forecasts

    def _current_regime(self, day: int) -> str:
        shift_day = self._config["season_shift_day"]
        if shift_day is not None and day >= int(shift_day):
            return "peak_season"
        return "baseline"

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return 0.0 if denominator <= 0 else numerator / denominator


# Backwards-compatible alias (older class name from the original env).
SupplyChainInventoryEnvironment = HackathonEnvironment
