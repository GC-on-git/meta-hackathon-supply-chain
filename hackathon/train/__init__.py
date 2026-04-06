# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv-native training helpers: map ``AgentObservation`` / ``AgentAction`` to NumPy vectors.

Training scripts use **in-process** ``SupplyChainEnv`` (``Environment`` from OpenEnv) — no Gym/Gymnasium.
Install optional ``[train]`` for ``torch`` and ``numpy``.
"""

from __future__ import annotations

import numpy as np

from hackathon.models import AgentAction, AgentObservation
from hackathon.server.hackathon_environment import SupplyChainEnv

# Matches current 7×3×6 feature layout in `_build_observation`.
STATE_VECTOR_DIM = 126
# 21 normalized order quantities + 21 shipping logits (>=0.5 => express).
ACTION_DIM = 42


def observation_to_vector(obs: AgentObservation) -> np.ndarray:
    """Fixed-length float32 vector from ``obs.state_vector`` (pad/truncate)."""
    v = np.asarray(obs.state_vector, dtype=np.float32).reshape(-1)
    out = np.zeros(STATE_VECTOR_DIM, dtype=np.float32)
    n = min(v.size, STATE_VECTOR_DIM)
    if n > 0:
        out[:n] = v[:n]
    return out


def vector_to_agent_action(vec: np.ndarray, max_order_qty: float) -> AgentAction:
    """Map ``vec`` in [0, 1]^42 to ``AgentAction`` (orders scaled by ``max_order_qty``)."""
    a = np.asarray(vec, dtype=np.float32).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"expected action dim {ACTION_DIM}, got {a.shape[0]}")
    max_q = float(max_order_qty)
    quantities = (np.clip(a[:21], 0.0, 1.0) * max_q).astype(np.float64).tolist()
    methods = (a[21:] >= 0.5).astype(np.int64).tolist()
    return AgentAction(order_quantities=quantities, shipping_methods=methods)


def new_supply_chain_env(
    difficulty: str = "easy",
    horizon: int = 120,
    seed: int | None = None,
) -> tuple[SupplyChainEnv, np.ndarray]:
    """Construct ``SupplyChainEnv``, ``reset``, return ``(env, initial_observation_vector)``."""
    env = SupplyChainEnv()
    obs = env.reset(difficulty=difficulty, seed=seed, horizon=horizon)
    return env, observation_to_vector(obs)


__all__ = [
    "ACTION_DIM",
    "STATE_VECTOR_DIM",
    "new_supply_chain_env",
    "observation_to_vector",
    "vector_to_agent_action",
]
