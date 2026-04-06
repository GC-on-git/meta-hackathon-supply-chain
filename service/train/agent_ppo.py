# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
**PPO** (Proximal Policy Optimization) in PyTorch on **OpenEnv** ``SupplyChainEnv``.

Collects rollouts via ``reset`` / ``step(AgentAction)`` — no Gymnasium / Stable-Baselines3.

Example::

    PYTHONPATH=. python -m service.train.agent_ppo --difficulty easy
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np

from service.train import (
    ACTION_DIM,
    STATE_VECTOR_DIM,
    observation_to_vector,
    vector_to_agent_action,
)
from service.server.hackathon_environment import SupplyChainEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO (PyTorch) on SupplyChainEnv (OpenEnv)")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-len", type=int, default=512, help="steps collected per PPO update")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models/ppo_supply_chain.pt")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.distributions import Beta
    except ImportError as e:
        raise SystemExit(
            'PyTorch is required. Install with: pip install -e ".[train]" or pip install torch numpy'
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def layer_init(m: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        nn.init.orthogonal_(m.weight, std)
        nn.init.constant_(m.bias, bias_const)
        return m

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = 256
            self.trunk = nn.Sequential(
                layer_init(nn.Linear(STATE_VECTOR_DIM, h)),
                nn.Tanh(),
                layer_init(nn.Linear(h, h)),
                nn.Tanh(),
            )
            self.alpha_head = layer_init(nn.Linear(h, ACTION_DIM), std=0.01)
            self.beta_head = layer_init(nn.Linear(h, ACTION_DIM), std=0.01)
            self.value_head = layer_init(nn.Linear(h, 1), std=1.0)

        def forward(self, x: torch.Tensor) -> Tuple[Beta, torch.Tensor]:
            h = self.trunk(x)
            alpha = F.softplus(self.alpha_head(h)) + 1.0
            beta = F.softplus(self.beta_head(h)) + 1.0
            dist = Beta(alpha, beta)
            v = self.value_head(h).squeeze(-1)
            return dist, v

        def evaluate(self, x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            dist, v = self.forward(x)
            logp = dist.log_prob(a).sum(-1)
            ent = dist.entropy().sum(-1)
            return logp, v, ent

        def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            dist, v = self.forward(x)
            a = dist.sample()
            logp = dist.log_prob(a).sum(-1)
            return a, logp, v

    def collect_rollout(
        env: SupplyChainEnv,
        ac: ActorCritic,
        rollout_len: int,
        device: torch.device,
    ) -> tuple:
        obs_l, act_l, rew_l, val_l, logp_l, done_l = [], [], [], [], [], []
        obs_p = env.reset(difficulty=args.difficulty, seed=None, horizon=args.horizon)
        s = observation_to_vector(obs_p)
        for _ in range(rollout_len):
            st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_t, logp_t, v_t = ac.act(st)
            a_np = a_t.cpu().numpy().reshape(-1)
            aa = vector_to_agent_action(a_np, env._max_order_qty)
            next_p = env.step(aa)
            r = float(next_p.reward) if next_p.reward is not None else 0.0
            d = float(next_p.done)
            obs_l.append(s)
            act_l.append(a_np)
            rew_l.append(r)
            val_l.append(float(v_t.item()))
            logp_l.append(float(logp_t.item()))
            done_l.append(d)
            s = observation_to_vector(next_p)
            if next_p.done:
                obs_p = env.reset(difficulty=args.difficulty, seed=None, horizon=args.horizon)
                s = observation_to_vector(obs_p)
        with torch.no_grad():
            st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_v = ac.forward(st)
            last_v = float(last_v.item())
        return (
            np.stack(obs_l),
            np.stack(act_l),
            np.array(rew_l, dtype=np.float32),
            np.array(val_l, dtype=np.float32),
            np.array(logp_l, dtype=np.float32),
            np.array(done_l, dtype=np.float32),
            last_v,
        )

    def gae_returns(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_v: float,
        gamma: float,
        lam: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        adv = np.zeros_like(rewards)
        last_gae = 0.0
        next_v = last_v
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_v * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
            next_v = values[t]
        ret = adv + values
        return adv, ret

    env = SupplyChainEnv()
    ac = ActorCritic().to(device)
    opt = optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)

    global_step = 0
    print(
        f"DEBUG PPO: total_steps={args.total_steps} rollout_len={args.rollout_len} device={device} "
        f"difficulty={args.difficulty!r} (OpenEnv in-process)"
    )

    while global_step < args.total_steps:
        remain = args.total_steps - global_step
        if remain <= 0:
            break
        roll = min(args.rollout_len, remain)
        obs, actions, rewards, values, old_logp, dones, last_v = collect_rollout(env, ac, roll, device)
        global_step += len(rewards)
        adv, rets = gae_returns(rewards, values, dones, last_v, args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(rets, dtype=torch.float32, device=device)

        n = obs_t.shape[0]
        idx = np.arange(n)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, args.minibatch):
                mb = idx[start : start + args.minibatch]
                logp, v, ent = ac.evaluate(obs_t[mb], act_t[mb])
                ratio = torch.exp(logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_t[mb]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(v, ret_t[mb])
                loss = pol_loss + args.value_coef * val_loss - args.entropy_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad_norm)
                opt.step()

        print(
            f"DEBUG PPO: step {global_step:6d}  mean_reward={float(rewards.mean()):.4f}  "
            f"mean_ret={float(rets.mean()):.4f}"
        )

    import os

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({"actor_critic": ac.state_dict(), "config": vars(args)}, args.save_path)
    print(f"DEBUG PPO: saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
