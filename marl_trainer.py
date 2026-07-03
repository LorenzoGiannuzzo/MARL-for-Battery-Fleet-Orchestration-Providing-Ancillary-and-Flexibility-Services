"""
MAPPO trainer for the multi-BESS BSP aggregator on Ray RLlib.

Step 6 of the multi-agent track. Trains a SHARED-POLICY PPO over the
MultiBESSEnv (Step 5) using Ray RLlib 2.x. Parameter sharing means a
SINGLE policy network is trained on experiences pooled from all N
agents; at inference time each agent applies the same policy to its
own local observation. This is the dominant approach in cooperative
MARL when agents are exchangeable (Yu et al. 2022, "The Surprising
Effectiveness of PPO in Cooperative MARL").

Methodological note on "MAPPO" vs "IPPO + shared policy":

Strictly, MAPPO has TWO ingredients:
  (1) parameter-shared policy across agents
  (2) centralised critic that sees the GLOBAL state (concat of all
      agents' observations) at training time

This module implements (1) but not (2): the critic is decentralised
(per-agent value function evaluated on the local observation). RLlib's
default PPO uses a decentralised critic; obtaining a true centralised
critic requires a custom RLModule that overrides _forward_train. For
the BSP-aggregated cooperative setup, the empirical evidence (Yu et al.
2022, de Witt et al. 2020) is that parameter-shared PPO with decentralised
critic is competitive with full MAPPO when:
  - the reward is decomposable into per-agent contributions (ours is)
  - the agents are exchangeable in role (ours are: identical bidding
    interface; heterogeneity in sizing is conveyed via the observation)
  - the global state is not strictly needed to disambiguate actions

In the paper we report this design choice explicitly and treat true
centralised-critic MAPPO as a possible future extension. For most
practical BSP-aggregation problems, shared-policy IPPO is the right
baseline because it's simpler, faster, and competitive.

Run:

    from marl_trainer import build_trainer, train_loop
    cfg = build_default_config(n_batteries=5, episode_hours=24)
    algo = cfg.build_algo()
    results = train_loop(algo, n_iterations=200)
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ray RLlib imports
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Project imports
from milp_optimizer import BatteryParameters
from marl_milp_continuous import MultiBatteryParameters
from marl_env import MultiBESSEnv

warnings.simplefilter("ignore")


# ============================================================================
# Env factory and registration
# ============================================================================

ENV_NAME = "multi_bess_bsp"


def _env_creator(env_config: Dict[str, Any]):
    """Ray Tune-compatible env creator. Receives a config dict and returns
    a Ray-wrapped multi-agent env.

    Expected env_config keys:
      - fleet_kind: 'homogeneous' | 'heterogeneous'
      - n_batteries: int (only for homogeneous)
      - capacity_mwh: float (only for homogeneous)
      - max_power_mw: float (only for homogeneous)
      - heterogeneous_seed: int (only for heterogeneous)
      - episode_hours: int
      - use_nonlinear_degradation: bool
      - fce_cumulative_initial: list of float or None
      - seed: int
    """
    fleet_kind = env_config.get("fleet_kind", "homogeneous")
    episode_hours = env_config.get("episode_hours", 24)
    use_nl = env_config.get("use_nonlinear_degradation", False)
    seed = env_config.get("seed", None)
    fce_init = env_config.get("fce_cumulative_initial", None)
    directional = env_config.get("directional_services", False)

    if fleet_kind == "homogeneous":
        n = env_config.get("n_batteries", 5)
        cap = env_config.get("capacity_mwh", 4.0)
        pmw = env_config.get("max_power_mw", 2.0)
        bp = BatteryParameters(
            capacity_mwh=cap, max_power_mw=pmw,
            degradation_cost_per_mwh=25000.0, cycle_life=6000,
        )
        fleet = MultiBatteryParameters([bp] * n)
    elif fleet_kind == "heterogeneous":
        het_seed = env_config.get("heterogeneous_seed", 0)
        fleet = MultiBatteryParameters.heterogeneous_fleet(seed=het_seed)
    else:
        raise ValueError(f"unknown fleet_kind: {fleet_kind}")

    raw_env = MultiBESSEnv(
        multi_params=fleet,
        episode_hours=episode_hours,
        use_nonlinear_degradation=use_nl,
        fce_cumulative_initial=fce_init,
        seed=seed,
        directional_services=directional,
    )
    # Realistic commitment market model (Steps A-D). Off by default; switched
    # on by passing enable_commitments via env_config (build_default_config).
    if env_config.get("enable_commitments", False):
        raw_env.configure_commitments(
            enable=True,
            lead_time=env_config.get("commitment_lead_time", 24),
            penalty_k=env_config.get("penalty_k", 1.5),
        )
    # Diagnostic full-foresight: give the agent all 24h true prices/services.
    if env_config.get("full_foresight", False):
        raw_env.configure_full_foresight(True)
    if env_config.get("overcommit_penalty", 0.0):
        raw_env.configure_overcommit_penalty(env_config.get("overcommit_penalty", 0.0))
    return ParallelPettingZooEnv(raw_env)


def register_multi_bess_env():
    """Register the env with Ray Tune. Idempotent."""
    register_env(ENV_NAME, _env_creator)


# ============================================================================
# PPO configuration with shared policy and parameter sharing
# ============================================================================

SHARED_POLICY_ID = "shared_bess_policy"


def _shared_policy_mapping(agent_id: str, *args, **kwargs) -> str:
    """All agents map to the same policy id, enabling parameter sharing."""
    return SHARED_POLICY_ID


def build_default_config(
    n_batteries: int = 5,
    episode_hours: int = 24,
    fleet_kind: str = "homogeneous",
    use_nonlinear_degradation: bool = False,
    fce_cumulative_initial: Optional[List[float]] = None,
    seed: int = 0,
    lr: float = 3e-4,
    train_batch_size: int = 2000,
    minibatch_size: int = 256,
    num_epochs: int = 10,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    clip_param: float = 0.2,
    entropy_coeff: float = 0.001,
    # num_env_runners=0 => in-process rollouts, no separate Ray worker.
    # On Windows/8 GB the worker's shared-memory object store leaks each
    # iteration and the raylet dies (CreateFileMapping GetLastError 1450/1455).
    # In-process rollout is the only stable setting here.
    num_env_runners: int = 0,
    rollout_fragment_length: int = 200,
    fcnet_hiddens: Tuple[int, ...] = (512, 256),
    directional_services: bool = False,
    enable_commitments: bool = False,
    commitment_lead_time: int = 24,
    penalty_k: float = 1.5,
    full_foresight: bool = False,
    overcommit_penalty: float = 0.0,
) -> PPOConfig:
    """Build a PPOConfig for shared-policy multi-agent training.

    Default hyperparameters are the Yu et al. 2022 reference for
    cooperative MARL on continuous-control-like discretised actions,
    adapted to our episode length (24h = 24 timesteps).

    Notes on key hyperparameters:
      - lr=3e-4: standard PPO learning rate
      - train_batch_size=2000: ~83 episodes per training iteration at 24h
      - minibatch_size=256: 8 SGD steps per epoch
      - num_epochs=10: PPO recommended for sample efficiency
      - gamma=0.99: 1-day horizon is ~100 steps in discounted terms,
        gives a reasonable lookahead at 24h episodes
      - lambda_=0.95: GAE smoothing
      - clip_param=0.2: classical PPO clip
      - entropy_coeff=0.001: light entropy regularisation to keep some
        exploration in the discretised action space
      - fcnet_hiddens=(128,128): modest MLP, sufficient for 21-dim obs
    """
    register_multi_bess_env()

    env_config = dict(
        fleet_kind=fleet_kind,
        n_batteries=n_batteries,
        episode_hours=episode_hours,
        use_nonlinear_degradation=use_nonlinear_degradation,
        fce_cumulative_initial=fce_cumulative_initial,
        seed=seed,
        directional_services=directional_services,
        enable_commitments=enable_commitments,
        commitment_lead_time=commitment_lead_time,
        penalty_k=penalty_k,
        full_foresight=full_foresight,
        overcommit_penalty=overcommit_penalty,
    )

    config = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies={SHARED_POLICY_ID},
            policy_mapping_fn=_shared_policy_mapping,
            policies_to_train=[SHARED_POLICY_ID],
        )
        .training(
            lr=lr,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            gamma=gamma,
            lambda_=lambda_,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": list(fcnet_hiddens),
                "fcnet_activation": "tanh",
            }
        )
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=rollout_fragment_length,
        )
        .resources(num_gpus=0)
        .debugging(seed=seed)
    )
    return config


# ============================================================================
# Training loop
# ============================================================================

def train_loop(
    algo,
    n_iterations: int,
    log_every: int = 10,
    log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    target_reward: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run training for n_iterations and collect per-iteration metrics.

    Each iteration is one full PPO update: env_runners gather train_batch_size
    timesteps, then the policy is updated for num_epochs over the data.

    Returns
    -------
    list of dict
        One dict per training iteration with keys:
          - iteration
          - episode_reward_mean   (sum of per-agent rewards per episode)
          - episode_reward_min
          - episode_reward_max
          - episodes_total
          - timesteps_total
          - elapsed_seconds
    """
    metrics = []
    t0 = time.time()
    for i in range(n_iterations):
        result = algo.train()
        # Ray RLlib 2.x metrics nesting can vary by version; pull conservatively
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get(
            "episode_return_mean",
            result.get("episode_reward_mean",
                       result.get("episode_return_mean")),
        )
        min_rew = env_runners.get(
            "episode_return_min",
            result.get("episode_reward_min",
                       result.get("episode_return_min")),
        )
        max_rew = env_runners.get(
            "episode_return_max",
            result.get("episode_reward_max",
                       result.get("episode_return_max")),
        )
        ts = result.get("num_env_steps_sampled_lifetime",
                        result.get("timesteps_total", 0))
        eps = env_runners.get(
            "num_episodes",
            result.get("episodes_total", 0),
        )
        entry = {
            "iteration": i,
            "episode_reward_mean": mean_rew,
            "episode_reward_min": min_rew,
            "episode_reward_max": max_rew,
            "episodes_total": eps,
            "timesteps_total": ts,
            "elapsed_seconds": time.time() - t0,
        }
        metrics.append(entry)
        if log_callback is not None:
            log_callback(entry)
        if (i + 1) % log_every == 0 or i == 0:
            mr_s = f"{mean_rew:.2f}" if mean_rew is not None else "None"
            print(f"  iter {i+1:3d}: mean_rew={mr_s}  "
                  f"timesteps={ts}  elapsed={entry['elapsed_seconds']:.1f}s")
        if target_reward is not None and mean_rew is not None and mean_rew >= target_reward:
            print(f"  target reward {target_reward} reached at iter {i+1}; stopping")
            break
    return metrics


# ============================================================================
# Evaluation: deterministic rollout of trained policy
# ============================================================================

def evaluate_policy(
    algo,
    n_episodes: int = 20,
    env_config_override: Optional[Dict[str, Any]] = None,
    deterministic: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Evaluate the trained shared policy over n_episodes.

    Uses the new API stack: get the RLModule via `algo.get_module(policy_id)`
    and forward observations through `forward_inference`.

    Returns
    -------
    dict with keys:
      - mean_total_reward, std_total_reward
      - mean_arbitrage, mean_flex, mean_degradation
      - per_episode_total_rewards: list of floats
    """
    import torch
    register_multi_bess_env()
    base_env_config = dict(algo.config.env_config or {})
    if env_config_override:
        base_env_config = {**base_env_config, **env_config_override}
    wrapped = _env_creator(base_env_config)

    # Get the shared RLModule for inference (new API stack)
    module = algo.get_module(SHARED_POLICY_ID)

    rewards_per_ep = []
    arb_per_ep, flex_per_ep, deg_per_ep = [], [], []

    for ep in range(n_episodes):
        obs, _ = wrapped.reset()
        total_rew = 0.0
        total_arb = 0.0
        total_flex = 0.0
        total_deg = 0.0
        done_flag = False
        while not done_flag:
            actions = {}
            for agent_id, agent_obs in obs.items():
                obs_tensor = torch.tensor(
                    np.asarray(agent_obs, dtype=np.float32)
                ).unsqueeze(0)
                with torch.no_grad():
                    if deterministic:
                        out = module.forward_inference({"obs": obs_tensor})
                    else:
                        out = module.forward_exploration({"obs": obs_tensor})
                # The RLModule outputs an 'action_dist_inputs' or 'actions' key
                if "actions" in out:
                    a_np = out["actions"][0].cpu().numpy()
                else:
                    # Build a Categorical / MultiCategorical from the logits
                    logits = out["action_dist_inputs"][0].cpu().numpy()
                    # MultiDiscrete([11]*5): logits has shape (55,), split per axis
                    n_axes = 5
                    bins_per_axis = len(logits) // n_axes
                    a_list = []
                    for k in range(n_axes):
                        sl = logits[k * bins_per_axis:(k + 1) * bins_per_axis]
                        if deterministic:
                            a_list.append(int(np.argmax(sl)))
                        else:
                            p = np.exp(sl - sl.max())
                            p /= p.sum()
                            # Reproducible: draw from a passed-in Generator
                            # rather than the process-global np.random state.
                            _draw = (rng.choice if rng is not None
                                     else np.random.choice)
                            a_list.append(int(_draw(bins_per_axis, p=p)))
                    a_np = np.array(a_list, dtype=np.int64)
                actions[agent_id] = a_np
            obs, rewards, terms, truncs, infos = wrapped.step(actions)
            for a, r in rewards.items():
                total_rew += r
                if a in infos and isinstance(infos[a], dict):
                    total_arb += infos[a].get("arbitrage", 0.0)
                    total_flex += infos[a].get("flex_revenue", 0.0)
                    total_deg += infos[a].get("degradation", 0.0)
            done_flag = (
                all(terms.values()) if terms else False
            ) or (all(truncs.values()) if truncs else False) or not bool(obs)
        rewards_per_ep.append(total_rew)
        arb_per_ep.append(total_arb)
        flex_per_ep.append(total_flex)
        deg_per_ep.append(total_deg)

    arr = np.array(rewards_per_ep, dtype=float)
    return {
        "mean_total_reward": float(arr.mean()),
        "std_total_reward":  float(arr.std()),
        "mean_arbitrage":    float(np.mean(arb_per_ep)),
        "mean_flex":         float(np.mean(flex_per_ep)),
        "mean_degradation":  float(np.mean(deg_per_ep)),
        "per_episode_total_rewards": rewards_per_ep,
    }


# ============================================================================
# Smoke training: a tiny end-to-end run that verifies the pipeline
# ============================================================================

def smoke_train(n_batteries: int = 2, n_iterations: int = 3) -> Dict[str, Any]:
    """Tiny end-to-end training run for testing. Returns metrics + final eval."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,
                 logging_level="ERROR", log_to_driver=False)
    cfg = build_default_config(
        n_batteries=n_batteries, episode_hours=24,
        train_batch_size=480, minibatch_size=120, num_epochs=2,
        num_env_runners=0,  # local rollout for smoke test
        rollout_fragment_length=24,
    )
    algo = cfg.build_algo()
    metrics = train_loop(algo, n_iterations=n_iterations, log_every=1)
    eval_res = evaluate_policy(algo, n_episodes=3, deterministic=False)
    algo.stop()
    return {"training_metrics": metrics, "evaluation": eval_res}