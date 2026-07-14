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


# ---------------------------------------------------------------------------
# REAL MARKET WINDOW PLUMBING (July 2026 fix)
# ---------------------------------------------------------------------------
# BUG THIS FIXES: the PPO training env was built WITHOUT `prices=`/`services=`,
# so MultiBESSEnv.reset() silently fell back to `_default_prices` (a single
# fixed 80 + 40*sin(2*pi*(t-6)/24) sinusoid, identical every episode) and
# `_default_services` (award_probability = 1.0 on FCR/aFRR/mFRR, every hour).
# The MILP oracle and the BC policy were built on the REAL market window and
# every policy was evaluated on the REAL test window, so the PPO was the only
# agent trained in a different (and far more generous) world: guaranteed awards
# and a price signal with zero variance. The pipeline log line
# "iters on train window" was simply false.
#
# FIX: hand the real MarketWindow to the training env and sample ONE REAL DAY
# from it at every reset. Because the window carries the real per-hour
# FlexibilityService definitions, the real award_probability, capacity_price and
# energy_price come along automatically -- no separate wiring needed.
#
# The window is kept in a module-level registry and referenced by KEY from
# env_config, rather than embedded in env_config directly: RLlib deep-copies
# env_config, and a 365-day window holds tens of thousands of
# FlexibilityService objects. This requires in-process rollouts
# (num_env_runners=0), which is already forced on this setup for Windows
# stability. With remote workers the registry would live only in the driver;
# `_env_creator` raises a clear error in that case rather than silently falling
# back to the synthetic market again.

_MARKET_WINDOW_REGISTRY: Dict[str, Any] = {}


def register_market_window(window: Any, key: str = "ppo_train_window") -> str:
    """Register a MarketWindow so `_env_creator` can look it up by key.

    Returns the key, to be passed to `build_default_config(market_window_key=)`.
    """
    _MARKET_WINDOW_REGISTRY[key] = window
    return key


class _WindowSamplingMultiBESSEnv(MultiBESSEnv):
    """MultiBESSEnv that draws one REAL market day at every reset, and (by
    default) carries the fleet SOC across episodes.

    -- Day sampling --
    An episode is 24 h but the training window is a full year, so each reset
    samples a day uniformly at random from the window and pins the env's
    prices/services to that day's real data. Uniform sampling (rather than
    walking the year in calendar order) is deliberate: with ~3 episodes per PPO
    iteration, sequential days would mean the policy sees January for the first
    iterations, February for the next, and so on. That makes the training data
    distribution non-stationary and biases the final policy toward whatever
    season it saw last. Uniform sampling keeps the batch distribution stable
    while still covering the whole year.

    The day sampler deliberately uses its OWN Generator: MultiBESSEnv.reset()
    re-seeds `self._rng` from `_default_seed` on every episode, so a sampler
    sharing `self._rng` would draw the SAME day forever and reintroduce exactly
    the single-day bug this class exists to fix.

    -- SOC carry-over (carry_soc_across_episodes=True) --
    MultiBESSEnv.reset() puts every SOC back to `soc_init` (0.5). That is wrong
    for this problem: a BSP does not wake up at 50% every morning. The MILP
    oracle, the BC roll-out and the evaluation all carry SOC from one day to the
    next (marl_pipeline sets `env._socs` after reset and rebuilds the
    observations), so the PPO was the only agent handed a free, unrealistic
    half-full battery every episode.

    With carry-over enabled, each episode starts from the SOC the fleet actually
    ended the previous episode with. The initial-SOC distribution then becomes
    self-consistent with the policy's own behaviour instead of a fixed point
    mass at 0.5, which is both more realistic and what the evaluation does.
    Carrying SOC across a random day jump is sound: the policy conditions on
    (prices, SOC), and the two are independent, so "any realistic SOC on any
    day" is exactly the state distribution we want it to master.

    NOT carried (deliberately):
      * commitments -- cleared per episode by the base reset(), which matches
        the evaluation exactly (marl_pipeline builds a fresh env per day).
      * FCE / capacity loss -- the base reset() rebuilds the LFP states from
        `fce_cumulative_initial`. Carrying them would age the fleet through
        ~660 episodes of training (about 1.8 simulated years), far beyond the
        one year the evaluation sees. With use_nonlinear_degradation=False the
        degradation cost is linear in throughput and does not read FCE at all,
        so the reward is unaffected either way; with the non-linear model,
        carrying it would actively corrupt training.
    """

    def __init__(self, market_window: Any, day_seed: Optional[int] = None,
                 carry_soc_across_episodes: bool = True, **kwargs):
        # Set before super().__init__(): the base class reads several attrs via
        # getattr() during construction, so this ordering matches the codebase.
        self._market_window = market_window
        self._day_rng = np.random.default_rng(day_seed)
        self._sampled_day: Optional[int] = None
        self._carry_soc = bool(carry_soc_across_episodes)
        self._n_resets = 0
        super().__init__(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Capture the SOC the fleet ENDED the previous episode with, BEFORE
        # super().reset() overwrites it with soc_init. Nothing else touches
        # _socs between episodes, so at this point it still holds the final
        # state of the last roll-out. Skipped on the very first reset, where
        # soc_init is the legitimate cold start.
        carried = None
        if self._carry_soc and self._n_resets > 0:
            carried = np.array(self._socs, dtype=np.float64, copy=True)

        d = int(self._day_rng.integers(0, int(self._market_window.n_days)))
        prices_day, services_day = self._market_window.slice_day(d)
        self._fixed_prices = list(prices_day)
        self._fixed_services = list(services_day)
        self._sampled_day = d

        obs, infos = super().reset(seed=seed, options=options)

        if carried is not None:
            self._socs = carried
            # CRITICAL: super().reset() already built `obs` from soc_init, so
            # they are stale now. Rebuild them from the carried SOC or the
            # policy would act on a 0.5 that is not the real state. This is the
            # same set-_socs-then-rebuild pattern marl_pipeline uses to roll the
            # MILP/BC/eval across days.
            obs = {a: self._build_observation(a) for a in self.agents}

        self._n_resets += 1
        return obs, infos


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

    # ---- Market data: REAL window (sampled per episode) or synthetic fallback ----
    window_key = env_config.get("market_window_key", None)
    market_window = (_MARKET_WINDOW_REGISTRY.get(window_key)
                     if window_key is not None else None)
    if window_key is not None and market_window is None:
        raise ValueError(
            f"market_window_key={window_key!r} was requested but is not in the "
            f"registry (known keys: {sorted(_MARKET_WINDOW_REGISTRY)}). This "
            f"happens when RLlib builds the env in a REMOTE worker process: the "
            f"registry lives in the driver. Keep num_env_runners=0 (the current "
            f"setting) or call register_market_window() inside the worker. "
            f"Refusing to fall back to the synthetic market silently."
        )

    env_kwargs = dict(
        multi_params=fleet,
        episode_hours=episode_hours,
        use_nonlinear_degradation=use_nl,
        fce_cumulative_initial=fce_init,
        seed=seed,
        directional_services=directional,
    )
    if market_window is not None:
        # Offset the day-sampler seed off the env seed so the day sequence is
        # reproducible but not aligned with the env's own RNG stream.
        carry_soc = bool(env_config.get("carry_soc_across_episodes", True))
        raw_env = _WindowSamplingMultiBESSEnv(
            market_window=market_window,
            day_seed=(0 if seed is None else int(seed)) + 7919,
            carry_soc_across_episodes=carry_soc,
            **env_kwargs,
        )
        print(f"  [env] training market: REAL window, "
              f"{int(market_window.n_days)} days, one real day sampled per "
              f"episode (real award_probability from the window)")
        print(f"  [env] SOC across episodes: "
              f"{'CARRIED from previous episode (matches eval roll-out)' if carry_soc else f'RESET to soc_init={raw_env.soc_init} every episode'}")
    else:
        raw_env = MultiBESSEnv(**env_kwargs)
        print("  [env] WARNING: no market_window_key in env_config -> the env "
              "falls back to the SYNTHETIC default sinusoid (80+40*sin) with "
              "award_probability=1.0 on every service. This is NOT the real "
              "market and is NOT comparable to the MILP/BC baselines.")
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
    # Expected-reward mode is a TRAINING-only alignment with the MILP: the PPO
    # training env books reserve revenues in expectation instead of sampling the
    # award/activation Bernoullis. The evaluation rollouts (built elsewhere in the
    # pipeline) never set this, so they still sample real outcomes.
    if env_config.get("expected_reward_training", False):
        raw_env.configure_expected_reward(True)
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
    expected_reward_training: bool = False,
    market_window_key: Optional[str] = None,
    carry_soc_across_episodes: bool = True,
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
        expected_reward_training=expected_reward_training,
        # Key into _MARKET_WINDOW_REGISTRY. None => synthetic fallback (loud).
        market_window_key=market_window_key,
        # Start each episode from the SOC the previous one ended at, instead of
        # snapping back to soc_init=0.5. Matches the MILP/BC/eval roll-out.
        # Only has an effect when a real market window is wired in.
        carry_soc_across_episodes=carry_soc_across_episodes,
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