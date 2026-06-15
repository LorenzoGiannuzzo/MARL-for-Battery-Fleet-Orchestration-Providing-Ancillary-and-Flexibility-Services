"""
Step 6 verification: MAPPO trainer setup (shared-policy PPO on RLlib).

Tests:

  6.1 Env registers and wraps cleanly in ParallelPettingZooEnv. Reset
      and step both return RLlib-compatible MultiAgentDict structures.
  6.2 PPOConfig builds with shared-policy multi-agent setting. The
      single policy id `shared_bess_policy` is correctly mapped from
      every agent id by the policy_mapping_fn.
  6.3 Algorithm builds and one training iteration completes without
      errors (smoke test, 1 PPO update over ~500 timesteps).
  6.4 Training produces monotonically (or at least non-decreasingly)
      improving mean episode reward over a small number of iterations.
      With a fresh policy starting from zero, the first 3-5 iterations
      should show clear learning on the easy multi-BESS BSP task.
  6.5 The trained policy can be evaluated on the env using the new
      API stack (`algo.get_module()` + `forward_inference()`). The
      evaluation produces a finite reward value and per-agent metrics
      (arbitrage, flex, degradation) that sum to the total.

NOTE on runtime: this test file performs ONE real (but small) training
run with 3 iterations. On a CPU laptop it takes ~5-10 seconds. The test
is NOT a verification of policy convergence (that requires many more
iterations and is intended to be run by the user on their own hardware),
only of pipeline correctness.

Run:

    python tests_step6.py
"""

from __future__ import annotations

import sys
import warnings
import numpy as np

warnings.simplefilter("ignore")

import logging
logging.getLogger("ray").setLevel(logging.ERROR)

_FAILED = []


def _check(cond, name, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def test_env_wrapping():
    print("\n--- Step 6.1: env registers and wraps in ParallelPettingZooEnv ---")
    from mappo_trainer import register_multi_bess_env, _env_creator
    register_multi_bess_env()
    wrapped = _env_creator({
        "fleet_kind": "homogeneous",
        "n_batteries": 3,
        "episode_hours": 24,
        "seed": 0,
    })
    obs, infos = wrapped.reset()
    _check(isinstance(obs, dict) and len(obs) == 3,
           "reset returns dict of 3 agent observations",
           f"got {len(obs)} keys: {list(obs.keys())}")
    expected_ids = {"bess_0", "bess_1", "bess_2"}
    _check(set(obs.keys()) == expected_ids,
           "agent ids match expected pattern",
           f"got {set(obs.keys())}")
    # Step with random actions (per-agent action space is exposed via dict)
    sample = wrapped.action_space.sample()
    if isinstance(sample, dict):
        actions = sample
    else:
        actions = {a: sample for a in obs}
    out = wrapped.step(actions)
    _check(len(out) == 5,
           "step returns 5-tuple (obs, rew, term, trunc, info)")
    obs2, rews, terms, truncs, infos = out
    _check(set(rews.keys()) == expected_ids,
           "rewards keyed by agent ids")


def test_config_build():
    print("\n--- Step 6.2: PPOConfig with shared policy builds ---")
    from mappo_trainer import build_default_config, SHARED_POLICY_ID
    cfg = build_default_config(n_batteries=3, episode_hours=24,
                                train_batch_size=480, minibatch_size=120,
                                num_epochs=2, num_env_runners=0,
                                rollout_fragment_length=24)
    _check(cfg.framework_str == "torch",
           "framework is torch")
    _check(SHARED_POLICY_ID in cfg.policies,
           f"single shared policy id {SHARED_POLICY_ID} is registered",
           f"policies={set(cfg.policies)}")
    # Test the policy mapping function
    for aid in ["bess_0", "bess_1", "bess_2"]:
        mapped = cfg.policy_mapping_fn(aid)
        _check(mapped == SHARED_POLICY_ID,
               f"agent {aid} maps to {SHARED_POLICY_ID}",
               f"got {mapped}")


def test_one_training_iteration():
    print("\n--- Step 6.3: one training iteration completes ---")
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR",
                 log_to_driver=False)
    from mappo_trainer import build_default_config
    cfg = build_default_config(n_batteries=2, episode_hours=24,
                                train_batch_size=480, minibatch_size=120,
                                num_epochs=2, num_env_runners=0,
                                rollout_fragment_length=24)
    algo = cfg.build_algo()
    try:
        result = algo.train()
        _check(True, "first training iteration completes without error")
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_return_mean")
        _check(mean_rew is not None and np.isfinite(mean_rew),
               "iteration returns finite episode_return_mean",
               f"got {mean_rew}")
    finally:
        algo.stop()


def test_training_progress():
    print("\n--- Step 6.4: training shows learning over 3 iterations ---")
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR",
                 log_to_driver=False)
    from mappo_trainer import build_default_config, train_loop
    cfg = build_default_config(n_batteries=2, episode_hours=24,
                                train_batch_size=480, minibatch_size=120,
                                num_epochs=2, num_env_runners=0,
                                rollout_fragment_length=24)
    algo = cfg.build_algo()
    try:
        metrics = train_loop(algo, n_iterations=3, log_every=1)
        rewards = [m["episode_reward_mean"] for m in metrics]
        print(f"    rewards across 3 iterations: {[round(r, 1) for r in rewards]}")
        _check(all(r is not None and np.isfinite(r) for r in rewards),
               "all iterations return finite rewards")
        # On this easy environment, a fresh policy improves rapidly. Allow
        # noise but require iter 3 reward > iter 1 reward.
        _check(rewards[-1] > rewards[0],
               "final iteration reward > first iteration reward (learning)",
               f"{rewards[0]:.2f} -> {rewards[-1]:.2f}")
    finally:
        # Keep algo alive for the next test (evaluation)
        pass
    return algo


def test_evaluation_pipeline(algo):
    print("\n--- Step 6.5: trained policy evaluates via new API stack ---")
    from mappo_trainer import evaluate_policy
    try:
        ev = evaluate_policy(algo, n_episodes=3, deterministic=False)
        print(f"    mean total reward: {ev['mean_total_reward']:.2f}")
        print(f"    arbitrage: {ev['mean_arbitrage']:.2f}")
        print(f"    flex:      {ev['mean_flex']:.2f}")
        print(f"    deg:       {ev['mean_degradation']:.2f}")
        _check(np.isfinite(ev["mean_total_reward"]),
               "evaluation produces finite mean total reward")
        components_sum = (ev["mean_arbitrage"] + ev["mean_flex"]
                          - ev["mean_degradation"])
        _check(abs(ev["mean_total_reward"] - components_sum) < 1e-3,
               "reward decomposes into arb + flex - deg",
               f"reward={ev['mean_total_reward']:.4f}, "
               f"components_sum={components_sum:.4f}")
    finally:
        algo.stop()


def main():
    print("=" * 70)
    print("Step 6: MAPPO trainer with shared-policy PPO on Ray RLlib")
    print("=" * 70)
    test_env_wrapping()
    test_config_build()
    test_one_training_iteration()
    algo = test_training_progress()
    test_evaluation_pipeline(algo)
    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All Step 6 tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
