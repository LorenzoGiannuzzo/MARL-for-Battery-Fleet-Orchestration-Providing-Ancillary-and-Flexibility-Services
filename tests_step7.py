"""
Step 7 verification: multi-agent behavioural cloning warm-start.

Tests:

  7.1 BCPolicyNet architecture matches the PPO RLModule's actor side
      exactly: same input dim, same hidden sizes, same logit count.
  7.2 Expert demo generation produces the expected dataset size and
      shape, with valid action bin indices (0 to N_ACTION_BINS-1).
  7.3 BC training reduces loss monotonically (mostly) and achieves
      high validation accuracy on this easy MILP-mimicking task.
  7.4 BC policy rolled out on the env achieves a reward significantly
      higher than a random policy (sanity: BC has learned something).
  7.5 Weight transfer is exact: after copying BC weights into the
      RLlib RLModule, argmax actions match between BC and RLModule
      on a held-out observation set.
  7.6 Washout pattern reproduction: when BC weights are transferred
      and PPO does ONE iteration with default lr+entropy, the reward
      drops sharply (this is informational, not a strict assertion).

Run:

    python tests_step7.py
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


def test_bc_architecture_match():
    print("\n--- Step 7.1: BCPolicyNet matches PPO RLModule's actor architecture ---")
    from bc_pretraining_multi import BCPolicyNet, ENCODER_HIDDEN, N_LOGITS
    net = BCPolicyNet()
    # Encoder layer 1: (hidden[0], obs_dim)
    _check(net.encoder_layer1.weight.shape == (ENCODER_HIDDEN[0], net.obs_dim),
           "encoder_layer1.weight shape matches RLModule",
           f"shape {tuple(net.encoder_layer1.weight.shape)}")
    _check(net.encoder_layer2.weight.shape == (ENCODER_HIDDEN[1], ENCODER_HIDDEN[0]),
           "encoder_layer2.weight shape matches RLModule",
           f"shape {tuple(net.encoder_layer2.weight.shape)}")
    _check(net.pi_head.weight.shape == (N_LOGITS, ENCODER_HIDDEN[1]),
           "pi_head.weight shape matches RLModule",
           f"shape {tuple(net.pi_head.weight.shape)}")
    # Forward pass shape check
    import torch
    obs = torch.randn(8, net.obs_dim)
    out = net(obs)
    _check(tuple(out.shape) == (8, 5, 11),
           "forward output shape is (batch, 5 axes, 11 bins)",
           f"got {tuple(out.shape)}")


def test_demo_generation():
    print("\n--- Step 7.2: expert demo generation ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from bc_pretraining_multi import (generate_expert_demos, ACTION_AXES,
                                       N_ACTION_BINS)
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp, bp])
    obs, actions = generate_expert_demos(fleet, n_scenarios=4,
                                          episode_hours=24, seed=0)
    expected = 4 * 2 * 24  # n_scenarios * N * hours
    _check(obs.shape == (expected, 21),
           f"obs has shape ({expected}, 21)",
           f"got {obs.shape}")
    _check(actions.shape == (expected, ACTION_AXES),
           f"actions have shape ({expected}, {ACTION_AXES})",
           f"got {actions.shape}")
    _check(actions.min() >= 0 and actions.max() < N_ACTION_BINS,
           f"all action bins in [0, {N_ACTION_BINS-1}]",
           f"min={actions.min()}, max={actions.max()}")


def test_bc_training_convergence():
    print("\n--- Step 7.3: BC training converges to high accuracy ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from bc_pretraining_multi import generate_expert_demos, train_bc
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp, bp])
    obs, actions = generate_expert_demos(fleet, n_scenarios=8,
                                          episode_hours=24, seed=0)
    net, history = train_bc(obs, actions, n_epochs=20, batch_size=32,
                              lr=3e-3, seed=0)
    initial = history[0]["train_loss"]
    final = history[-1]["train_loss"]
    print(f"    initial loss: {initial:.4f}, final loss: {final:.4f}")
    print(f"    final val acc per axis: {[round(a,3) for a in history[-1]['val_acc_per_axis']]}")
    _check(final < initial * 0.5,
           "final BC loss < 50% of initial loss",
           f"initial={initial:.4f} -> final={final:.4f}")
    mean_acc = float(np.mean(history[-1]["val_acc_per_axis"]))
    _check(mean_acc > 0.8,
           "BC mean validation accuracy across axes > 0.80",
           f"got {mean_acc:.3f}")


def test_bc_outperforms_random():
    print("\n--- Step 7.4: BC policy outperforms random on env ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from bc_pretraining_multi import (generate_expert_demos, train_bc,
                                       BCPolicyNet)
    from multi_bess_env import MultiBESSEnv
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp, bp])
    # Train BC
    obs, actions = generate_expert_demos(fleet, n_scenarios=8,
                                          episode_hours=24, seed=0)
    net, _ = train_bc(obs, actions, n_epochs=20, batch_size=32,
                       lr=3e-3, seed=0)
    # Roll out BC on env
    def rollout_with(net_or_none, n_ep, seed):
        env = MultiBESSEnv(fleet, episode_hours=24, seed=seed)
        rng = np.random.default_rng(seed)
        totals = []
        for ep in range(n_ep):
            o, _ = env.reset(seed=seed + ep)
            ep_total = 0.0
            while env.agents:
                if net_or_none is None:
                    acts = {a: env.action_space(a).sample() for a in env.agents}
                else:
                    acts = {a: net_or_none.predict(o[a], deterministic=True)
                            for a in env.agents}
                o, rews, _, _, _ = env.step(acts)
                ep_total += sum(rews.values())
            totals.append(ep_total)
        return float(np.mean(totals))

    bc_mean = rollout_with(net, n_ep=3, seed=100)
    rnd_mean = rollout_with(None, n_ep=3, seed=100)
    print(f"    BC mean per-episode reward:     {bc_mean:.2f}")
    print(f"    random mean per-episode reward: {rnd_mean:.2f}")
    _check(bc_mean > 2.0 * rnd_mean,
           "BC reward > 2x random reward",
           f"BC {bc_mean:.2f} vs random {rnd_mean:.2f}")


def test_weight_transfer_exact():
    print("\n--- Step 7.5: weight transfer to RLModule is exact (argmax match) ---")
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR",
                 log_to_driver=False)
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from bc_pretraining_multi import (generate_expert_demos, train_bc,
                                       transfer_bc_weights_to_algo,
                                       verify_transfer)
    from mappo_trainer import build_default_config, SHARED_POLICY_ID
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp, bp])
    obs, actions = generate_expert_demos(fleet, n_scenarios=8,
                                          episode_hours=24, seed=0)
    net, _ = train_bc(obs, actions, n_epochs=20, batch_size=32,
                       lr=3e-3, seed=0)
    cfg = build_default_config(n_batteries=2, episode_hours=24,
                                train_batch_size=480, minibatch_size=120,
                                num_epochs=1, num_env_runners=0,
                                rollout_fragment_length=24)
    algo = cfg.build_algo()
    try:
        result = transfer_bc_weights_to_algo(net, algo, SHARED_POLICY_ID)
        _check(result["transferred"] == 6,
               "all 6 BC tensors transferred",
               f"got {result['transferred']}")
        _check(len(result["skipped"]) == 0,
               "no skipped tensors during transfer")
        v = verify_transfer(net, algo, SHARED_POLICY_ID, obs[:50])
        per_axis = v["match_rate_per_axis"]
        overall = v["overall_match_rate"]
        print(f"    argmax match per axis: {[round(r,3) for r in per_axis]}")
        print(f"    overall match rate:    {overall:.3f}")
        _check(overall > 0.99,
               "post-transfer argmax match rate > 0.99 (BC weights live in RLModule)",
               f"got {overall:.3f}")
    finally:
        algo.stop()


def test_washout_reproduction():
    print("\n--- Step 7.6: BC washout pattern reproduces in multi-agent ---")
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR",
                 log_to_driver=False)
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from bc_pretraining_multi import (generate_expert_demos, train_bc,
                                       transfer_bc_weights_to_algo)
    from mappo_trainer import (build_default_config, SHARED_POLICY_ID,
                                train_loop)
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp, bp])
    obs, actions = generate_expert_demos(fleet, n_scenarios=8,
                                          episode_hours=24, seed=0)
    net, _ = train_bc(obs, actions, n_epochs=20, batch_size=32,
                       lr=3e-3, seed=0)
    cfg = build_default_config(n_batteries=2, episode_hours=24,
                                train_batch_size=480, minibatch_size=120,
                                num_epochs=2, num_env_runners=0,
                                rollout_fragment_length=24)
    algo = cfg.build_algo()
    try:
        transfer_bc_weights_to_algo(net, algo, SHARED_POLICY_ID)
        metrics = train_loop(algo, n_iterations=2, log_every=1)
        bc_iter1 = metrics[0]["episode_reward_mean"]
        post_iter2 = metrics[1]["episode_reward_mean"]
        print(f"    iter 1 (BC start):     {bc_iter1:.2f}")
        print(f"    iter 2 (post-update):  {post_iter2:.2f}")
        # Soft assertion: BC starting reward should be substantial (>4000 EUR
        # on this env config); post-update may drop (washout) or stay stable
        # depending on luck. We assert BC start is high and report the drop
        # as informational.
        _check(bc_iter1 > 4000.0,
               "BC starting reward is substantial (> 4000 EUR)",
               f"got {bc_iter1:.2f}")
        if post_iter2 < bc_iter1 * 0.5:
            print(f"    --> WASHOUT detected: iter 2 reward < 50% of iter 1")
            print(f"        ({post_iter2:.2f} / {bc_iter1:.2f} = "
                  f"{100 * post_iter2 / bc_iter1:.1f}%)")
        else:
            print(f"    --> no washout this run (iter 2 = "
                  f"{100 * post_iter2 / bc_iter1:.1f}% of iter 1)")
    finally:
        algo.stop()


def main():
    print("=" * 70)
    print("Step 7: multi-agent behavioural cloning warm-start")
    print("=" * 70)
    test_bc_architecture_match()
    test_demo_generation()
    test_bc_training_convergence()
    test_bc_outperforms_random()
    test_weight_transfer_exact()
    test_washout_reproduction()
    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All Step 7 tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
