# -*- coding: utf-8 -*-
"""
marl_ppo_comparison.py — Aggiunge il PPO (DRL) al confronto MILP vs BC vs PPO.

Si aggancia ai moduli ESISTENTI senza reinventarli:
  - marl_trainer.build_default_config / train_loop  (shared-policy IPPO, Ray RLlib 2.x)
  - marl_bc.transfer_bc_weights_to_algo / verify_transfer (BC->actor)
  - marl_pipeline.evaluate_policy_multi_day  (stesso accounting di BC e random)

Espone due politiche allenate, valutate con lo STESSO loop env di BC/random:
  - PPO "vanilla"        : critic e actor da zero
  - PPO "BC warm-start"  : actor inizializzato dai pesi BC (transfer), critic random,
                           PIU' ancoraggio KL verso la BC congelata (Fix B).

Fix B (ancoraggio KL):
  Il transfer dell'actor funziona (verify_transfer == 1.0) ma il critic parte
  random: nelle prime iterazioni gli advantage rumorosi "lavano via" la policy
  BC. Oltre alle mitigazioni soft (lr=0.1x, entropy=0) aggiungiamo un termine
  beta(t)*KL(pi_BC || pi_theta) che tiene la policy ancorata all'esperto finché
  il critic non si stabilizza, con beta che decade a 0. Vedi marl_bc_kl_anchor.py.

  Le metriche di convergenza (policy/vf loss, entropy, KL nativa, bc_kl, beta,
  timer) sono raccolte da marl_ppo_convergence.train_loop_rich e stashate su
  algo._lorenzo_training_metrics per i grafici di marl_analysis.

NB: gira sulla TUA macchina (Ray + GPU + dati veri). Qui solo il codice.

Uso (dentro run_full_pipeline, dopo aver allenato il BC e calcolato test_demo):

    from marl_ppo_comparison import train_ppo_policy, make_rllib_policy_fn

    ppo_vanilla = train_ppo_policy(fleet, n_iterations=..., bc_net=None,
                                   use_nonlinear_degradation=use_nonlinear_degradation)
    ppo_warm    = train_ppo_policy(fleet, n_iterations=..., bc_net=bc_net,
                                   use_nonlinear_degradation=use_nonlinear_degradation)

    for tag, algo in (("ppo_vanilla", ppo_vanilla), ("ppo_bc", ppo_warm)):
        ev = evaluate_policy_multi_day(
            fleet, test_window,
            policy_fn=make_rllib_policy_fn(algo),
            use_nonlinear_degradation=use_nonlinear_degradation,
            initial_soc=test_initial_soc,
            initial_fce_cumulative=test_initial_fce,
            env_seed=eval_seed + 1000,
        )
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

import numpy as np

from marl_trainer import (
    build_default_config, train_loop, register_multi_bess_env,
    SHARED_POLICY_ID,
)
from marl_bc import (
    BCPolicyNet, transfer_bc_weights_to_algo, verify_transfer,
)


def train_ppo_policy(
    fleet,
    n_iterations: int,
    bc_net: Optional[BCPolicyNet] = None,
    # July 2026 fix: the REAL market window the PPO trains on. When None the
    # training env silently fell back to a fixed synthetic sinusoid with
    # award_probability=1.0, while MILP/BC used the real window -- the PPO was
    # the only agent trained in a different world. Pass train_window here.
    market_window=None,
    # July 2026 fix: start each training episode from the SOC the previous one
    # ended at, instead of resetting to soc_init=0.5. The MILP oracle, the BC
    # roll-out and the evaluation all carry SOC across days; the PPO was the
    # only agent given a fresh half-full battery every episode. Requires
    # market_window (no effect on the synthetic fallback env).
    carry_soc_across_episodes: bool = True,
    use_nonlinear_degradation: bool = False,
    episode_hours: int = 24,
    seed: int = 0,
    # ---------------- WARM-START FACTORS (ablation axes) -----------------
    # These four used to move together whenever bc_net was not None, which
    # is exactly the confound Reviewers 1 (pt 1), 2 (pt 6) and 3 (pt 1)
    # raised: the reported warm-start gain could not be attributed to
    # behavioural cloning because initialisation, learning rate, entropy
    # bonus and the KL anchor all changed at once. Each is now independent:
    #
    #   factor 1  initialisation   bc_net is not None
    #   factor 2  learning rate    warmstart_lr_scale  (1.0 = same as vanilla)
    #   factor 3  entropy bonus    warmstart_entropy_coeff (None = keep base)
    #   factor 4  KL anchoring     use_kl_anchor
    #
    # Defaults reproduce the pre-revision warm start exactly, so existing
    # call sites are unaffected.
    warmstart_lr_scale: float = 0.1,
    warmstart_entropy_coeff: Optional[float] = 0.0,
    use_kl_anchor: bool = True,
    # --- Fix B: ancoraggio KL verso la BC congelata (solo warm-start) ---
    bc_kl_beta_start: float = 1.0,
    bc_kl_anneal_iters: Optional[int] = None,
    bc_kl_beta_floor: float = 0.1,
    verbose: bool = True,
    directional_services: bool = False,
    # --- Capacity/rollout overrides — Windows-safe AND BC-compatible ---
    # CRITICAL: fcnet_hiddens MUST match the BC network architecture
    # (BCPolicyNet ENCODER_HIDDEN in marl_bc.py). If they
    # differ, transfer_bc_weights_to_algo silently skips the mismatched
    # encoder tensors and the "warm-start" becomes mostly random.
    # Symptom in the log: "BC tensor(s) not transferred" + low overall
    # match_rate (~0.54 instead of ~1.0). The KL anchor then pulls the
    # random-ish PPO policy toward the BC, but very slowly, and the warm
    # ends up worse than vanilla.
    # SET TO (256, 128): this is the PPO architecture from the run where the
    # warm-start beat the MILP by ~28% (14.7M vs 11.5M EUR on the 2024 test
    # window). BCPolicyNet.ENCODER_HIDDEN has been aligned to (256, 128) too,
    # so the transfer is now clean (verify_transfer == 1.0) AND the winning
    # PPO capacity is restored. Keep these two in lock-step.
    fcnet_hiddens=(512, 256),
    # train_batch_size=4000: restored to the value used in the 14M run.
    # NB: was temporarily lowered to 2000 for the 410-dim full-foresight OOM on
    # an 8 GB machine; keep 4000 only if memory allows (fewer BESS, no 410-dim
    # obs, or more free RAM), otherwise it may OOM again.
    train_batch_size: int = 4000,
    rollout_fragment_length: int = 200,
    enable_commitments: bool = False,
    commitment_lead_time: int = 24,
    penalty_k: float = 1.5,
    full_foresight: bool = False,
    overcommit_penalty: float = 0.0,
    expected_reward_training: bool = False,
    sustain=None,
):
    """Allena una shared-policy PPO sul MultiBESSEnv e restituisce l'algo RLlib."""
    # NOTE on Ray init: we deliberately do NOT call ray.init() here. Earlier runs
    # worked when RLlib auto-initialised Ray with its own defaults; every custom
    # ray.init we tried (object_store cap, local_mode) broke on this Windows / Ray
    # version. The real fix for the per-iteration memory growth is
    # num_env_runners=0 (set in build_default_config), which keeps rollouts in the
    # main process with no separate worker actor. Let RLlib handle Ray itself.
    register_multi_bess_env()

    # ---- Wire the REAL market window into the training env ----
    # One real day is sampled per episode; the window's own per-hour services
    # carry the real award_probability / capacity_price / energy_price.
    window_key = None
    if market_window is not None:
        from marl_trainer import register_market_window
        window_key = register_market_window(
            market_window, key=f"ppo_train_window_seed{seed}")
        if verbose:
            print(f"  [ppo] training market: REAL window "
                  f"({int(market_window.n_days)} days), one day sampled per "
                  f"episode")
            print(f"  [ppo] SOC across episodes: "
                  f"{'carried over' if carry_soc_across_episodes else 'reset to soc_init'}")
    elif verbose:
        print("  [ppo] WARNING: market_window=None -> training on the SYNTHETIC "
              "sinusoid with award_probability=1.0. Results will NOT be "
              "comparable to the MILP/BC baselines.")

    cfg = build_default_config(
        n_batteries=fleet.n_batteries,
        episode_hours=episode_hours,
        use_nonlinear_degradation=use_nonlinear_degradation,
        seed=seed,
        directional_services=directional_services,
        fcnet_hiddens=fcnet_hiddens,
        train_batch_size=train_batch_size,
        rollout_fragment_length=rollout_fragment_length,
        enable_commitments=enable_commitments,
        commitment_lead_time=commitment_lead_time,
        penalty_k=penalty_k,
        full_foresight=full_foresight,
        overcommit_penalty=overcommit_penalty,
        expected_reward_training=expected_reward_training,
        market_window_key=window_key,
        carry_soc_across_episodes=carry_soc_across_episodes,
        sustain_hours=sustain,
    )

    is_warm = bc_net is not None
    # Which warm-start factors are actually active in THIS run. Logged so
    # that every line of the ablation table can be traced back to a
    # configuration without re-reading the call site.
    active_factors = []
    if is_warm:
        active_factors.append("init(BC)")
        # --- factor 2: learning rate ---------------------------------
        # warmstart_lr_scale == 1.0 leaves the vanilla learning rate in
        # place, which is what isolates initialisation from the LR change.
        try:
            base_lr = float(cfg.lr) if getattr(cfg, "lr", None) else 5e-5
        except (TypeError, ValueError):
            base_lr = 5e-5
        training_kwargs = {}
        if abs(float(warmstart_lr_scale) - 1.0) > 1e-12:
            training_kwargs["lr"] = base_lr * float(warmstart_lr_scale)
            active_factors.append(f"lr x{warmstart_lr_scale:g}")
        # --- factor 3: entropy bonus ---------------------------------
        # None means "do not touch it", i.e. keep the vanilla entropy
        # coefficient. 0.0 removes the bonus (the pre-revision behaviour).
        if warmstart_entropy_coeff is not None:
            training_kwargs["entropy_coeff"] = float(warmstart_entropy_coeff)
            active_factors.append(f"entropy={warmstart_entropy_coeff:g}")
        if training_kwargs:
            cfg = cfg.training(**training_kwargs)

        # --- factor 4: KL anchoring toward the frozen clone -----------
        # Complementary to the transfer: the transfer supplies the starting
        # point, the KL term stops the policy leaving it before the critic
        # is reliable. Switchable so the anchor can be tested on its own.
        if use_kl_anchor:
            if bc_kl_anneal_iters is None:
                bc_kl_anneal_iters = max(1, n_iterations // 2)
            from marl_bc_kl_anchor import make_kl_anchored_config
            cfg = make_kl_anchored_config(
                cfg, bc_net,
                policy_id=SHARED_POLICY_ID,
                beta_start=bc_kl_beta_start,
                anneal_iters=bc_kl_anneal_iters,
                beta_floor=bc_kl_beta_floor,
            )
            active_factors.append(f"kl_anchor(beta0={bc_kl_beta_start:g},"
                                  f"floor={bc_kl_beta_floor:g})")
            if verbose:
                print(f"  [ppo] KL anchor: beta_start={bc_kl_beta_start}, "
                      f"anneal_iters={bc_kl_anneal_iters}, "
                      f"beta_floor={bc_kl_beta_floor}")
        elif verbose:
            print("  [ppo] KL anchor: DISABLED (ablation)")
    if verbose:
        print(f"  [ppo] warm-start factors active: "
              f"{', '.join(active_factors) if active_factors else 'none (vanilla)'}")

    algo = cfg.build_algo()  # RLlib 2.55: build_algo() (build() is deprecated)

    if is_warm:
        info = transfer_bc_weights_to_algo(bc_net, algo, SHARED_POLICY_ID)
        if verbose:
            print(f"  [ppo] BC->actor transfer: {info['transferred']} tensori "
                  f"({len(info['skipped'])} saltati)")
        # sanity check: la policy PPO appena trasferita deve riprodurre il BC
        try:
            # Use the BC net's OWN input dimension, not the module constant
            # OBS_DIM. With full_foresight the observation is 410-dim, so a
            # fixed OBS_DIM=26 here builds the wrong-shaped probe and the check
            # fails with a shape-mismatch ("1x26 and 410x512") even though the
            # transfer itself was fine. bc_net.obs_dim is always correct.
            from marl_env import OBS_DIM
            probe_dim = int(getattr(bc_net, "obs_dim", OBS_DIM))
            obs_samples = np.random.uniform(-1, 1, size=(256, probe_dim)).astype(np.float32)
            vt = verify_transfer(bc_net, algo, SHARED_POLICY_ID, obs_samples)
            if verbose:
                print(f"  [ppo] verify_transfer: {vt}")
        except Exception as exc:
            if verbose:
                print(f"  [ppo] verify_transfer skipped: {exc}")

    t0 = time.time()
    # train_loop_rich: come train_loop ma raccoglie anche le metriche interne
    # del learner (policy/vf loss, entropy, KL nativa, bc_kl, beta, timer) per
    # i grafici di convergenza. Retro-compatibile (espone episode_reward_mean).
    try:
        from marl_ppo_convergence import train_loop_rich
        metrics = train_loop_rich(algo, n_iterations=n_iterations,
                                  policy_id=SHARED_POLICY_ID, verbose=verbose)
    except Exception as exc:
        # Fallback al train_loop classico se ppo_convergence non è disponibile
        if verbose:
            print(f"  [ppo] train_loop_rich non disponibile ({exc}); uso train_loop")
        metrics = train_loop(algo, n_iterations=n_iterations)

    if verbose:
        last = metrics[-1] if isinstance(metrics, list) and metrics else metrics
        print(f"  [ppo] {'BC-warm' if is_warm else 'vanilla'} trained "
              f"{n_iterations} iters in {time.time()-t0:.1f}s; last={last}")
    # Stash metrics on the algo object for downstream extraction by analysis code
    try:
        algo._lorenzo_training_metrics = metrics
    except Exception:
        pass
    return algo


def diagnose_zero_policy(algo, observations, directional_services: bool,
                         bc_net=None, policy_id: str = SHARED_POLICY_ID,
                         label: str = "policy") -> dict:
    """Explain why a trained policy scored exactly zero at evaluation.

    Evaluation uses the per-axis ARGMAX while training samples from the
    distribution. Those can disagree completely: with 11 bins per axis and a
    near-uniform distribution, the mode can sit on bin 0 (the no-op) while
    the sampled behaviour still earns. So a profit of exactly 0.00 EUR admits
    two very different readings, and they call for opposite fixes:

      the policy is DESTROYED   -> protect the initialisation
      the argmax is the no-op   -> evaluate as trained, or sharpen the policy

    This reports the numbers that separate them:

      argmax histogram      a single spike at bin 0 means the mode is the
                            no-op on that axis
      mean max probability  1/11 = 0.091 is uniform, so the argmax is
                            essentially arbitrary; near 1.0 is a confident
                            policy
      entropy per axis      ln(11) = 2.398 is the uniform maximum
      match vs the clone    verify_transfer runs BEFORE the updates; this
                            runs AFTER, so it says how far the policy moved

    Never raises: a diagnostic must not be able to break the run it is
    diagnosing.
    """
    import numpy as _np
    out = {}
    try:
        import torch
        from marl_env import (N_ACTION_BINS, ACTION_AXES,
                              ACTION_AXES_DIRECTIONAL)
        n_axes = ACTION_AXES_DIRECTIONAL if directional_services else ACTION_AXES
        module = algo.get_module(policy_id)

        obs = _np.asarray(observations, dtype=_np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        t_obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits = module.forward_inference(
                {"obs": t_obs})["action_dist_inputs"]
        logits = logits.detach().cpu().numpy().reshape(
            -1, n_axes, N_ACTION_BINS)

        argmax = logits.argmax(axis=-1)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = _np.exp(shifted)
        probs /= probs.sum(axis=-1, keepdims=True)
        max_p = probs.max(axis=-1).mean(axis=0)
        ent = (-(probs * _np.log(probs + 1e-12)).sum(axis=-1)).mean(axis=0)
        frac_zero = (argmax == 0).mean(axis=0)

        out["argmax_frac_bin0"] = [float(x) for x in frac_zero]
        out["mean_max_prob"] = [float(x) for x in max_p]
        out["entropy_per_axis"] = [float(x) for x in ent]

        print(f"  [diag/{label}] n_obs={obs.shape[0]}, "
              f"uniform reference: max_prob=0.091, entropy=2.398")
        print(f"  [diag/{label}] fraction of observations whose argmax is "
              f"bin 0, per axis:")
        print(f"                 {[f'{x:.2f}' for x in frac_zero]}")
        print(f"  [diag/{label}] mean max probability per axis:")
        print(f"                 {[f'{x:.3f}' for x in max_p]}")
        print(f"  [diag/{label}] entropy per axis:")
        print(f"                 {[f'{x:.3f}' for x in ent]}")

        if bc_net is not None:
            with torch.no_grad():
                bc_logits = bc_net(t_obs)
            bc_a = bc_logits.detach().cpu().numpy().reshape(
                -1, n_axes, N_ACTION_BINS).argmax(axis=-1)
            per_axis = (argmax == bc_a).mean(axis=0)
            overall = float((argmax == bc_a).mean())
            out["bc_match_per_axis"] = [float(x) for x in per_axis]
            out["bc_match_overall"] = overall
            print(f"  [diag/{label}] agreement with the clone AFTER training: "
                  f"{overall:.3f} overall")
            print(f"                 per axis "
                  f"{[f'{x:.2f}' for x in per_axis]}")
            bc_zero = (bc_a == 0).mean(axis=0)
            print(f"  [diag/{label}] for reference, the clone's own argmax is "
                  f"bin 0 this often:")
            print(f"                 {[f'{x:.2f}' for x in bc_zero]}")
    except Exception as exc:
        print(f"  [diag/{label}] diagnostic unavailable: "
              f"{type(exc).__name__}: {exc}")
    return out


def make_rllib_policy_fn(algo, policy_id: str = SHARED_POLICY_ID,
                         deterministic: bool = True,
                         directional_services: bool = False,
                         rng: Optional[np.random.Generator] = None):
    """Adatta un algo RLlib a policy_fn(obs)->action(n_axes,) per
    evaluate_policy_multi_day, così il PPO usa lo STESSO accounting env
    di BC e random (apples-to-apples).

    rng: se passato e deterministic=False, il sampling stocastico usa questo
    Generator invece dello stato globale np.random, per riproducibilità."""
    module = algo.get_module(policy_id)
    import torch
    from marl_env import N_ACTION_BINS, ACTION_AXES, ACTION_AXES_DIRECTIONAL
    n_axes = ACTION_AXES_DIRECTIONAL if directional_services else ACTION_AXES
    _choice = rng.choice if rng is not None else np.random.choice

    def policy(obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        out = module.forward_inference({"obs": t})
        logits = out["action_dist_inputs"]                  # (1, n_axes*11)
        logits = logits.detach().cpu().numpy().reshape(-1)
        logits = logits.reshape(n_axes, N_ACTION_BINS)
        if deterministic:
            return logits.argmax(axis=-1).astype(np.int64)
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        return np.array([_choice(N_ACTION_BINS, p=probs[k])
                         for k in range(n_axes)], dtype=np.int64)

    return policy