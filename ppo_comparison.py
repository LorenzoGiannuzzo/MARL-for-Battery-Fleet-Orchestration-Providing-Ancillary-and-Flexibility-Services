# -*- coding: utf-8 -*-
"""
ppo_comparison.py — Aggiunge il PPO (DRL) al confronto MILP vs BC vs PPO.

Si aggancia ai moduli ESISTENTI senza reinventarli:
  - mappo_trainer.build_default_config / train_loop  (shared-policy IPPO, Ray RLlib 2.x)
  - bc_pretraining_multi.transfer_bc_weights_to_algo / verify_transfer (BC->actor)
  - pipeline_step8.evaluate_policy_multi_day  (stesso accounting di BC e random)

Espone due politiche allenate, valutate con lo STESSO loop env di BC/random:
  - PPO "vanilla"        : critic e actor da zero
  - PPO "BC warm-start"  : actor inizializzato dai pesi BC (transfer), critic random,
                           PIU' ancoraggio KL verso la BC congelata (Fix B).

Fix B (ancoraggio KL):
  Il transfer dell'actor funziona (verify_transfer == 1.0) ma il critic parte
  random: nelle prime iterazioni gli advantage rumorosi "lavano via" la policy
  BC. Oltre alle mitigazioni soft (lr=0.1x, entropy=0) aggiungiamo un termine
  beta(t)*KL(pi_BC || pi_theta) che tiene la policy ancorata all'esperto finché
  il critic non si stabilizza, con beta che decade a 0. Vedi bc_kl_anchor.py.

  Le metriche di convergenza (policy/vf loss, entropy, KL nativa, bc_kl, beta,
  timer) sono raccolte da ppo_convergence.train_loop_rich e stashate su
  algo._lorenzo_training_metrics per i grafici di analysis_step8.

NB: gira sulla TUA macchina (Ray + GPU + dati veri). Qui solo il codice.

Uso (dentro run_full_pipeline, dopo aver allenato il BC e calcolato test_demo):

    from ppo_comparison import train_ppo_policy, make_rllib_policy_fn

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

from mappo_trainer import (
    build_default_config, train_loop, register_multi_bess_env,
    SHARED_POLICY_ID,
)
from bc_pretraining_multi import (
    BCPolicyNet, transfer_bc_weights_to_algo, verify_transfer,
)


def train_ppo_policy(
    fleet,
    n_iterations: int,
    bc_net: Optional[BCPolicyNet] = None,
    use_nonlinear_degradation: bool = False,
    episode_hours: int = 24,
    seed: int = 0,
    # washout-mitigation: applicati SOLO quando si parte da warm-start BC,
    # perché l'actor è già buono ma il critic è random.
    warmstart_lr_scale: float = 0.1,
    warmstart_entropy_coeff: float = 0.0,
    # --- Fix B: ancoraggio KL verso la BC congelata (solo warm-start) ---
    bc_kl_beta_start: float = 1.0,
    bc_kl_anneal_iters: Optional[int] = None,
    verbose: bool = True,
    directional_services: bool = False,
    # --- Capacity/rollout overrides — Windows-safe defaults ---
    # On Windows, Ray's plasma shared-memory store (CreateFileMapping) gets
    # exhausted quickly with large per-iter sample batches when N_BESS is high
    # and the MLP is wide. Empirically, train_batch_size=8000 + fcnet=(512,256)
    # crashes after a single iteration on a 50-BESS directional setup with the
    # error "CreateFileMapping() failed. GetLastError() = 1450".
    # The defaults below are conservative: (256, 128) is only modestly bigger
    # than (128, 128) and won't blow up shared memory; train_batch_size=4000
    # is double the original RLlib default (2000) which gives less noisy
    # gradients than the default while staying within Windows constraints.
    # On Linux these can be raised back to (512, 256) / 8000.
    fcnet_hiddens=(256, 128),
    train_batch_size: int = 4000,
    rollout_fragment_length: int = 200,
):
    """Allena una shared-policy PPO sul MultiBESSEnv e restituisce l'algo RLlib."""
    register_multi_bess_env()

    cfg = build_default_config(
        n_batteries=fleet.n_batteries,
        episode_hours=episode_hours,
        use_nonlinear_degradation=use_nonlinear_degradation,
        seed=seed,
        directional_services=directional_services,
        fcnet_hiddens=fcnet_hiddens,
        train_batch_size=train_batch_size,
        rollout_fragment_length=rollout_fragment_length,
    )

    is_warm = bc_net is not None
    if is_warm:
        # 1) Mitigazioni soft: riduci lr e azzera l'entropy bonus per non
        #    distruggere l'actor BC mentre il critic random produce vantaggi
        #    rumorosi (critic washout).
        try:
            base_lr = float(cfg.lr) if getattr(cfg, "lr", None) else 5e-5
        except (TypeError, ValueError):
            base_lr = 5e-5
        cfg = cfg.training(lr=base_lr * warmstart_lr_scale,
                           entropy_coeff=warmstart_entropy_coeff)

        # 2) Fix B: ancoraggio KL verso la policy BC congelata. Complementare
        #    al transfer: il transfer dà il punto di partenza, la KL impedisce
        #    di abbandonarlo prima che il critic sia affidabile.
        if bc_kl_anneal_iters is None:
            bc_kl_anneal_iters = max(1, n_iterations // 2)
        from bc_kl_anchor import make_kl_anchored_config
        cfg = make_kl_anchored_config(
            cfg, bc_net,
            policy_id=SHARED_POLICY_ID,
            beta_start=bc_kl_beta_start,
            anneal_iters=bc_kl_anneal_iters,
        )
        if verbose:
            print(f"  [ppo] Fix B KL anchor: beta_start={bc_kl_beta_start}, "
                  f"anneal_iters={bc_kl_anneal_iters}")

    algo = cfg.build_algo()  # RLlib 2.55: build_algo() (build() is deprecated)

    if is_warm:
        info = transfer_bc_weights_to_algo(bc_net, algo, SHARED_POLICY_ID)
        if verbose:
            print(f"  [ppo] BC->actor transfer: {info['transferred']} tensori "
                  f"({len(info['skipped'])} saltati)")
        # sanity check: la policy PPO appena trasferita deve riprodurre il BC
        try:
            from multi_bess_env import OBS_DIM
            obs_samples = np.random.uniform(-1, 1, size=(256, OBS_DIM)).astype(np.float32)
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
        from ppo_convergence import train_loop_rich
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


def make_rllib_policy_fn(algo, policy_id: str = SHARED_POLICY_ID,
                         deterministic: bool = True,
                         directional_services: bool = False):
    """Adatta un algo RLlib a policy_fn(obs)->action(n_axes,) per
    evaluate_policy_multi_day, così il PPO usa lo STESSO accounting env
    di BC e random (apples-to-apples)."""
    module = algo.get_module(policy_id)
    import torch
    from multi_bess_env import N_ACTION_BINS, ACTION_AXES, ACTION_AXES_DIRECTIONAL
    n_axes = ACTION_AXES_DIRECTIONAL if directional_services else ACTION_AXES

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
        return np.array([np.random.choice(N_ACTION_BINS, p=probs[k])
                         for k in range(n_axes)], dtype=np.int64)

    return policy