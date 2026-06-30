# -*- coding: utf-8 -*-
"""
bc_kl_anchor.py — Fix B: ancoraggio KL verso la policy BC durante il finetune PPO.

PROBLEMA (diagnosticato dai run smoke/production):
  Il warm-start trasferisce correttamente l'actor BC nell'RLModule PPO
  (verify_transfer == 1.0), MA il critic di PPO parte random. Nelle prime
  iterazioni gli advantage stimati dal critic random hanno segno arbitrario;
  PPO aggiorna l'actor in direzioni sbagliate e "lava via" (washout) la
  policy BC prima che il critic si stabilizzi. Le mitigazioni lr-basso /
  entropy-0 rallentano ma non impediscono il fenomeno (warm parte da
  mean_rew identico al vanilla, non dal livello del BC).

SOLUZIONE (Fix B):
  Aggiungiamo all'obiettivo PPO un termine di ancoraggio
        L_total = L_PPO  +  beta(t) * E[ KL( pi_BC || pi_theta ) ]
  dove pi_BC è una SNAPSHOT CONGELATA della policy BC (presa subito dopo il
  transfer) e beta(t) decade linearmente da beta_start a 0 in
  `anneal_iters` iterazioni. Finché il critic non è affidabile, il termine KL
  tiene pi_theta vicino all'esperto; man mano che il critic impara e beta->0,
  PPO riprende il controllo e può migliorare oltre il BC.

  Usiamo KL( pi_BC || pi_theta ) (forward KL, "mass-covering") perché penalizza
  pi_theta quando NON copre le azioni che il BC sceglieva: è esattamente
  l'ancoraggio "non dimenticare cosa faceva l'esperto".

IMPLEMENTAZIONE:
  - `BCReferenceModule`: wrappa la BCPolicyNet congelata; produce i logit di
    riferimento per ogni batch di osservazioni.
  - `KLAnchoredPPOTorchLearner`: sottoclasse di PPOTorchLearner. Override di
    `compute_loss_for_module` che chiama super() per la loss PPO standard e
    aggiunge beta*KL. Logga `bc_kl` e `bc_kl_beta` tra le metriche del learner,
    così finiscono automaticamente nei grafici di convergenza.
  - `make_kl_anchored_config(...)`: helper che prende una PPOConfig già
    costruita e vi inserisce questo learner + i parametri di ancoraggio.

NB: new API stack, RLlib 2.4x+. Nessuna dipendenza extra oltre a torch/ray.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override


# ============================================================================
# Frozen BC reference policy
# ============================================================================

class BCReferenceModule:
    """Snapshot congelata della policy BC, usata come riferimento per la KL.

    Tiene una copia (deep, in eval mode, requires_grad=False) dei pesi della
    BCPolicyNet al momento del transfer. Espone `logits(obs)` che restituisce
    i logit per-axis (batch, n_axes, n_bins) SENZA gradiente.
    """

    def __init__(self, bc_net, device: str = "cpu"):
        import copy
        self.net = copy.deepcopy(bc_net).to(device)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.n_axes = int(bc_net.n_axes)

    @torch.no_grad()
    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        """(batch, obs_dim) -> (batch, n_axes, n_bins), no grad."""
        obs = obs.to(self.device, dtype=torch.float32)
        return self.net.forward(obs)  # BCPolicyNet already reshapes per-axis


# Module-level registry so the Learner (which RLlib instantiates internally)
# can retrieve the frozen BC reference that was set up on the driver.
_BC_REFERENCE_REGISTRY: Dict[str, BCReferenceModule] = {}


def register_bc_reference(policy_id: str, ref: BCReferenceModule) -> None:
    _BC_REFERENCE_REGISTRY[policy_id] = ref


def get_bc_reference(policy_id: str) -> Optional[BCReferenceModule]:
    return _BC_REFERENCE_REGISTRY.get(policy_id)


# ============================================================================
# KL-anchored PPO learner
# ============================================================================

class KLAnchoredPPOTorchLearner(PPOTorchLearner):
    """PPO learner che aggiunge beta(t) * KL(pi_BC || pi_theta) alla loss.

    Parametri di ancoraggio letti da config.learner_config_dict:
      - bc_kl_beta_start : float   (default 1.0)  coefficiente iniziale
      - bc_kl_anneal_iters : int   (default 20)   iterazioni per beta->0
      - bc_kl_policy_id : str      (default 'shared_policy')

    L'annealing è gestito in `before_gradient_based_update`, che incrementa un
    contatore di iterazioni e ricalcola beta corrente.
    """

    @override(PPOTorchLearner)
    def build(self) -> None:
        super().build()
        lcd = self.config.learner_config_dict or {}
        self._bc_kl_beta_start = float(lcd.get("bc_kl_beta_start", 1.0))
        self._bc_kl_anneal_iters = int(lcd.get("bc_kl_anneal_iters", 20))
        # Fix 2: floor below which beta is NOT allowed to decay. The original
        # anneal drove beta linearly to 0, which dissolved the BC anchor on the
        # final iterations and let the warm-start policy drift back into the
        # degenerate overcommit pattern (observed: iter 190->200 reward collapse
        # 16004->13365 as beta->0). A floor keeps a residual anchor for the whole
        # run so the policy stays in the BC basin. 0.1 by default.
        self._bc_kl_beta_floor = float(lcd.get("bc_kl_beta_floor", 0.1))
        self._bc_kl_policy_id = lcd.get("bc_kl_policy_id", "shared_policy")
        self._bc_kl_iter = 0
        self._bc_kl_beta_now = self._bc_kl_beta_start

    @override(PPOTorchLearner)
    def before_gradient_based_update(self, *, timesteps: Dict[str, Any]) -> None:
        super().before_gradient_based_update(timesteps=timesteps)
        # Linear anneal beta_start -> beta_floor over anneal_iters iterations.
        # The floor keeps a residual BC anchor for the whole run (Fix 2):
        # instead of beta_start*(1-frac) which reaches 0, we interpolate down
        # to beta_floor and hold there.
        frac = min(1.0, self._bc_kl_iter / max(1, self._bc_kl_anneal_iters))
        floor = getattr(self, "_bc_kl_beta_floor", 0.0)
        self._bc_kl_beta_now = floor + (self._bc_kl_beta_start - floor) * (1.0 - frac)
        self._bc_kl_iter += 1

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self, *, module_id, config, batch, fwd_out,
    ):
        # 1) Standard PPO loss (logs policy_loss, vf_loss, entropy, kl, etc.)
        total_loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )

        # 2) BC anchoring term, only if a frozen reference exists for this module.
        ref = get_bc_reference(module_id) or get_bc_reference(self._bc_kl_policy_id)
        beta = float(getattr(self, "_bc_kl_beta_now", 0.0))
        if ref is None or beta <= 0.0:
            # still log a zero so the chart has a continuous series
            self.metrics.log_dict(
                {"bc_kl": 0.0, "bc_kl_beta": beta},
                key=module_id, window=1,
            )
            return total_loss

        module = self.module[module_id].unwrapped()
        obs = batch[Columns.OBS]
        # Current policy logits for this batch
        cur_logits = fwd_out[Columns.ACTION_DIST_INPUTS]            # (B, n_axes*n_bins)
        n_axes = ref.n_axes
        B = cur_logits.shape[0]
        n_bins = cur_logits.shape[-1] // n_axes
        cur_logits = cur_logits.view(B, n_axes, n_bins)
        # Frozen BC reference logits for the SAME observations
        ref_logits = ref.logits(obs)                               # (B, n_axes, n_bins)
        if ref_logits.shape != cur_logits.shape:
            ref_logits = ref_logits.view(B, n_axes, n_bins)

        # Forward KL: sum_a softmax(ref) * (logsoftmax(ref) - logsoftmax(cur)),
        # summed over bins, meaned over axes and batch.
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        cur_logp = torch.log_softmax(cur_logits, dim=-1)
        ref_p = ref_logp.exp()
        kl_per_axis = (ref_p * (ref_logp - cur_logp)).sum(dim=-1)  # (B, n_axes)
        bc_kl = kl_per_axis.mean()

        total_loss = total_loss + beta * bc_kl

        self.metrics.log_dict(
            {"bc_kl": bc_kl, "bc_kl_beta": beta},
            key=module_id, window=1,
        )
        return total_loss


# ============================================================================
# Config helper
# ============================================================================

def make_kl_anchored_config(
    base_config,
    bc_net,
    policy_id: str = "shared_policy",
    beta_start: float = 1.0,
    anneal_iters: int = 20,
    beta_floor: float = 0.1,
    device: str = "cpu",
):
    """Inserisce il learner KL-ancorato e registra la BC reference congelata.

    Va chiamata DOPO build_default_config e PRIMA di build_algo().
    Restituisce la config modificata. Il transfer dei pesi BC nell'actor va
    comunque fatto (transfer_bc_weights_to_algo) dopo build_algo: l'ancoraggio
    KL e il transfer sono complementari (transfer = punto di partenza,
    KL = vincolo che impedisce di allontanarsene troppo presto).
    """
    # Register the frozen BC reference for the learner to retrieve.
    register_bc_reference(policy_id, BCReferenceModule(bc_net, device=device))

    learner_cfg = {
        "bc_kl_beta_start": float(beta_start),
        "bc_kl_anneal_iters": int(anneal_iters),
        "bc_kl_beta_floor": float(beta_floor),
        "bc_kl_policy_id": policy_id,
    }
    # Newer RLlib: .learners(); older: .training(). Prefer .learners() and
    # fall back gracefully so the same code runs on either version.
    try:
        cfg = base_config.learners(
            learner_class=KLAnchoredPPOTorchLearner,
            learner_config_dict=learner_cfg,
        )
    except (AttributeError, TypeError):
        cfg = base_config.training(
            learner_class=KLAnchoredPPOTorchLearner,
            learner_config_dict=learner_cfg,
        )
    return cfg