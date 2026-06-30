"""
Multi-agent behavioural cloning warm-start for the BSP-aggregated PPO.

Step 7 of the multi-agent track. Generates expert demonstrations by
running the multi-BESS MILP (Step 4) on training scenarios, converts the
continuous MILP solutions to discrete MultiDiscrete actions compatible
with the PettingZoo env (Step 5), trains a behavioural-cloning policy
that matches the architecture of the PPO RLModule's actor (Step 6), and
transfers the cloned weights into a fresh PPOAlgorithm for finetuning.

Parameter sharing makes the multi-agent BC training pleasantly simple:
the SAME policy network is used by all N BESS, so the dataset pools
expert (obs, action) pairs from all agents in all scenarios into a single
training set. This gives a dataset that is N times larger than the
analogous single-BESS dataset (Lorenzo's earlier Block 6 work), which we
hope makes the BC policy more robust against the washout pattern that
plagued the naive BC->PPO finetune in the single-BESS paper.

The washout-mitigation tricks identified in the single-BESS work are
applied verbatim here for the finetune step: entropy_coeff=0 (no entropy
bonus to erode the BC pattern) and lr=0.1x (slower advantage updates so
the critic has time to catch up to the actor without immediately
overwriting it).
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Project imports
from milp_single import BatteryParameters
from milp_fleet import (MultiBatteryParameters,
                                   MultiBESSMILPOptimizer)
from markets import FlexibilityService, ServiceType
from marl_env import (MultiBESSEnv, N_ACTION_BINS, ACTION_AXES, OBS_DIM)

warnings.simplefilter("ignore")


# ============================================================================
# BC policy network: architecture mirrors PPO RLModule's actor side
# ============================================================================
# RLlib DefaultPPOTorchRLModule structure (inspected at build time):
#   encoder.actor_encoder.net.mlp.0: Linear(OBS_DIM, 512)
#   tanh
#   encoder.actor_encoder.net.mlp.2: Linear(512, 256)
#   tanh
#   pi.net.mlp.0: Linear(256, N_LOGITS) where N_LOGITS = n_axes * N_ACTION_BINS bins
#                 (= 77 for the directional 7-axis fleet, 55 for 5-axis)
#
# We match this exactly so weight transfer is a tensor-by-tensor copy.
# CRITICAL: ENCODER_HIDDEN here MUST equal fcnet_hiddens passed to the PPO
# (marl_ppo.train_ppo_policy). If they differ, the encoder tensors are
# silently skipped on transfer and the warm-start degrades (verify_transfer
# << 1.0). Both are set to (512, 256).

ENCODER_HIDDEN = (512, 256)
N_LOGITS = ACTION_AXES * N_ACTION_BINS  # 55


class BCPolicyNet(nn.Module):
    """PyTorch BC policy network with the same architecture as PPO's actor.

    Forward returns logits of shape (batch, n_axes, N_ACTION_BINS). At training time we
    apply per-axis cross-entropy against the expert action. At inference
    time we take argmax per axis (deterministic) or sample.
    """

    def __init__(self, obs_dim: int = OBS_DIM,
                 hidden: Tuple[int, ...] = ENCODER_HIDDEN,
                 n_logits: int = N_LOGITS,
                 n_axes: int = ACTION_AXES):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.n_axes = int(n_axes)
        # n_logits is forced to n_axes * N_ACTION_BINS so it always matches
        self.n_logits = self.n_axes * N_ACTION_BINS
        # Encoder (matches encoder.actor_encoder.net.mlp.0 and .2)
        self.encoder_layer1 = nn.Linear(obs_dim, hidden[0])
        self.encoder_layer2 = nn.Linear(hidden[0], hidden[1])
        # Pi head (matches pi.net.mlp.0)
        self.pi_head = nn.Linear(hidden[1], self.n_logits)
        self.act = nn.Tanh()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.act(self.encoder_layer1(obs))
        x = self.act(self.encoder_layer2(x))
        logits = self.pi_head(x)
        # Reshape to per-axis logits (batch, n_axes, N_ACTION_BINS)
        return logits.view(-1, self.n_axes, N_ACTION_BINS)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Single-observation prediction. Returns int64 action of shape (n_axes,)."""
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(t)[0]  # (n_axes, N_ACTION_BINS)
            if deterministic:
                return logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
            probs = torch.softmax(logits, dim=-1)
            return torch.distributions.Categorical(probs=probs).sample().cpu().numpy().astype(np.int64)


# ============================================================================
# Expert demo generation
# ============================================================================

def _discretise_milp_action(P_ch: float, P_dis: float,
                              R_fcr: float, R_afrr: float, R_mfrr: float,
                              P_max: float) -> np.ndarray:
    """Convert one (BESS, hour) MILP solution into a 5-axis discrete action."""
    if P_max <= 0:
        return np.zeros(ACTION_AXES, dtype=np.int64)
    def bin_(x: float) -> int:
        frac = max(0.0, min(1.0, x / P_max))
        return int(round(frac * (N_ACTION_BINS - 1)))
    return np.array([bin_(P_ch), bin_(P_dis), bin_(R_fcr),
                      bin_(R_afrr), bin_(R_mfrr)], dtype=np.int64)


def _discretise_milp_action_directional(
    P_ch: float, P_dis: float, R_fcr: float,
    R_afrr_up: float, R_afrr_dn: float,
    R_mfrr_up: float, R_mfrr_dn: float,
    P_max: float,
) -> np.ndarray:
    """Convert one (BESS, hour) MILP directional solution to a 7-axis action.

    Axes: [charge, discharge, FCR, aFRR_up, aFRR_dn, mFRR_up, mFRR_dn].
    """
    n_axes = 7
    if P_max <= 0:
        return np.zeros(n_axes, dtype=np.int64)
    def bin_(x: float) -> int:
        frac = max(0.0, min(1.0, x / P_max))
        return int(round(frac * (N_ACTION_BINS - 1)))
    return np.array([
        bin_(P_ch), bin_(P_dis), bin_(R_fcr),
        bin_(R_afrr_up), bin_(R_afrr_dn),
        bin_(R_mfrr_up), bin_(R_mfrr_dn),
    ], dtype=np.int64)


def generate_expert_demos(
    fleet: MultiBatteryParameters,
    n_scenarios: int = 20,
    episode_hours: int = 24,
    use_nonlinear_degradation: bool = False,
    fce_cumulative_initial: Optional[List[float]] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the expert dataset by rolling MILP-optimal actions on the env.

    Per scenario, we draw a fresh price/services pattern, solve the MILP,
    convert each (BESS, hour) optimal solution to a discrete action, then
    REPLAY those actions in the env to recover the observations that the
    BC policy will need to learn from. This guarantees the observations
    are exactly what the agent would see at deployment.

    Returns
    -------
    obs : np.ndarray of shape (M, OBS_DIM), M = n_scenarios * N * episode_hours
    actions : np.ndarray of shape (M, 5)
    """
    rng = np.random.default_rng(seed)
    N = fleet.n_batteries
    all_obs = []
    all_actions = []

    for s in range(n_scenarios):
        # Sample a scenario: prices with random base + amplitude, services with
        # default catalog (could be extended to randomise services too).
        base = rng.uniform(60.0, 100.0)
        swing = rng.uniform(20.0, 60.0)
        phase = rng.uniform(0, 24)
        prices = [base + swing * np.sin(2 * np.pi * (t - phase) / 24)
                  for t in range(episode_hours)]
        services = MultiBESSEnv._default_services(episode_hours)

        # Solve MILP for this scenario.
        # STRADA B: build the MILP in DIRECTIONAL (5-service) mode with
        # sustain hours aligned to the env (marl_env._decode_clip_actions):
        #   FCR  4h  (Terna FCR Cooperation; env uses fcr_sustain=4.0)
        #   aFRR 1h, mFRR 2h (env defaults).
        # Previously this constructed the MILP with directional_services=False
        # (3 aggregate services) and the default fcr_sustain_hours=0.25, so the
        # expert planned in a more permissive world than the env it is later
        # evaluated in; the env then clipped its FCR/flex reservations, which
        # is the main driver of the MILP continuous->discrete collapse and of
        # the BC inheriting sub-optimal 3-service actions.
        opt = MultiBESSMILPOptimizer(
            fleet, None,
            use_nonlinear_degradation=use_nonlinear_degradation,
            fce_cumulative_initial=fce_cumulative_initial,
            directional_services=True,
            fcr_sustain_hours=4.0,
            afrr_sustain_hours=1.0,
            mfrr_sustain_hours=2.0,
        )
        result = opt.optimize(episode_hours, prices, services)
        if result.solver_status != "Optimal":
            continue

        # Convert per-(BESS, hour) MILP decisions into a sequence of actions
        action_sequence: List[Dict[str, np.ndarray]] = []
        for t in range(episode_hours):
            actions_t = {}
            for i in range(N):
                P_max = fleet.batteries[i].max_power_mw
                P_ch = opt.variables['P_charge'][(i, t)].varValue
                P_dis = opt.variables['P_discharge'][(i, t)].varValue
                # STRADA B: read the 5 DIRECTIONAL reservation variables that
                # exist when the MILP is built with directional_services=True
                # (see _create_variables: _service_keys = ['fcr','afrr_up',
                # 'afrr_dn','mfrr_up','mfrr_dn']). The old code read the 3
                # non-directional keys (R_afrr/R_mfrr), which only exist in
                # legacy mode and discard the up/dn split the env expects.
                R_fcr = opt.variables['R_fcr'][(i, t)].varValue
                R_afrr_up = opt.variables['R_afrr_up'][(i, t)].varValue
                R_afrr_dn = opt.variables['R_afrr_dn'][(i, t)].varValue
                R_mfrr_up = opt.variables['R_mfrr_up'][(i, t)].varValue
                R_mfrr_dn = opt.variables['R_mfrr_dn'][(i, t)].varValue
                actions_t[f"bess_{i}"] = _discretise_milp_action_directional(
                    P_ch, P_dis, R_fcr,
                    R_afrr_up, R_afrr_dn, R_mfrr_up, R_mfrr_dn, P_max,
                )
            action_sequence.append(actions_t)

        # Replay on env to gather observations at the same states
        env = MultiBESSEnv(
            fleet, episode_hours=episode_hours,
            prices=prices, services=services,
            use_nonlinear_degradation=use_nonlinear_degradation,
            fce_cumulative_initial=fce_cumulative_initial,
            seed=int(rng.integers(0, 1_000_000)),
        )
        obs, _ = env.reset()
        for t in range(episode_hours):
            for agent_id in obs:
                all_obs.append(obs[agent_id].copy())
                all_actions.append(action_sequence[t][agent_id].copy())
            obs, _, _, _, _ = env.step(action_sequence[t])

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.int64)
    return obs_arr, act_arr


# ============================================================================
# BC training
# ============================================================================

class _BCDataset(Dataset):
    def __init__(self, obs: np.ndarray, actions: np.ndarray):
        self.obs = obs.astype(np.float32)
        self.actions = actions.astype(np.int64)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


def train_bc(
    obs: np.ndarray,
    actions: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    val_split: float = 0.1,
    seed: int = 0,
    n_axes: int = ACTION_AXES,
) -> Tuple[BCPolicyNet, List[Dict[str, float]]]:
    """Train a BC policy on the expert (obs, action) dataset.

    Returns the trained net plus a per-epoch history of train/val losses
    and per-axis accuracy.
    """
    rng = np.random.default_rng(seed)
    M = len(obs)
    perm = rng.permutation(M)
    n_val = int(M * val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = _BCDataset(obs[train_idx], actions[train_idx])
    val_ds = _BCDataset(obs[val_idx], actions[val_idx]) if n_val > 0 else None

    torch.manual_seed(seed)
    # Infer obs_dim from the data so the BC net matches whatever observation
    # schema the demos use (26 normal, 410 with full_foresight). Falls back to
    # the module default if shape is unavailable.
    inferred_obs_dim = int(obs.shape[1]) if getattr(obs, "ndim", 1) == 2 else OBS_DIM
    net = BCPolicyNet(n_axes=n_axes, obs_dim=inferred_obs_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                  if val_ds is not None else None)

    history = []
    # ---- Ordinal-aware loss --------------------------------------------------
    # The action bins (0..10 = 0..100% of P_max) are ORDERED, but plain
    # cross-entropy treats them as unordered categories: predicting bin 6 when
    # the target is 5 is penalised exactly as much as predicting bin 0. On the
    # axes the MILP modulates continuously (discharge, aFRR_dn) this caps BC
    # accuracy at ~79%. We replace CE with cross-entropy against Gaussian soft
    # labels centred on the true bin: near misses are penalised less than far
    # ones, so the net learns the ordinal structure. sigma controls smoothing
    # (sigma->0 recovers plain CE). Axes that are near-binary (0 or full) are
    # unaffected; the gain concentrates on the modulated axes.
    def _ordinal_ce(logits_k, target_k, sigma):
        if sigma <= 0:
            return nn.functional.cross_entropy(logits_k, target_k)
        n_bins = logits_k.shape[-1]
        bins = torch.arange(n_bins, device=logits_k.device,
                            dtype=torch.float32).view(1, -1)
        t = target_k.view(-1, 1).float()
        w = torch.exp(-((bins - t) ** 2) / (2.0 * sigma * sigma))
        w = w / w.sum(dim=-1, keepdim=True)
        logp = nn.functional.log_softmax(logits_k, dim=-1)
        return -(w * logp).sum(dim=-1).mean()

    bc_loss_sigma = float(globals().get("BC_LOSS_SIGMA", 1.0))

    for epoch in range(n_epochs):
        net.train()
        total_train_loss = 0.0
        n_train_samples = 0
        for obs_b, act_b in train_loader:
            opt.zero_grad()
            logits = net(obs_b)  # (B, n_axes, 11)
            loss = 0.0
            for k in range(n_axes):
                loss = loss + _ordinal_ce(
                    logits[:, k, :], act_b[:, k], bc_loss_sigma
                )
            loss = loss / n_axes
            loss.backward()
            opt.step()
            total_train_loss += float(loss.item()) * obs_b.shape[0]
            n_train_samples += obs_b.shape[0]
        train_loss = total_train_loss / max(n_train_samples, 1)

        val_loss = None
        val_acc = None
        if val_loader is not None:
            net.eval()
            total_val_loss = 0.0
            n_val_samples = 0
            correct_per_axis = np.zeros(n_axes)
            with torch.no_grad():
                for obs_b, act_b in val_loader:
                    logits = net(obs_b)
                    loss = 0.0
                    for k in range(n_axes):
                        loss = loss + _ordinal_ce(
                            logits[:, k, :], act_b[:, k], bc_loss_sigma
                        )
                    loss = loss / n_axes
                    total_val_loss += float(loss.item()) * obs_b.shape[0]
                    n_val_samples += obs_b.shape[0]
                    pred = logits.argmax(dim=-1)
                    for k in range(n_axes):
                        correct_per_axis[k] += (pred[:, k] == act_b[:, k]).sum().item()
            val_loss = total_val_loss / max(n_val_samples, 1)
            val_acc = correct_per_axis / max(n_val_samples, 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_acc_per_axis": val_acc.tolist() if val_acc is not None else None,
        })
    return net, history


# ============================================================================
# Weight transfer: BC -> RLlib RLModule
# ============================================================================

def transfer_bc_weights_to_algo(bc_net: BCPolicyNet, algo, policy_id: str) -> None:
    """Copy BC policy network weights into the RLlib RLModule.

    Maps:
      bc_net.encoder_layer1 -> module.encoder.actor_encoder.net.mlp.0
      bc_net.encoder_layer2 -> module.encoder.actor_encoder.net.mlp.2
      bc_net.pi_head        -> module.pi.net.mlp.0

    Value-function side is left at its random initialisation. The
    washout-mitigation tricks (lr=0.1x, entropy=0) compensate for the
    untrained critic during the first few finetuning iterations.
    """
    module = algo.get_module(policy_id)
    state = module.get_state()
    # Build a new state dict matching the existing structure
    new_state = {}
    for k, v in state.items():
        new_state[k] = v.clone() if hasattr(v, "clone") else v

    # Convert BC weights to the same dtype/device as the module
    target_keys = {
        "encoder.actor_encoder.net.mlp.0.weight": bc_net.encoder_layer1.weight.data,
        "encoder.actor_encoder.net.mlp.0.bias":   bc_net.encoder_layer1.bias.data,
        "encoder.actor_encoder.net.mlp.2.weight": bc_net.encoder_layer2.weight.data,
        "encoder.actor_encoder.net.mlp.2.bias":   bc_net.encoder_layer2.bias.data,
        "pi.net.mlp.0.weight":                    bc_net.pi_head.weight.data,
        "pi.net.mlp.0.bias":                      bc_net.pi_head.bias.data,
    }
    transferred = 0
    skipped = []
    for k, v in target_keys.items():
        if k in new_state:
            target = new_state[k]
            # The RLModule's get_state may return numpy arrays OR torch tensors,
            # depending on the RLlib version. Match the target type.
            v_np = v.detach().cpu().numpy()
            if hasattr(target, "shape"):
                if tuple(target.shape) != tuple(v_np.shape):
                    skipped.append((k, tuple(target.shape), tuple(v_np.shape)))
                    continue
            if isinstance(target, np.ndarray):
                new_state[k] = v_np.astype(target.dtype, copy=True)
            elif isinstance(target, torch.Tensor):
                new_state[k] = torch.from_numpy(v_np).to(
                    dtype=target.dtype, device=target.device
                )
            else:
                # Fallback: assume torch-compatible
                new_state[k] = torch.from_numpy(v_np)
            transferred += 1
        else:
            skipped.append((k, "missing", tuple(v.shape)))

    module.set_state(new_state)
    if skipped:
        print(f"  [warn] {len(skipped)} BC tensor(s) not transferred:")
        for k, exp, got in skipped:
            print(f"    {k}: target {exp}, BC {got}")
    return {"transferred": transferred, "skipped": skipped}


# ============================================================================
# Verification: post-transfer the RLModule must predict BC's actions
# ============================================================================

def verify_transfer(bc_net: BCPolicyNet, algo, policy_id: str,
                    obs_samples: np.ndarray, atol: float = 1e-4) -> Dict[str, Any]:
    """For each obs sample, check that RLModule argmax matches BC argmax.

    Note: this is an EXACT-MATCH test on argmax actions, which is a stronger
    check than just comparing logits (logits may differ by a constant if
    there are unaccounted bias terms in the RLModule structure).
    """
    module = algo.get_module(policy_id)
    # Infer n_axes from the BC net (it may be 5 legacy or 7 directional)
    n_axes = getattr(bc_net, "n_axes", ACTION_AXES)
    matches_per_axis = np.zeros(n_axes, dtype=int)
    n = len(obs_samples)
    for o in obs_samples:
        t = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        bc_action = bc_net.predict(o, deterministic=True)
        with torch.no_grad():
            rl_out = module.forward_inference({"obs": t})
        if "actions" in rl_out:
            rl_action = rl_out["actions"][0].cpu().numpy()
        else:
            logits = rl_out["action_dist_inputs"][0].cpu().numpy()
            bins_per_axis = N_ACTION_BINS
            rl_action = np.array([
                int(np.argmax(logits[k * bins_per_axis:(k + 1) * bins_per_axis]))
                for k in range(n_axes)
            ], dtype=np.int64)
        for k in range(n_axes):
            if rl_action[k] == bc_action[k]:
                matches_per_axis[k] += 1
    return {
        "n_samples": n,
        "match_rate_per_axis": (matches_per_axis / max(n, 1)).tolist(),
        "overall_match_rate": float(matches_per_axis.sum() / max(n * n_axes, 1)),
    }


# ============================================================================
# Washout-mitigated finetuning
# ============================================================================

def build_finetune_config(
    base_config_fn,
    base_lr: float = 3e-4,
    base_entropy_coeff: float = 0.001,
    lr_factor_during_finetune: float = 0.1,
    finetune_entropy_coeff: float = 0.0,
    **base_kwargs,
):
    """Build a PPOConfig with the washout-mitigation tricks applied.

    Parameters
    ----------
    base_config_fn : callable
        Function that builds the base config (typically build_default_config).
    base_lr : float
        Reference learning rate (the BC finetune uses lr_factor * base_lr).
    base_entropy_coeff : float
        Reference entropy coeff (overridden by finetune_entropy_coeff).
    lr_factor_during_finetune : float
        Multiplier applied to base_lr during finetuning. Default 0.1 (10x
        slower than fresh training), matching the single-BESS recipe.
    finetune_entropy_coeff : float
        Entropy bonus during finetuning. Default 0.0 (no entropy bonus,
        so BC pattern is not eroded by exploration).
    """
    return base_config_fn(
        lr=base_lr * lr_factor_during_finetune,
        entropy_coeff=finetune_entropy_coeff,
        **base_kwargs,
    )