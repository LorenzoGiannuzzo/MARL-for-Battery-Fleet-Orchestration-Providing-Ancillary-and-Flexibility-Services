"""
Block 6: Behavioural Cloning (BC) warm-start for the PPO BESS agent.

The core scientific contribution of the paper. Vanilla PPO leaves a large
optimality gap relative to the MILP (~58% on real Italian data, almost
entirely explained by PPO under-reserving the dominant aFRR service). This
module teaches the PPO policy network to imitate the MILP expert before any
reinforcement learning happens, so that PPO fine-tuning starts from a
near-optimal policy instead of a random one.

Three stages, implemented here:

  1. collect_milp_demonstrations(...)
     Roll the MILP day-by-day over a window of (real or synthetic) prices,
     and for each hour record the pair (observation_as_the_env_sees_it,
     milp_action_quantised_to_the_env_grid). The observation is built by an
     actual ExtendedBatteryTradingEnv stepped along the MILP's own SOC
     trajectory, so the demonstration observations are exactly the
     distribution the PPO will see at deployment. The MILP's continuous
     reservation levels (MW) are quantised to the nearest discrete level the
     MultiDiscrete([21,4,4,4]) action space supports.

  2. behavioural_cloning(...)
     Supervised training of the SB3 PPO policy network. For a MultiDiscrete
     action space, SB3 uses a MultiCategoricalDistribution: four independent
     categorical heads (arbitrage, FCR, aFRR, mFRR). The BC loss is the sum
     of the four per-head cross-entropy terms. We optimise the policy's own
     parameters in place so the resulting model can be handed straight to
     PPO.learn() for fine-tuning.

  3. (fine-tuning is done by the caller via the normal PPO.learn path; this
     module only produces the warm-started model.)

The mapping between MILP continuous output and the env grid is the delicate
part and is unit-tested in tests_block6.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ============================================================================
# MILP action -> env MultiDiscrete grid quantisation
# ============================================================================

def _nearest_index(value: float, grid: List[float]) -> int:
    """Index of the grid level closest to `value` (ties -> lower index)."""
    arr = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(arr - value)))


def quantise_milp_action(
    arbitrage_mw: float,
    fcr_mw: float,
    afrr_mw: float,
    mfrr_mw: float,
    arbitrage_values: List[float],
    fcr_percentages: List[float],
    afrr_percentages: List[float],
    mfrr_percentages: List[float],
    max_power_mw: float,
) -> np.ndarray:
    """Map a continuous MILP decision to the env's MultiDiscrete action.

    Parameters
    ----------
    arbitrage_mw : float
        Net MILP arbitrage power for the hour (signed: + charge, - discharge).
        NB: the env convention for arbitrage_values is the same signed axis
        from -P_max (full discharge) to +P_max (full charge).
    fcr_mw, afrr_mw, mfrr_mw : float
        MILP reserved capacity for each service in MW (non-negative).
    arbitrage_values : list of float
        The env's signed arbitrage levels in MW (length 21).
    fcr/afrr/mfrr_percentages : list of float
        The env's reservation fractions per service (e.g. [0, .5, .75, .9]).
    max_power_mw : float
        Battery nominal power, used to convert reserved MW into a fraction.

    Returns
    -------
    np.ndarray of shape (4,), dtype int
        [arb_idx, fcr_idx, afrr_idx, mfrr_idx] indices into the env grids.
    """
    arb_idx = _nearest_index(arbitrage_mw, arbitrage_values)
    fcr_frac = (fcr_mw / max_power_mw) if max_power_mw > 0 else 0.0
    afrr_frac = (afrr_mw / max_power_mw) if max_power_mw > 0 else 0.0
    mfrr_frac = (mfrr_mw / max_power_mw) if max_power_mw > 0 else 0.0
    fcr_idx = _nearest_index(fcr_frac, fcr_percentages)
    afrr_idx = _nearest_index(afrr_frac, afrr_percentages)
    mfrr_idx = _nearest_index(mfrr_frac, mfrr_percentages)
    return np.array([arb_idx, fcr_idx, afrr_idx, mfrr_idx], dtype=np.int64)


# ============================================================================
# Demonstration dataset
# ============================================================================

@dataclass
class DemonstrationDataset:
    """A flat (observations, actions) dataset of MILP expert behaviour."""
    observations: np.ndarray  # shape (N, obs_dim), float32
    actions: np.ndarray       # shape (N, 4), int64

    def __len__(self) -> int:
        return self.observations.shape[0]

    def action_distribution_report(self) -> str:
        """Human-readable per-component action histogram (for diagnostics)."""
        names = ["arbitrage", "FCR", "aFRR", "mFRR"]
        lines = [f"  Demonstration dataset: {len(self)} (obs, action) pairs"]
        for j, name in enumerate(names):
            vals, counts = np.unique(self.actions[:, j], return_counts=True)
            frac = {int(v): f"{100*c/len(self):.1f}%"
                    for v, c in zip(vals, counts)}
            lines.append(f"    {name:9s} idx distribution: {frac}")
        return "\n".join(lines)


def collect_milp_demonstrations(
    run_milp_day_fn,
    make_env_fn,
    battery_params,
    flex_market,
    prices: List[float],
    services: List[List],
    cfg,
    train_error_distribution: str,
    verbose: bool = True,
) -> DemonstrationDataset:
    """Generate (observation, quantised-MILP-action) pairs over a price window.

    The function steps an actual ExtendedBatteryTradingEnv so that the
    recorded observations match exactly what the PPO will receive. At each
    hour the env observation is captured BEFORE stepping; the MILP action for
    that hour (quantised) is the supervised target; then the env is advanced
    using that same quantised action so the SOC trajectory the observations
    are built on is the one the imitated policy would itself produce.

    Parameters
    ----------
    run_milp_day_fn : callable
        main.run_milp_day. Solves a 24h MILP and returns reservations +
        an (optional) arbitrage power trajectory.
    make_env_fn : callable
        main.make_env. Builds an ExtendedBatteryTradingEnv.
    battery_params, flex_market, cfg :
        Passed straight through to run_milp_day / make_env.
    prices : list of float
        The full price window to roll over (e.g. last pretrain_months of the
        training year). Length should be a multiple of 24; trailing partial
        days are dropped.
    services : list of per-hour FlexibilityService lists
        Same length as prices.
    train_error_distribution : str
        Forecast-error distribution used when building the env observation,
        so the demonstration observation distribution matches training.

    Returns
    -------
    DemonstrationDataset
    """
    n_days = len(prices) // 24
    if n_days == 0:
        raise ValueError("Need at least 24 hours of prices to collect demos.")

    # One env spanning the whole window; we read observations from it and
    # advance it hour by hour using the quantised MILP actions.
    env = make_env_fn(prices[: n_days * 24], train_error_distribution, cfg.seed,
                      enable_reward_shaping=False,
                      flex_price_error_pct=cfg.flex_price_error_pct,
                      forecast_calibration_json=cfg.forecast_calibration_json,
                      forecast_regime=cfg.forecast_regime)
    obs, _ = env.reset()

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []

    t0 = time.time()
    for day in range(n_days):
        day_slice = slice(day * 24, (day + 1) * 24)
        daily_prices = prices[day_slice]
        daily_services = services[day_slice]

        # Solve the MILP for this day. forecast_errors=None means the MILP
        # plans on the true prices (perfect-foresight expert); this is the
        # standard choice for an imitation target (the expert is as good as
        # possible, and the student learns its structure).
        milp_res = run_milp_day_fn(
            battery_params, flex_market, daily_prices, daily_services,
            forecast_errors=None,
        )
        reservations = milp_res["flexibility_reservations"]
        # Arbitrage power per hour: power_trajectory is signed
        # (P_discharge - P_charge). The env arbitrage axis is signed with
        # + = charge, so we negate to match (discharge>0 in MILP => the env
        # sees a negative/charge-axis value of opposite sign).
        power_traj = milp_res.get("power_trajectory", [0.0] * 24)

        for h in range(24):
            # Capture the observation the env currently exposes
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())

            # Build the quantised MILP action for this hour.
            # MILP power_trajectory[h] = P_discharge - P_charge (signed).
            # The env arbitrage_values axis runs -P_max..+P_max with
            # + meaning CHARGE, so the env-equivalent arbitrage target is
            # the negative of the MILP signed power.
            milp_arb_env_axis = -float(power_traj[h]) if h < len(power_traj) else 0.0
            fcr_mw = reservations.get("FCR", [0.0] * 24)[h]
            afrr_mw = reservations.get("aFRR", [0.0] * 24)[h]
            mfrr_mw = reservations.get("mFRR", [0.0] * 24)[h]

            action = quantise_milp_action(
                arbitrage_mw=milp_arb_env_axis,
                fcr_mw=fcr_mw, afrr_mw=afrr_mw, mfrr_mw=mfrr_mw,
                arbitrage_values=list(env.arbitrage_values),
                fcr_percentages=list(env.fcr_percentages),
                afrr_percentages=list(env.afrr_percentages),
                mfrr_percentages=list(env.mfrr_percentages),
                max_power_mw=battery_params.max_power_mw,
            )
            act_list.append(action)

            # Advance the env with the quantised MILP action so the next
            # observation is conditioned on the expert-induced SOC.
            obs, _r, done, trunc, _info = env.step(action)
            if done or trunc:
                obs, _ = env.reset()

    elapsed = time.time() - t0
    ds = DemonstrationDataset(
        observations=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(act_list, dtype=np.int64),
    )
    if verbose:
        print(f"  Collected {len(ds)} demonstrations from {n_days} MILP days "
              f"in {elapsed:.1f}s")
        print(ds.action_distribution_report())
    return ds


# ============================================================================
# Behavioural cloning training
# ============================================================================

def behavioural_cloning(
    model,
    dataset: DemonstrationDataset,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    val_fraction: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Supervised pretraining of an SB3 PPO policy on MILP demonstrations.

    The policy network of `model` (an SB3 PPO with a MultiDiscrete action
    space) is trained in place by minimising the sum of four per-component
    cross-entropy losses against the quantised MILP actions. After this call,
    `model` can be passed directly to PPO.learn() for fine-tuning: the warm
    weights are retained.

    Parameters
    ----------
    model : stable_baselines3.PPO
        Must already be constructed on the ExtendedBatteryTradingEnv so its
        observation/action spaces match the dataset.
    dataset : DemonstrationDataset
    n_epochs, batch_size, lr : training hyperparameters.
    val_fraction : float
        Held-out fraction for a validation curve.

    Returns
    -------
    dict with 'train_loss', 'val_loss', 'val_accuracy' lists (per epoch).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for behavioural cloning.")

    rng = np.random.default_rng(seed)
    policy = model.policy
    device = policy.device
    policy.train()

    obs = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    acts = torch.as_tensor(dataset.actions, dtype=torch.long, device=device)

    n = obs.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # The MultiDiscrete nvec gives the per-head class counts.
    nvec = list(model.action_space.nvec)  # e.g. [21, 4, 4, 4]

    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def _per_head_logits(observations: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of logits tensors, one per action component.

        SB3's MultiCategoricalDistribution concatenates the per-head logits
        along the last axis; we extract them and split by nvec. We re-run the
        feature extractor + action net via get_distribution to stay version
        robust.
        """
        dist = policy.get_distribution(observations)
        # dist.distribution is a list of torch.distributions.Categorical,
        # one per action component, for MultiCategoricalDistribution.
        cats = dist.distribution
        return [c.logits for c in cats]

    def _loss_and_acc(observations, targets):
        logits_list = _per_head_logits(observations)
        loss = 0.0
        correct = torch.ones(observations.shape[0], dtype=torch.bool,
                             device=device)
        for j, logits in enumerate(logits_list):
            loss = loss + F.cross_entropy(logits, targets[:, j])
            pred = logits.argmax(dim=-1)
            correct = correct & (pred == targets[:, j])
        return loss, correct.float().mean().item()

    for epoch in range(n_epochs):
        # ---- train ----
        policy.train()
        ep_order = rng.permutation(len(train_idx))
        train_losses = []
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[ep_order[start:start + batch_size]]
            b_obs = obs[batch]
            b_act = acts[batch]
            optimiser.zero_grad()
            loss, _ = _loss_and_acc(b_obs, b_act)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())

        # ---- validate ----
        policy.eval()
        with torch.no_grad():
            val_loss, val_acc = _loss_and_acc(obs[val_idx], acts[val_idx])
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(val_loss.item()))
        history["val_accuracy"].append(float(val_acc))

        if verbose and (epoch % max(1, n_epochs // 10) == 0
                        or epoch == n_epochs - 1):
            print(f"    BC epoch {epoch+1:3d}/{n_epochs}: "
                  f"train_loss={history['train_loss'][-1]:.4f}, "
                  f"val_loss={history['val_loss'][-1]:.4f}, "
                  f"val_acc(all-4-heads)={history['val_accuracy'][-1]:.3f}")

    policy.eval()
    return history
