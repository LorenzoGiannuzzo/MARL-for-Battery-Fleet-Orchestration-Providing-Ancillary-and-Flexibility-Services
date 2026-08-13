"""
End-to-end pipeline for the multi-agent BSP paper.

Step 8 of the multi-agent track. Ties together:

  - Italian market data (real or synthetic) over a contiguous date range,
    from `italian_market_data.py`.
  - Multi-day MILP demonstration generation with ROLLING-FCE state across
    days: each day's MILP inherits the SOC and cumulative throughput from
    the previous day, so the fleet ages realistically across the year.
  - Behavioural cloning training on the pooled expert dataset, using the
    shared-policy BC machinery from `marl_bc.py`.
  - Out-of-sample evaluation on held-out test days, comparing:
      A) MILP optimum (oracle baseline)
      B) BC-trained policy (deployable policy)
      C) Random policy (lower-bound sanity)

The function `run_full_pipeline` is the headline orchestrator. It returns
a dict of summary numbers ready to be tabulated for the paper:

    {
      'milp_train_total_profit_eur':  ...,
      'milp_test_total_profit_eur':   ...,
      'bc_test_mean_profit_eur':      ...,
      'random_test_mean_profit_eur':  ...,
      'bc_gap_to_milp_percent':       ...,
      ...
    }

Designed to scale from a smoke-test setup (5 BESS, 10 days) to a full-year
run (50 BESS, 365 days). Default arguments are at the small end so the
test suite runs fast; the paper run uses the production_config() preset.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from milp_optimizer import BatteryParameters
from marl_milp_continuous import (MultiBatteryParameters,
                                   MultiBESSMILPOptimizer)
from marl_milp_discrete import MultiBESSMILPOptimizerDiscrete
from marl_env import MultiBESSEnv, N_ACTION_BINS, ACTION_AXES, ACTION_AXES_DIRECTIONAL
from marl_bc import (BCPolicyNet, train_bc,
                                    _discretise_milp_action)
from italian_market_data import (MarketWindow, make_synthetic_market_window,
                                   PUN2024Calibration, ServiceCalibration)
from degradation_model import LFPBatteryState
from market_constants import SustainDurations, DEFAULT_SUSTAIN
from fleet_coupling import FleetCoupling
from marl_env import mask_global_block

warnings.simplefilter("ignore")

# Force stdout to be line-buffered so progress prints appear in real-time
# on Windows PyCharm Run console (which otherwise buffers heavily).
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ============================================================================
# Multi-day MILP demo generation with rolling FCE
# ============================================================================

@dataclass
class MultiDayDemoResult:
    """Output of generate_multi_day_expert_demos."""
    obs: np.ndarray            # (M, OBS_DIM)
    actions: np.ndarray        # (M, ACTION_AXES)
    daily_milp_profits: List[float]     # MILP optimizer objective per day
    daily_env_realized_profits: List[float]  # env-realized reward per day under MILP actions
    daily_milp_status: List[str]
    final_soc_per_bess: List[float]
    final_fce_per_bess: List[float]
    final_capacity_lost_per_bess: List[float] = field(default_factory=list)
    total_milp_profit: float = 0.0
    total_env_realized_profit: float = 0.0
    elapsed_seconds: float = 0.0
    trajectory: Optional[Dict[str, Any]] = None  # hourly trajectory when record_trajectory=True


def generate_multi_day_expert_demos(
    fleet: MultiBatteryParameters,
    market_window: MarketWindow,
    use_nonlinear_degradation: bool = False,
    initial_soc: Optional[List[float]] = None,
    initial_fce_cumulative: Optional[List[float]] = None,
    initial_capacity_lost: Optional[List[float]] = None,
    nonlinear_dod_assumed: float = 0.6,
    nonlinear_replacement_cost: float = 200000.0,
    env_seed: int = 0,
    record_trajectory: bool = False,
    directional_services: bool = False,
    enable_commitments: bool = False,
    commitment_lead_time: int = 24,
    penalty_k: float = 1.5,
    full_foresight: bool = False,
    overcommit_penalty: float = 0.0,
    milp_mode: str = "continuous",
    sustain: SustainDurations = DEFAULT_SUSTAIN,
    duration_features: bool = True,
    coupling: FleetCoupling = FleetCoupling.uncoupled(),
    # Kept on ONE observation width across every stage of the pipeline. The
    # demo generator and the clone do not use a critic, but if their
    # observation were narrower than the PPO's, the two would be trained and
    # evaluated on different vectors and every downstream comparison would be
    # measuring the mismatch. The clone simply ignores the tail, at the cost
    # of six input weights.
    centralized_critic: bool = False,
) -> MultiDayDemoResult:
    """Day-by-day MILP demo generation with rolling per-BESS state.

    For each day in the market window:
      1. Solve the MILP for that day with the current SOC and FCE
         carried over from the previous day.
      2. Convert the MILP solution to discrete env actions per BESS,
         per hour.
      3. Roll the env one day forward applying these actions; record
         the (obs, action) pairs from the env (so observations are
         exactly what an agent at deployment would see).
      4. Update SOC, FCE, and capacity_lost from the env state for the
         next day so SOH does not visually reset across daily env
         re-instantiations.

    The non-linear degradation rate inside the MILP is recomputed at each
    day's __init__ from the current FCE, so the MILP becomes more lenient
    on throughput as the fleet ages (matching the env's physics).

    Returns
    -------
    MultiDayDemoResult
    """
    t0 = time.time()
    N = fleet.n_batteries
    n_days = market_window.n_days

    if initial_soc is None:
        initial_soc = [0.5] * N
    if initial_fce_cumulative is None:
        initial_fce_cumulative = [0.0] * N
    if initial_capacity_lost is None:
        initial_capacity_lost = [0.0] * N
    if (len(initial_soc) != N or len(initial_fce_cumulative) != N
            or len(initial_capacity_lost) != N):
        raise ValueError("initial state arrays length mismatch")

    soc = list(initial_soc)
    fce = list(initial_fce_cumulative)
    cap_lost = list(initial_capacity_lost)

    all_obs = []
    all_actions = []
    daily_profits = []
    daily_env_realized = []
    daily_statuses = []

    # Trajectory buffers (only filled if record_trajectory=True)
    n_axes = ACTION_AXES_DIRECTIONAL if directional_services else ACTION_AXES
    traj_actions_mean = []
    traj_soc_mean = []; traj_soc_min = []; traj_soc_max = []
    traj_soh_mean = []; traj_soh_min = []; traj_soh_max = []
    traj_arbitrage = []; traj_flex = []; traj_deg = []; traj_penalty = []
    # Per-service revenue trajectory buffers (filled from env.last_aggregate_info).
    # In directional mode we track 5 services; in legacy mode, 3.
    traj_rev_fcr = []; traj_rev_afrr_up = []; traj_rev_afrr_dn = []
    traj_rev_mfrr_up = []; traj_rev_mfrr_dn = []
    traj_rev_afrr = []; traj_rev_mfrr = []
    traj_bid_fcr_mw = []; traj_bid_afrr_mw = []; traj_bid_mfrr_mw = []
    # Directional bid splits (used for per-service carpet plots)
    traj_bid_afrr_up_mw = []; traj_bid_afrr_dn_mw = []
    traj_bid_mfrr_up_mw = []; traj_bid_mfrr_dn_mw = []
    traj_price = []
    # Per-service energy price per hour (for the spread vs PUN chart)
    traj_price_fcr = []; traj_price_afrr_up = []; traj_price_afrr_dn = []
    traj_price_mfrr_up = []; traj_price_mfrr_dn = []
    traj_price_afrr = []; traj_price_mfrr = []
    traj_action_bin_counts = np.zeros((n_axes, N_ACTION_BINS), dtype=np.int64)

    # Progress print configuration: print one summary line every PROGRESS_EVERY
    # days. For long windows (>50 days) this gives the user visibility on
    # whether CBC is making progress vs hanging on a degenerate day. For
    # short windows the per-day prints would be too noisy, so we skip.
    PROGRESS_EVERY = 10
    enable_progress = (n_days > 50)
    if enable_progress:
        print(f"  [demo] starting MILP demo gen on {n_days} days, "
              f"N={N} BESS, directional={directional_services}", flush=True)

    for day in range(n_days):
        prices_day, services_day = market_window.slice_day(day)

        if enable_progress and day > 0 and day % PROGRESS_EVERY == 0:
            from collections import Counter
            elapsed_so_far = time.time() - t0
            avg_per_day = elapsed_so_far / day
            eta_minutes = avg_per_day * (n_days - day) / 60.0
            status_counts = Counter(daily_statuses)
            n_opt = status_counts.get("Optimal", 0)
            running_profit = sum(daily_profits)
            print(f"  [demo] day {day}/{n_days} | "
                  f"elapsed {elapsed_so_far:.0f}s ({avg_per_day:.1f}s/day) | "
                  f"Optimal: {n_opt}/{day} | "
                  f"running profit: {running_profit:,.0f} EUR | "
                  f"ETA: {eta_minutes:.1f} min", flush=True)

        # SOC clamping (defensive): the env clips SOC to [0,1] but the MILP
        # requires SOC in [soc_min, soc_max] per battery. Carry-over after
        # stochastic activations in directional mode can drift slightly out
        # of MILP bounds. Project to the feasible interior with a small
        # epsilon to leave the bound constraints inactive at t=0.
        soc_clamped = []
        for i in range(N):
            bp = fleet.batteries[i]
            lo = bp.soc_min + 1e-4
            hi = bp.soc_max - 1e-4
            soc_clamped.append(float(np.clip(soc[i], lo, hi)))

        # Build the day's MILP with current rolling state.
        # Pass sustain hours explicitly so the MILP plans in the SAME world the
        # env evaluates in. Both now read the SAME SustainDurations object
        # (market_constants), replacing the previous arrangement where the MILP
        # was overridden to 4.0 h here to chase a hard-coded 4.0 h in the env
        # clip, while the MILP's own default was 0.25 h and the env's
        # availability check used 0.5 h. See market_constants for the SO GL
        # Art. 156(10) basis of the 15-minute default.
        #
        # milp_mode selects the baseline:
        #   "continuous" -> MultiBESSMILPOptimizer: continuous power, the
        #                   solution is projected onto the 40-bin grid AFTER
        #                   solving (the projection block below). This is the
        #                   original oracle (paper baseline 1).
        #   "discrete"   -> MultiBESSMILPOptimizerDiscrete: power axes are
        #                   constrained to the exact PPO 40-bin grid as native
        #                   MILP one-hot variables, feasible-by-construction.
        #                   No post-hoc projection handicap (paper baseline 2).
        if milp_mode == "discrete":
            opt = MultiBESSMILPOptimizerDiscrete(
                fleet, None,
                use_nonlinear_degradation=use_nonlinear_degradation,
                nonlinear_dod_assumed=nonlinear_dod_assumed,
                nonlinear_replacement_cost=nonlinear_replacement_cost,
                fce_cumulative_initial=fce,
                directional_services=directional_services,
                **sustain.as_milp_kwargs(),
                coupling=coupling,
                enable_commitments=enable_commitments,
                penalty_k=penalty_k,
                n_action_bins=N_ACTION_BINS,
            )
        elif milp_mode == "continuous":
            opt = MultiBESSMILPOptimizer(
                fleet, None,
                use_nonlinear_degradation=use_nonlinear_degradation,
                nonlinear_dod_assumed=nonlinear_dod_assumed,
                nonlinear_replacement_cost=nonlinear_replacement_cost,
                fce_cumulative_initial=fce,
                directional_services=directional_services,
                **sustain.as_milp_kwargs(),
                coupling=coupling,
                enable_commitments=enable_commitments,
                penalty_k=penalty_k,
            )
        else:
            raise ValueError(
                f"milp_mode must be 'continuous' or 'discrete', got {milp_mode!r}"
            )
        opt.current_soc = soc_clamped  # carry SOC over, clamped to MILP bounds
        result = opt.optimize(24, prices_day, services_day)
        daily_statuses.append(result.solver_status)
        if result.solver_status != "Optimal":
            # MILP failed for this day. Fall back to a ZERO-ACTION rollout
            # so SOC carry-over progresses naturally instead of getting
            # stuck for the rest of the window. Also still record obs+actions
            # of the env rollout so BC has SOMETHING to learn (action=0 is a
            # weak signal but won't hurt BC if rare; harmful only if many
            # days fall back). We log a warning.
            if day < 5 or day % 50 == 0:
                print(f"  [warn] day {day}: MILP status={result.solver_status}, "
                      f"using zero-action fallback")
            zero_action_sequence = []
            for t in range(24):
                actions_t = {}
                for i in range(N):
                    n_axes = (7 if directional_services else 5)
                    actions_t[f"bess_{i}"] = np.zeros(n_axes, dtype=np.int64)
                zero_action_sequence.append(actions_t)

            env = MultiBESSEnv(
                fleet,
                episode_hours=24,
                prices=prices_day,
                services=services_day,
                use_nonlinear_degradation=use_nonlinear_degradation,
                fce_cumulative_initial=fce,
                capacity_lost_initial=cap_lost,
                soc_init=0.5,
                seed=env_seed + day,
                directional_services=directional_services,
                sustain_hours=sustain,
                duration_features=duration_features,
                centralized_critic=centralized_critic,
                coupling=coupling,
            )
            if enable_commitments:
                env.configure_commitments(True, commitment_lead_time, penalty_k)
            if full_foresight:
                env.configure_full_foresight(True)
            if overcommit_penalty:
                env.configure_overcommit_penalty(overcommit_penalty)
                if full_foresight:
                    env.configure_full_foresight(True)
                if overcommit_penalty:
                    env.configure_overcommit_penalty(overcommit_penalty)
            env.reset(seed=env_seed + day)
            env._socs = np.array(soc_clamped, dtype=np.float64)
            obs = {a: env._build_observation(a) for a in env.agents}
            day_env_profit = 0.0
            for t in range(24):
                actions_t = zero_action_sequence[t]
                # Do NOT append obs/actions: zero-action is not a valid
                # expert demo. We just roll the env to update SOC/FCE.
                obs, rewards, _, _, _ = env.step(actions_t)
                day_env_profit += sum(rewards.values())
            for i in range(N):
                soc[i] = float(env._socs[i])
                fce[i] = float(env._lfp_states[i].fce_cumulative)
                cap_lost[i] = float(
                    env._lfp_states[i].capacity_lost_fraction)
            daily_profits.append(0.0)
            daily_env_realized.append(day_env_profit)
            continue
        daily_profits.append(result.net_profit)

        # Discretise the MILP solution into per-(BESS, hour) actions.
        # For milp_mode="continuous" this projects the continuous solution onto
        # the 40-bin grid. For milp_mode="discrete" the solution already lies
        # exactly on the grid (the optimiser pinned every power axis to a bin),
        # so bin_(x) = round((x/P_max)*39) recovers the same bin and this step
        # is the IDENTITY: no information is lost, no extra handicap is applied.
        action_sequence: List[Dict[str, np.ndarray]] = []
        from action_projection import _discretise_milp_action_directional
        for t in range(24):
            actions_t = {}
            for i in range(N):
                P_max = fleet.batteries[i].max_power_mw
                P_ch = opt.variables['P_charge'][(i, t)].varValue
                P_dis = opt.variables['P_discharge'][(i, t)].varValue
                R_fcr = opt.variables['R_fcr'][(i, t)].varValue
                if directional_services:
                    R_afrr_up = opt.variables['R_afrr_up'][(i, t)].varValue
                    R_afrr_dn = opt.variables['R_afrr_dn'][(i, t)].varValue
                    R_mfrr_up = opt.variables['R_mfrr_up'][(i, t)].varValue
                    R_mfrr_dn = opt.variables['R_mfrr_dn'][(i, t)].varValue
                    actions_t[f"bess_{i}"] = _discretise_milp_action_directional(
                        P_ch, P_dis, R_fcr,
                        R_afrr_up, R_afrr_dn, R_mfrr_up, R_mfrr_dn, P_max,
                    )
                else:
                    R_afrr = opt.variables['R_afrr'][(i, t)].varValue
                    R_mfrr = opt.variables['R_mfrr'][(i, t)].varValue
                    actions_t[f"bess_{i}"] = _discretise_milp_action(
                        P_ch, P_dis, R_fcr, R_afrr, R_mfrr, P_max,
                    )
            action_sequence.append(actions_t)

        # Roll the env one day forward, recording the env-REALISED profit
        env = MultiBESSEnv(
            fleet,
            episode_hours=24,
            prices=prices_day,
            services=services_day,
            use_nonlinear_degradation=use_nonlinear_degradation,
            fce_cumulative_initial=fce,
            capacity_lost_initial=cap_lost,
            soc_init=0.5,
            seed=env_seed + day,
            directional_services=directional_services,
            sustain_hours=sustain,
            duration_features=duration_features,
            centralized_critic=centralized_critic,
            coupling=coupling,
        )
        if enable_commitments:
            env.configure_commitments(True, commitment_lead_time, penalty_k)
            if full_foresight:
                env.configure_full_foresight(True)
            if overcommit_penalty:
                env.configure_overcommit_penalty(overcommit_penalty)
        env.reset(seed=env_seed + day)
        env._socs = np.array(soc, dtype=np.float64)
        obs = {a: env._build_observation(a) for a in env.agents}

        day_env_profit = 0.0
        for t in range(24):
            actions_t = action_sequence[t]

            if record_trajectory:
                act_stack = np.array([actions_t[f"bess_{i}"] for i in range(N)],
                                      dtype=np.int64)
                act_frac = act_stack.astype(np.float32) / (N_ACTION_BINS - 1)
                traj_actions_mean.append(act_frac.mean(axis=0))
                for k in range(n_axes):
                    traj_action_bin_counts[k] += np.bincount(
                        act_stack[:, k], minlength=N_ACTION_BINS)
                bid_fcr = 0.0; bid_afrr = 0.0; bid_mfrr = 0.0
                bid_aup = 0.0; bid_adn = 0.0; bid_mup = 0.0; bid_mdn = 0.0
                for i in range(N):
                    P_max = fleet.batteries[i].max_power_mw
                    frac = actions_t[f"bess_{i}"].astype(np.float32) / (N_ACTION_BINS - 1)
                    bid_fcr += frac[2] * P_max
                    if directional_services:
                        # 7-axis: split bid per direction
                        bid_aup += frac[3] * P_max
                        bid_adn += frac[4] * P_max
                        bid_mup += frac[5] * P_max
                        bid_mdn += frac[6] * P_max
                        # aggregated for back-compat (sum of up+dn)
                        bid_afrr = bid_aup + bid_adn
                        bid_mfrr = bid_mup + bid_mdn
                    else:
                        bid_afrr += frac[3] * P_max
                        bid_mfrr += frac[4] * P_max
                traj_bid_fcr_mw.append(bid_fcr)
                traj_bid_afrr_mw.append(bid_afrr)
                traj_bid_mfrr_mw.append(bid_mfrr)
                traj_bid_afrr_up_mw.append(bid_aup)
                traj_bid_afrr_dn_mw.append(bid_adn)
                traj_bid_mfrr_up_mw.append(bid_mup)
                traj_bid_mfrr_dn_mw.append(bid_mdn)
                traj_price.append(prices_day[t])
                # Per-service energy prices for this hour
                from flexibility_market import ServiceType as _ST
                price_lookup = {s.service_type: s.energy_price
                                for s in services_day[t]}
                traj_price_fcr.append(float(price_lookup.get(_ST.FCR, 0.0)))
                if directional_services:
                    traj_price_afrr_up.append(float(price_lookup.get(_ST.AFRR_UP, 0.0)))
                    traj_price_afrr_dn.append(float(price_lookup.get(_ST.AFRR_DN, 0.0)))
                    traj_price_mfrr_up.append(float(price_lookup.get(_ST.MFRR_UP, 0.0)))
                    traj_price_mfrr_dn.append(float(price_lookup.get(_ST.MFRR_DN, 0.0)))
                else:
                    traj_price_afrr.append(float(price_lookup.get(_ST.AFRR, 0.0)))
                    traj_price_mfrr.append(float(price_lookup.get(_ST.MFRR, 0.0)))
                socs_arr = np.array(env._socs, dtype=np.float64)
                traj_soc_mean.append(float(socs_arr.mean()))
                traj_soc_min.append(float(socs_arr.min()))
                traj_soc_max.append(float(socs_arr.max()))

            for agent_id in obs:
                # Record the ACTOR's view. The clone is a decentralised
                # policy: it must not learn from the fleet-level block that a
                # deployed unit cannot see, and the PPO actor it warm-starts
                # masks that block anyway. Training the clone on the full
                # vector dropped verify_transfer from 1.000 to 0.604, meaning
                # the warm start no longer began where the clone was.
                all_obs.append(
                    mask_global_block(obs[agent_id], centralized_critic))
                all_actions.append(actions_t[agent_id].copy())
            obs, rewards, _, _, infos = env.step(actions_t)
            day_env_profit += sum(rewards.values())

            if record_trajectory:
                hour_arb = sum(infos[a]['arbitrage'] for a in infos if isinstance(infos[a], dict))
                hour_flex = sum(infos[a]['flex_revenue'] for a in infos if isinstance(infos[a], dict))
                hour_deg = sum(infos[a]['degradation'] for a in infos if isinstance(infos[a], dict))
                hour_pen = sum(infos[a].get('non_delivery_penalty', 0.0) + infos[a].get('overcommit_penalty', 0.0)
                               for a in infos if isinstance(infos[a], dict))
                traj_arbitrage.append(hour_arb)
                traj_flex.append(hour_flex)
                traj_deg.append(hour_deg)
                traj_penalty.append(hour_pen)
                # Per-service revenue from env.last_aggregate_info
                agg = getattr(env, 'last_aggregate_info', {}) or {}
                traj_rev_fcr.append(float(agg.get('rev_fcr', 0.0)))
                if directional_services:
                    traj_rev_afrr_up.append(float(agg.get('rev_afrr_up', 0.0)))
                    traj_rev_afrr_dn.append(float(agg.get('rev_afrr_dn', 0.0)))
                    traj_rev_mfrr_up.append(float(agg.get('rev_mfrr_up', 0.0)))
                    traj_rev_mfrr_dn.append(float(agg.get('rev_mfrr_dn', 0.0)))
                else:
                    traj_rev_afrr.append(float(agg.get('rev_afrr', 0.0)))
                    traj_rev_mfrr.append(float(agg.get('rev_mfrr', 0.0)))
                # SOH snapshot (read post-step from env's LFP state per agent)
                soh_arr = np.array([s.soh for s in env._lfp_states], dtype=np.float64)
                traj_soh_mean.append(float(soh_arr.mean()))
                traj_soh_min.append(float(soh_arr.min()))
                traj_soh_max.append(float(soh_arr.max()))

        daily_env_realized.append(day_env_profit)

        # Update rolling state from env after the day
        for i in range(N):
            soc[i] = float(env._socs[i])
            fce[i] = float(env._lfp_states[i].fce_cumulative)
            cap_lost[i] = float(env._lfp_states[i].capacity_lost_fraction)

    trajectory = None
    if record_trajectory:
        trajectory = {
            "actions_mean_per_hour": np.array(traj_actions_mean, dtype=np.float32),
            "action_bin_counts": traj_action_bin_counts,
            "soc_mean": np.array(traj_soc_mean, dtype=np.float32),
            "soc_min": np.array(traj_soc_min, dtype=np.float32),
            "soc_max": np.array(traj_soc_max, dtype=np.float32),
            "soh_mean": np.array(traj_soh_mean, dtype=np.float32),
            "soh_min": np.array(traj_soh_min, dtype=np.float32),
            "soh_max": np.array(traj_soh_max, dtype=np.float32),
            "arbitrage_per_hour": np.array(traj_arbitrage, dtype=np.float32),
            "flex_per_hour": np.array(traj_flex, dtype=np.float32),
            "deg_per_hour": np.array(traj_deg, dtype=np.float32),
            "penalty_per_hour": np.array(traj_penalty, dtype=np.float32),
            "bid_fcr_mw_per_hour": np.array(traj_bid_fcr_mw, dtype=np.float32),
            "bid_afrr_mw_per_hour": np.array(traj_bid_afrr_mw, dtype=np.float32),
            "bid_mfrr_mw_per_hour": np.array(traj_bid_mfrr_mw, dtype=np.float32),
            "bid_afrr_up_mw_per_hour": np.array(traj_bid_afrr_up_mw, dtype=np.float32),
            "bid_afrr_dn_mw_per_hour": np.array(traj_bid_afrr_dn_mw, dtype=np.float32),
            "bid_mfrr_up_mw_per_hour": np.array(traj_bid_mfrr_up_mw, dtype=np.float32),
            "bid_mfrr_dn_mw_per_hour": np.array(traj_bid_mfrr_dn_mw, dtype=np.float32),
            "price_per_hour": np.array(traj_price, dtype=np.float32),
            "price_fcr_per_hour":     np.array(traj_price_fcr, dtype=np.float32),
            "price_afrr_up_per_hour": np.array(traj_price_afrr_up, dtype=np.float32),
            "price_afrr_dn_per_hour": np.array(traj_price_afrr_dn, dtype=np.float32),
            "price_mfrr_up_per_hour": np.array(traj_price_mfrr_up, dtype=np.float32),
            "price_mfrr_dn_per_hour": np.array(traj_price_mfrr_dn, dtype=np.float32),
            "price_afrr_per_hour":    np.array(traj_price_afrr, dtype=np.float32),
            "price_mfrr_per_hour":    np.array(traj_price_mfrr, dtype=np.float32),
            "rev_fcr_per_hour":     np.array(traj_rev_fcr, dtype=np.float32),
            "rev_afrr_up_per_hour": np.array(traj_rev_afrr_up, dtype=np.float32),
            "rev_afrr_dn_per_hour": np.array(traj_rev_afrr_dn, dtype=np.float32),
            "rev_mfrr_up_per_hour": np.array(traj_rev_mfrr_up, dtype=np.float32),
            "rev_mfrr_dn_per_hour": np.array(traj_rev_mfrr_dn, dtype=np.float32),
            "rev_afrr_per_hour":    np.array(traj_rev_afrr, dtype=np.float32),
            "rev_mfrr_per_hour":    np.array(traj_rev_mfrr, dtype=np.float32),
        }

    # Status summary
    from collections import Counter
    status_counts = Counter(daily_statuses)
    n_optimal = status_counts.get("Optimal", 0)
    n_total = len(daily_statuses)
    if n_optimal < n_total:
        print(f"  [demo summary] MILP status: {dict(status_counts)} "
              f"({n_optimal}/{n_total} Optimal)")

    return MultiDayDemoResult(
        obs=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int64),
        daily_milp_profits=daily_profits,
        daily_env_realized_profits=daily_env_realized,
        daily_milp_status=daily_statuses,
        final_soc_per_bess=soc,
        final_fce_per_bess=fce,
        final_capacity_lost_per_bess=cap_lost,
        total_milp_profit=float(sum(daily_profits)),
        total_env_realized_profit=float(sum(daily_env_realized)),
        elapsed_seconds=time.time() - t0,
        trajectory=trajectory,
    )


# ============================================================================
# Multi-day policy evaluation (env rollout)
# ============================================================================

@dataclass
class MultiDayEvalResult:
    """Output of evaluate_policy_multi_day."""
    total_profit_eur: float
    daily_profits: List[float]
    total_arbitrage: float
    total_flex_revenue: float
    total_degradation: float
    final_soc_per_bess: List[float]
    final_fce_per_bess: List[float]
    final_soh_per_bess: List[float]
    final_capacity_lost_per_bess: List[float] = field(default_factory=list)
    # Optional hourly trajectory (populated when record_trajectory=True)
    # All arrays are length n_hours = n_days * 24, indexed in chronological order
    trajectory: Optional[Dict[str, Any]] = None


def evaluate_policy_multi_day(
    fleet: MultiBatteryParameters,
    market_window: MarketWindow,
    policy_fn,  # callable: obs_array -> action_array
    use_nonlinear_degradation: bool = False,
    initial_soc: Optional[List[float]] = None,
    initial_fce_cumulative: Optional[List[float]] = None,
    initial_capacity_lost: Optional[List[float]] = None,
    env_seed: int = 0,
    record_trajectory: bool = False,
    directional_services: bool = False,
    progress_label: str = "policy",
    enable_commitments: bool = False,
    commitment_lead_time: int = 24,
    penalty_k: float = 1.5,
    full_foresight: bool = False,
    overcommit_penalty: float = 0.0,
    sustain: SustainDurations = DEFAULT_SUSTAIN,
    duration_features: bool = True,
    coupling: FleetCoupling = FleetCoupling.uncoupled(),
    centralized_critic: bool = False,
) -> MultiDayEvalResult:
    """Evaluate a policy across multiple days with rolling state.

    The policy is applied per-agent per-hour using its local observation.
    """
    N = fleet.n_batteries
    if initial_soc is None:
        initial_soc = [0.5] * N
    if initial_fce_cumulative is None:
        initial_fce_cumulative = [0.0] * N
    if initial_capacity_lost is None:
        initial_capacity_lost = [0.0] * N

    soc = list(initial_soc)
    fce = list(initial_fce_cumulative)
    cap_lost = list(initial_capacity_lost)
    total_profit = 0.0
    daily_profits = []
    total_arb = 0.0
    total_flex = 0.0
    total_deg = 0.0
    final_soh = [1.0] * N

    # Trajectory recording buffers (only used if record_trajectory=True)
    n_axes = ACTION_AXES_DIRECTIONAL if directional_services else ACTION_AXES
    traj_actions_mean = []
    traj_soc_mean = []
    traj_soc_min = []
    traj_soc_max = []
    traj_arbitrage = []
    traj_flex = []
    traj_deg = []
    # Per-service revenue buffers
    traj_rev_fcr = []; traj_rev_afrr_up = []; traj_rev_afrr_dn = []
    traj_rev_mfrr_up = []; traj_rev_mfrr_dn = []
    traj_rev_afrr = []; traj_rev_mfrr = []
    traj_bid_fcr_mw = []
    traj_bid_afrr_mw = []
    traj_bid_mfrr_mw = []
    # Directional bid splits
    traj_bid_afrr_up_mw = []; traj_bid_afrr_dn_mw = []
    traj_bid_mfrr_up_mw = []; traj_bid_mfrr_dn_mw = []
    traj_price = []
    # Per-service prices
    traj_price_fcr = []; traj_price_afrr_up = []; traj_price_afrr_dn = []
    traj_price_mfrr_up = []; traj_price_mfrr_dn = []
    traj_price_afrr = []; traj_price_mfrr = []
    traj_soh_mean = []; traj_soh_min = []; traj_soh_max = []
    traj_action_bin_counts = np.zeros((n_axes, N_ACTION_BINS), dtype=np.int64)

    # Progress print: emit one summary line every PROGRESS_EVERY days for
    # long windows (>50 days), so the user sees per-policy progress during
    # the rollout phase. The progress_label is propagated by the caller
    # (run_full_pipeline) so each policy gets a distinct prefix.
    eval_t0 = time.time()
    PROGRESS_EVERY = 10
    enable_progress = (market_window.n_days > 50)
    if enable_progress:
        print(f"  [eval {progress_label}] starting on {market_window.n_days} days, "
              f"N={N} BESS", flush=True)

    for day in range(market_window.n_days):
        prices_day, services_day = market_window.slice_day(day)

        if enable_progress and day > 0 and day % PROGRESS_EVERY == 0:
            elapsed_so_far = time.time() - eval_t0
            avg_per_day = elapsed_so_far / day
            eta_minutes = avg_per_day * (market_window.n_days - day) / 60.0
            running_profit = total_profit
            print(f"  [eval {progress_label}] day {day}/{market_window.n_days} | "
                  f"elapsed {elapsed_so_far:.0f}s ({avg_per_day:.1f}s/day) | "
                  f"running profit: {running_profit:,.0f} EUR | "
                  f"ETA: {eta_minutes:.1f} min", flush=True)
        env = MultiBESSEnv(
            fleet,
            episode_hours=24,
            prices=prices_day,
            services=services_day,
            use_nonlinear_degradation=use_nonlinear_degradation,
            fce_cumulative_initial=fce,
            capacity_lost_initial=cap_lost,
            soc_init=0.5,
            seed=env_seed + day,
            directional_services=directional_services,
            sustain_hours=sustain,
            duration_features=duration_features,
            centralized_critic=centralized_critic,
            coupling=coupling,
        )
        if enable_commitments:
            env.configure_commitments(True, commitment_lead_time, penalty_k)
            if full_foresight:
                env.configure_full_foresight(True)
            if overcommit_penalty:
                env.configure_overcommit_penalty(overcommit_penalty)
        env.reset(seed=env_seed + day)
        env._socs = np.array(soc, dtype=np.float64)
        obs = {a: env._build_observation(a) for a in env.agents}

        day_profit = 0.0
        day_arb = 0.0
        day_flex = 0.0
        day_deg = 0.0
        hour = 0
        while env.agents:
            actions = {a: policy_fn(obs[a]) for a in env.agents}

            if record_trajectory:
                act_stack = np.array([actions[a] for a in env.agents], dtype=np.int64)
                act_frac = act_stack.astype(np.float32) / (N_ACTION_BINS - 1)
                traj_actions_mean.append(act_frac.mean(axis=0))
                for k in range(n_axes):
                    bin_counts = np.bincount(act_stack[:, k],
                                              minlength=N_ACTION_BINS)
                    traj_action_bin_counts[k] += bin_counts
                bid_fcr = 0.0; bid_afrr = 0.0; bid_mfrr = 0.0
                bid_aup = 0.0; bid_adn = 0.0; bid_mup = 0.0; bid_mdn = 0.0
                for a in env.agents:
                    i = env.agent_name_mapping[a]
                    P_max = fleet.batteries[i].max_power_mw
                    frac = actions[a].astype(np.float32) / (N_ACTION_BINS - 1)
                    bid_fcr += frac[2] * P_max
                    if directional_services:
                        bid_aup += frac[3] * P_max
                        bid_adn += frac[4] * P_max
                        bid_mup += frac[5] * P_max
                        bid_mdn += frac[6] * P_max
                        bid_afrr = bid_aup + bid_adn
                        bid_mfrr = bid_mup + bid_mdn
                    else:
                        bid_afrr += frac[3] * P_max
                        bid_mfrr += frac[4] * P_max
                traj_bid_fcr_mw.append(bid_fcr)
                traj_bid_afrr_mw.append(bid_afrr)
                traj_bid_mfrr_mw.append(bid_mfrr)
                traj_bid_afrr_up_mw.append(bid_aup)
                traj_bid_afrr_dn_mw.append(bid_adn)
                traj_bid_mfrr_up_mw.append(bid_mup)
                traj_bid_mfrr_dn_mw.append(bid_mdn)
                traj_price.append(prices_day[hour])
                # Per-service energy prices for this hour
                from flexibility_market import ServiceType as _ST_eval
                price_lookup = {s.service_type: s.energy_price
                                for s in services_day[hour]}
                traj_price_fcr.append(float(price_lookup.get(_ST_eval.FCR, 0.0)))
                if directional_services:
                    traj_price_afrr_up.append(float(price_lookup.get(_ST_eval.AFRR_UP, 0.0)))
                    traj_price_afrr_dn.append(float(price_lookup.get(_ST_eval.AFRR_DN, 0.0)))
                    traj_price_mfrr_up.append(float(price_lookup.get(_ST_eval.MFRR_UP, 0.0)))
                    traj_price_mfrr_dn.append(float(price_lookup.get(_ST_eval.MFRR_DN, 0.0)))
                else:
                    traj_price_afrr.append(float(price_lookup.get(_ST_eval.AFRR, 0.0)))
                    traj_price_mfrr.append(float(price_lookup.get(_ST_eval.MFRR, 0.0)))
                # SOC stats BEFORE step (post-action SOC will be recorded next hour)
                socs_arr = np.array(env._socs, dtype=np.float64)
                traj_soc_mean.append(float(socs_arr.mean()))
                traj_soc_min.append(float(socs_arr.min()))
                traj_soc_max.append(float(socs_arr.max()))

            obs, rewards, _, _, infos = env.step(actions)
            hour_arb = 0.0; hour_flex = 0.0; hour_deg = 0.0
            for a in rewards:
                day_profit += rewards[a]
                hour_arb += infos[a]['arbitrage']
                hour_flex += infos[a]['flex_revenue']
                hour_deg += infos[a]['degradation']
            day_arb += hour_arb
            day_flex += hour_flex
            day_deg += hour_deg

            if record_trajectory:
                traj_arbitrage.append(hour_arb)
                traj_flex.append(hour_flex)
                traj_deg.append(hour_deg)
                agg = getattr(env, 'last_aggregate_info', {}) or {}
                traj_rev_fcr.append(float(agg.get('rev_fcr', 0.0)))
                if directional_services:
                    traj_rev_afrr_up.append(float(agg.get('rev_afrr_up', 0.0)))
                    traj_rev_afrr_dn.append(float(agg.get('rev_afrr_dn', 0.0)))
                    traj_rev_mfrr_up.append(float(agg.get('rev_mfrr_up', 0.0)))
                    traj_rev_mfrr_dn.append(float(agg.get('rev_mfrr_dn', 0.0)))
                else:
                    traj_rev_afrr.append(float(agg.get('rev_afrr', 0.0)))
                    traj_rev_mfrr.append(float(agg.get('rev_mfrr', 0.0)))
                soh_arr = np.array([s.soh for s in env._lfp_states], dtype=np.float64)
                traj_soh_mean.append(float(soh_arr.mean()))
                traj_soh_min.append(float(soh_arr.min()))
                traj_soh_max.append(float(soh_arr.max()))
            hour += 1

        for i in range(N):
            soc[i] = float(env._socs[i])
            fce[i] = float(env._lfp_states[i].fce_cumulative)
            cap_lost[i] = float(env._lfp_states[i].capacity_lost_fraction)
            final_soh[i] = float(env._lfp_states[i].soh)

        daily_profits.append(day_profit)
        total_profit += day_profit
        total_arb += day_arb
        total_flex += day_flex
        total_deg += day_deg

    trajectory = None
    if record_trajectory:
        trajectory = {
            "actions_mean_per_hour": np.array(traj_actions_mean, dtype=np.float32),
            "action_bin_counts": traj_action_bin_counts,
            "soc_mean": np.array(traj_soc_mean, dtype=np.float32),
            "soc_min": np.array(traj_soc_min, dtype=np.float32),
            "soc_max": np.array(traj_soc_max, dtype=np.float32),
            "soh_mean": np.array(traj_soh_mean, dtype=np.float32),
            "soh_min": np.array(traj_soh_min, dtype=np.float32),
            "soh_max": np.array(traj_soh_max, dtype=np.float32),
            "arbitrage_per_hour": np.array(traj_arbitrage, dtype=np.float32),
            "flex_per_hour": np.array(traj_flex, dtype=np.float32),
            "deg_per_hour": np.array(traj_deg, dtype=np.float32),
            "bid_fcr_mw_per_hour": np.array(traj_bid_fcr_mw, dtype=np.float32),
            "bid_afrr_mw_per_hour": np.array(traj_bid_afrr_mw, dtype=np.float32),
            "bid_mfrr_mw_per_hour": np.array(traj_bid_mfrr_mw, dtype=np.float32),
            "bid_afrr_up_mw_per_hour": np.array(traj_bid_afrr_up_mw, dtype=np.float32),
            "bid_afrr_dn_mw_per_hour": np.array(traj_bid_afrr_dn_mw, dtype=np.float32),
            "bid_mfrr_up_mw_per_hour": np.array(traj_bid_mfrr_up_mw, dtype=np.float32),
            "bid_mfrr_dn_mw_per_hour": np.array(traj_bid_mfrr_dn_mw, dtype=np.float32),
            "price_per_hour": np.array(traj_price, dtype=np.float32),
            "price_fcr_per_hour":     np.array(traj_price_fcr, dtype=np.float32),
            "price_afrr_up_per_hour": np.array(traj_price_afrr_up, dtype=np.float32),
            "price_afrr_dn_per_hour": np.array(traj_price_afrr_dn, dtype=np.float32),
            "price_mfrr_up_per_hour": np.array(traj_price_mfrr_up, dtype=np.float32),
            "price_mfrr_dn_per_hour": np.array(traj_price_mfrr_dn, dtype=np.float32),
            "price_afrr_per_hour":    np.array(traj_price_afrr, dtype=np.float32),
            "price_mfrr_per_hour":    np.array(traj_price_mfrr, dtype=np.float32),
            "rev_fcr_per_hour":     np.array(traj_rev_fcr, dtype=np.float32),
            "rev_afrr_up_per_hour": np.array(traj_rev_afrr_up, dtype=np.float32),
            "rev_afrr_dn_per_hour": np.array(traj_rev_afrr_dn, dtype=np.float32),
            "rev_mfrr_up_per_hour": np.array(traj_rev_mfrr_up, dtype=np.float32),
            "rev_mfrr_dn_per_hour": np.array(traj_rev_mfrr_dn, dtype=np.float32),
            "rev_afrr_per_hour":    np.array(traj_rev_afrr, dtype=np.float32),
            "rev_mfrr_per_hour":    np.array(traj_rev_mfrr, dtype=np.float32),
        }

    return MultiDayEvalResult(
        total_profit_eur=total_profit,
        daily_profits=daily_profits,
        total_arbitrage=total_arb,
        total_flex_revenue=total_flex,
        total_degradation=total_deg,
        final_soc_per_bess=soc,
        final_fce_per_bess=fce,
        final_soh_per_bess=final_soh,
        final_capacity_lost_per_bess=cap_lost,
        trajectory=trajectory,
    )


# ============================================================================
# Convenience policy adapters
# ============================================================================

def make_bc_policy_fn(bc_net: BCPolicyNet, deterministic: bool = True,
                      centralized_critic: bool = False):
    """The clone acting as a decentralised policy.

    `centralized_critic` must match the environment the clone is evaluated in:
    the observation carries the fleet block only when it is on, and the clone
    was trained without it either way.
    """
    def policy(obs):
        return bc_net.predict(mask_global_block(obs, centralized_critic),
                              deterministic=deterministic)
    return policy


def make_random_policy_fn(seed: int = 0, n_axes: int = ACTION_AXES):
    rng = np.random.default_rng(seed)
    def policy(obs):
        return rng.integers(0, N_ACTION_BINS, size=n_axes, dtype=np.int64)
    return policy


def make_milp_actions_replay_fn(action_sequence: List[Dict[str, np.ndarray]]):
    """A 'policy' that replays a fixed pre-computed action sequence by hour
    and agent_id. Used to verify the MILP oracle on the env (sanity check)."""
    # Not exposed publicly; left here as a debugging aid.
    raise NotImplementedError


# ============================================================================
# Headline orchestrator
# ============================================================================

@dataclass
class FullPipelineResult:
    """Top-level summary returned by run_full_pipeline."""
    # Configuration echo
    n_batteries: int
    train_days: int
    test_days: int
    use_nonlinear_degradation: bool

    # Per-stage timing
    demo_generation_seconds: float
    bc_training_seconds: float
    evaluation_seconds: float

    # Train-window MILP oracle
    train_total_milp_profit_eur: float
    train_milp_optimal_day_count: int

    # Test-window MILP oracle (gold standard)
    test_total_milp_profit_eur: float
    test_milp_daily_profits: List[float]

    # BC training
    bc_dataset_size: int
    bc_final_train_loss: float
    bc_final_val_accuracy_per_axis: List[float]

    # Test-window BC policy
    test_total_bc_profit_eur: float
    test_bc_daily_profits: List[float]

    # Test-window random baseline
    test_total_random_profit_eur: float

    # Headline gap
    bc_gap_to_milp_percent: float
    bc_uplift_over_random_factor: float

    # PPO comparison (Step 6 algorithm; populated when run_ppo_* flags True)
    test_milp_objective_eur: Optional[float] = None     # continuous MILP optimum
    test_ppo_vanilla_profit_eur: Optional[float] = None
    test_ppo_bc_warmstart_profit_eur: Optional[float] = None
    # Same policies evaluated by SAMPLING from the action distribution
    # instead of taking the per-axis argmax. The two can differ enormously
    # for a high-entropy policy: the mode can be the no-op while the mass
    # elsewhere is what earns. Reporting only one of them makes the number
    # an artefact of an undeclared protocol choice.
    test_ppo_vanilla_profit_stochastic_eur: Optional[float] = None
    test_ppo_bc_warmstart_profit_stochastic_eur: Optional[float] = None
    test_ppo_vanilla_daily_profits: Optional[List[float]] = None
    test_ppo_bc_warmstart_daily_profits: Optional[List[float]] = None

    # Per-service decomposition on the test window
    decomposition: Optional[Dict[str, Dict[str, float]]] = None

    # Hourly trajectories for each policy on the test window (populated when
    # the eval functions record them). Used by marl_analysis for terminal
    # reports and chart generation. Keys: "milp", "bc", "ppo_vanilla",
    # "ppo_bc", "random".
    trajectories: Optional[Dict[str, Dict[str, Any]]] = None

    # PPO training learning curves (per-iteration mean episode reward) for
    # the marl_analysis chart generator. Keys: "ppo_vanilla", "ppo_bc".
    ppo_training_metrics: Optional[Dict[str, list]] = None

    # Full BC training history (per-epoch train_loss + val_acc_per_axis), for
    # the BC training curves chart in marl_analysis.
    bc_history: Optional[List[Dict[str, Any]]] = None

    # MILP expert demonstrations on the train window (obs+actions). Exposed
    # for the BC-vs-MILP action confusion-matrix analysis.
    train_demo_obs: Optional[np.ndarray] = None
    train_demo_actions: Optional[np.ndarray] = None
    bc_train_predictions: Optional[np.ndarray] = None


def run_full_pipeline(
    fleet: MultiBatteryParameters,
    train_start: datetime,
    train_days: int,
    test_days: int,
    use_nonlinear_degradation: bool = False,
    pun_calibration: Optional[PUN2024Calibration] = None,
    service_calibration: Optional[ServiceCalibration] = None,
    data_seed: int = 0,
    bc_n_epochs: int = 50,
    bc_batch_size: int = 64,
    bc_lr: float = 3e-3,
    bc_seed: int = 0,
    eval_seed: int = 100,
    real_pun_xlsx_path: Optional[str] = None,
    real_msd_xlsx_paths: Optional[List[str]] = None,
    # "NORD" for a northern-zone fleet (matches the EsitiMSD_*_Nord
    # series); "PUN" reproduces the pre-revision behaviour. See
    # italian_market_data.load_pun_from_gme_xlsx.
    price_column: str = "PUN",
    run_ppo_vanilla: bool = False,
    run_ppo_bc_warmstart: bool = False,
    ppo_iterations: int = 200,
    directional_services: bool = False,
    master_seed: Optional[int] = None,
    enable_commitments: bool = False,
    commitment_lead_time: int = 24,
    penalty_k: float = 1.5,
    full_foresight: bool = False,
    overcommit_penalty: float = 0.0,
    expected_reward_training: bool = False,
    milp_mode: str = "continuous",
    # --- revision hooks ---------------------------------------------
    # sustain: the SustainDurations scenario shared by env, MILP and BC.
    # The three warm-start factors below used to be bundled together, which
    # is the confound Reviewers 1/2/3 all flagged; marl_ablation.py varies
    # them one at a time.
    sustain: SustainDurations = DEFAULT_SUSTAIN,
    use_kl_anchor: bool = True,
    # One PPO policy per asset cluster instead of one shared across all
    # units. Tests whether parameter sharing is what breaks when the clusters
    # have genuinely different storage durations.
    per_cluster_policies: bool = False,
    # Duration-relative observation features (marl_env 26-28): usable energy,
    # headroom and duration expressed in hours at rated power, the coordinates
    # in which the sustain constraint is asset-invariant.
    duration_features: bool = True,
    warmstart_lr_scale: float = 0.1,
    warmstart_entropy_coeff: Optional[float] = 0.0,
    # Shared grid connection limit: the only constraint that couples the units
    # to each other, and therefore the only thing that turns N independent
    # problems into a coordination problem. Reaches BOTH the MILP and the
    # environment, or the benchmark would plan against a constraint the policy
    # is never evaluated under.
    coupling: FleetCoupling = FleetCoupling.uncoupled(),
    # Centralised critic: appends the fleet-level global state to every
    # observation and gives the value head a module that reads it while the
    # policy head does not (CTDE). Only meaningful together with a coupling:
    # with independent units the joint value is the sum of the individual
    # ones and a global critic has nothing extra to learn.
    centralized_critic: bool = False,
) -> FullPipelineResult:
    """End-to-end pipeline: data -> MILP demos -> BC training -> evaluation.

    The train and test windows are CONTIGUOUS: train covers
    [train_start, train_start + train_days), test covers
    [train_start + train_days, train_start + train_days + test_days).
    The test window starts with the SOC and FCE state at the end of the
    train window, so the test is genuinely a continuation of fleet
    operation under different policies.

    If `real_pun_xlsx_path` is provided, hourly PUN prices come from the
    real GME .xlsx. If additionally `real_msd_xlsx_paths` is supplied,
    real Terna MSD ex-ante hourly prices overlay the synthetic aFRR/mFRR
    energy_price. Capacity prices and activation probabilities remain at
    the calibrated defaults from ServiceCalibration (sourced from the
    user's Italian_BESS_Markets_Reference workbook).
    """
    N = fleet.n_batteries

    # ------------------------------------------------------------------
    # Reproducibility: pin all global RNGs ONCE, before anything random
    # happens (env build, MILP, BC init, PPO weight init). If master_seed is
    # given, derive every per-stage seed from it so a single number fully
    # determines the run; otherwise keep the explicitly-passed seeds and just
    # pin globals to data_seed for backward compatibility.
    # ------------------------------------------------------------------
    from seeding import set_global_seeds, derive_seeds
    if master_seed is not None:
        _seeds = derive_seeds(master_seed)
        data_seed = _seeds["data_seed"]
        bc_seed = _seeds["bc_seed"]
        eval_seed = _seeds["eval_seed"]
        ppo_seed = _seeds["ppo_seed"]
        set_global_seeds(master_seed)
        print(f"[pipeline] reproducible mode: master_seed={master_seed} -> "
              f"data={data_seed}, bc={bc_seed}, eval={eval_seed}, ppo={ppo_seed}")
    else:
        ppo_seed = eval_seed
        set_global_seeds(data_seed)

    print(f"[pipeline] fleet coupling: {coupling.describe()}")
    print(f"[pipeline] centralised critic: "
          f"{'ON (CTDE)' if centralized_critic else 'OFF'}")
    if centralized_critic and not coupling.is_active:
        print("  [pipeline] NOTE: a centralised critic without a coupling has "
              "nothing extra to learn; the joint value is the sum of the "
              "individual ones. Pair it with --connection-limit.")
    print(f"[pipeline] duration-relative obs features (26-28): "
          f"{'ON' if duration_features else 'OFF (zeroed)'}")
    print(f"[pipeline] sustain windows: {sustain.describe()}")
    if not sustain.is_sogl_compliant():
        print("  [pipeline] WARNING: tau_FCR is outside the SO GL Art. 156(10) "
              "range [15, 30] min; valid only as a sensitivity point.")

    test_start = train_start + timedelta(days=train_days)
    test_end = test_start + timedelta(days=test_days)

    # Build train and test market windows
    print(f"[pipeline] building market windows: train={train_days}d, test={test_days}d")
    if real_pun_xlsx_path is not None:
        from italian_market_data import make_market_window_from_real_pun
        train_window = make_market_window_from_real_pun(
            start_date=train_start, end_date=test_start,
            pun_xlsx_path=real_pun_xlsx_path,
            price_column=price_column,
            service_calibration=service_calibration,
            msd_xlsx_paths=real_msd_xlsx_paths,
            directional_services=directional_services,
        )
        test_window = make_market_window_from_real_pun(
            start_date=test_start, end_date=test_end,
            pun_xlsx_path=real_pun_xlsx_path,
            price_column=price_column,
            service_calibration=service_calibration,
            msd_xlsx_paths=real_msd_xlsx_paths,
            directional_services=directional_services,
        )
        msd_note = "+MSD" if real_msd_xlsx_paths else "no MSD"
        print(f"  day-ahead series: {price_column} "
              f"from {real_pun_xlsx_path} ({msd_note})")
        if str(price_column).upper() == "PUN" and real_msd_xlsx_paths:
            print("  [pipeline] NOTE: the day-ahead leg uses the national "
                  "PUN while the reserve leg uses northern-zone MSD data. "
                  "Storage settles zonally; pass price_column='NORD' for "
                  "a consistent northern-zone study.")
        print(f"  train prices: mean {train_window.prices_hourly.mean():.2f} EUR/MWh, "
              f"std {train_window.prices_hourly.std():.2f}")
        print(f"  test prices:  mean {test_window.prices_hourly.mean():.2f} EUR/MWh, "
              f"std {test_window.prices_hourly.std():.2f}")
    else:
        train_window = make_synthetic_market_window(
            start_date=train_start, end_date=test_start,
            pun_calibration=pun_calibration,
            service_calibration=service_calibration,
            seed=data_seed,
        )
        test_window = make_synthetic_market_window(
            start_date=test_start, end_date=test_end,
            pun_calibration=pun_calibration,
            service_calibration=service_calibration,
            seed=data_seed + 1,
        )
        print(f"  using SYNTHETIC PUN")

    # MILP demos on the train window
    print(f"[pipeline] generating MILP expert demos on train window...")
    train_demo = generate_multi_day_expert_demos(
        fleet, train_window,
        use_nonlinear_degradation=use_nonlinear_degradation,
        env_seed=eval_seed,
        directional_services=directional_services,
        enable_commitments=enable_commitments,
        commitment_lead_time=commitment_lead_time,
        penalty_k=penalty_k,
            full_foresight=full_foresight,
            overcommit_penalty=overcommit_penalty,
            milp_mode=milp_mode,
            sustain=sustain,
            duration_features=duration_features,
            coupling=coupling,
            centralized_critic=centralized_critic,
    )
    print(f"  demos: {train_demo.obs.shape[0]} samples, "
          f"total train MILP profit: {train_demo.total_milp_profit:.2f} EUR, "
          f"time: {train_demo.elapsed_seconds:.1f}s")

    # BC training on pooled dataset
    print(f"[pipeline] BC training: {bc_n_epochs} epochs on "
          f"{train_demo.obs.shape[0]} samples...")
    t_bc0 = time.time()
    n_axes = ACTION_AXES_DIRECTIONAL if directional_services else ACTION_AXES
    bc_net, history = train_bc(
        train_demo.obs, train_demo.actions,
        n_epochs=bc_n_epochs, batch_size=bc_batch_size, lr=bc_lr, seed=bc_seed,
        n_axes=n_axes,
    )
    bc_time = time.time() - t_bc0
    print(f"  BC final loss: {history[-1]['train_loss']:.4f}, "
          f"acc per axis: {[round(a, 3) for a in history[-1]['val_acc_per_axis']]}")

    # Pre-compute BC predictions on the entire train demo set (for the
    # BC-vs-MILP action confusion-matrix chart in marl_analysis). Cheap
    # single batch forward pass; predictions are deterministic argmax.
    try:
        import torch as _torch
        bc_net.eval()
        with _torch.no_grad():
            obs_t = _torch.tensor(train_demo.obs, dtype=_torch.float32)
            logits = bc_net.forward(obs_t)  # (M, n_axes, N_ACTION_BINS)
            bc_train_predictions = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    except Exception as exc:
        print(f"  [warn] BC batch predict failed: {exc}")
        bc_train_predictions = None

    # Test window starts with the post-train SOC and FCE state
    test_initial_soc = train_demo.final_soc_per_bess
    test_initial_fce = train_demo.final_fce_per_bess
    test_initial_cap_lost = train_demo.final_capacity_lost_per_bess or [0.0] * fleet.n_batteries

    # MILP oracle on the test window
    print(f"[pipeline] MILP oracle on test window...")
    t_eval0 = time.time()
    test_demo = generate_multi_day_expert_demos(
        fleet, test_window,
        use_nonlinear_degradation=use_nonlinear_degradation,
        initial_soc=test_initial_soc,
        initial_fce_cumulative=test_initial_fce,
        initial_capacity_lost=test_initial_cap_lost,
        env_seed=eval_seed + 1000,
        record_trajectory=True,
        directional_services=directional_services,
        enable_commitments=enable_commitments,
        commitment_lead_time=commitment_lead_time,
        penalty_k=penalty_k,
            full_foresight=full_foresight,
            overcommit_penalty=overcommit_penalty,
            milp_mode=milp_mode,
            sustain=sustain,
            duration_features=duration_features,
            coupling=coupling,
            centralized_critic=centralized_critic,
    )

    # BC policy on the test window
    print(f"[pipeline] BC policy on test window...")
    bc_eval = evaluate_policy_multi_day(
        fleet, test_window,
        policy_fn=make_bc_policy_fn(bc_net, deterministic=True,
                                    centralized_critic=centralized_critic),
        progress_label="BC",
        use_nonlinear_degradation=use_nonlinear_degradation,
        initial_soc=test_initial_soc,
        initial_fce_cumulative=test_initial_fce,
        initial_capacity_lost=test_initial_cap_lost,
        env_seed=eval_seed + 1000,
        record_trajectory=True,
        directional_services=directional_services,
        enable_commitments=enable_commitments,
        commitment_lead_time=commitment_lead_time,
        penalty_k=penalty_k,
            full_foresight=full_foresight,
            overcommit_penalty=overcommit_penalty,
        centralized_critic=centralized_critic,
        coupling=coupling,
        duration_features=duration_features,
        sustain=sustain,
    )

    # Random baseline on the test window
    print(f"[pipeline] random baseline on test window...")
    random_eval = evaluate_policy_multi_day(
        fleet, test_window,
        policy_fn=make_random_policy_fn(seed=eval_seed + 999, n_axes=n_axes),
        progress_label="random",
        use_nonlinear_degradation=use_nonlinear_degradation,
        initial_soc=test_initial_soc,
        initial_fce_cumulative=test_initial_fce,
        initial_capacity_lost=test_initial_cap_lost,
        env_seed=eval_seed + 1000,
        record_trajectory=True,
        directional_services=directional_services,
        enable_commitments=enable_commitments,
        commitment_lead_time=commitment_lead_time,
        penalty_k=penalty_k,
            full_foresight=full_foresight,
            overcommit_penalty=overcommit_penalty,
        centralized_critic=centralized_critic,
        coupling=coupling,
        duration_features=duration_features,
        sustain=sustain,
    )

    # ---- PPO comparison (Step 6 algorithm) ----
    # The PPO is trained on the env with the SAME fleet config; evaluated on
    # the test window with the SAME evaluate_policy_multi_day used for BC and
    # random, so all four numbers are apples-to-apples (env-realised).
    ppo_results: Dict[str, Any] = {}
    ppo_stochastic: Dict[str, float] = {}
    ppo_training_metrics: Dict[str, list] = {}
    if run_ppo_vanilla or run_ppo_bc_warmstart:
        from marl_trainer import CLUSTER_POLICY_IDS, SHARED_POLICY_ID
        from marl_ppo_comparison import (train_ppo_policy,
                                         make_rllib_policy_fn,
                                         diagnose_zero_policy)
    if run_ppo_vanilla:
        print(f"[pipeline] PPO vanilla: {ppo_iterations} iters on train window...")
        algo_v = train_ppo_policy(
            fleet, n_iterations=ppo_iterations, bc_net=None,
            market_window=train_window,
            use_nonlinear_degradation=use_nonlinear_degradation, seed=ppo_seed,
            directional_services=directional_services,
            enable_commitments=enable_commitments,
            commitment_lead_time=commitment_lead_time,
            penalty_k=penalty_k,
            full_foresight=full_foresight,
            overcommit_penalty=overcommit_penalty,
            expected_reward_training=expected_reward_training,
            sustain=sustain,
            duration_features=duration_features,
            coupling=coupling,
            centralized_critic=centralized_critic,
        )
        ppo_training_metrics["ppo_vanilla"] = [
            float(m.get("episode_reward_mean", 0.0) or 0.0)
            for m in getattr(algo_v, "_lorenzo_training_metrics", [])
        ]
        print(f"[pipeline] PPO vanilla on test window...")
        ppo_v_eval = evaluate_policy_multi_day(
            fleet, test_window,
            policy_fn=make_rllib_policy_fn(algo_v, directional_services=directional_services),
            progress_label="PPO vanilla",
            use_nonlinear_degradation=use_nonlinear_degradation,
            initial_soc=test_initial_soc,
            initial_fce_cumulative=test_initial_fce,
            initial_capacity_lost=test_initial_cap_lost,
            env_seed=eval_seed + 1000,
            record_trajectory=True,
            directional_services=directional_services,
            enable_commitments=enable_commitments,
            commitment_lead_time=commitment_lead_time,
            penalty_k=penalty_k,
            full_foresight=full_foresight,
            overcommit_penalty=overcommit_penalty,
            sustain=sustain,
            duration_features=duration_features,
            coupling=coupling,
            centralized_critic=centralized_critic,
        )
        ppo_results["ppo_vanilla"] = ppo_v_eval
        try:
            algo_v.stop()
        except Exception:
            pass
    if run_ppo_bc_warmstart:
        print(f"[pipeline] PPO BC-warmstart: {ppo_iterations} iters on train window...")
        algo_w = train_ppo_policy(
            fleet, n_iterations=ppo_iterations, bc_net=bc_net,
            bc_kl_beta_start=1.0, bc_kl_anneal_iters=ppo_iterations,
            # ABLATION HOOKS (Reviewer 1 pt 1, Reviewer 2 pt 6, Reviewer 3
            # pt 1): the warm start changed four things at once. These three
            # are now independently switchable so marl_ablation.py can vary
            # exactly one factor at a time.
            use_kl_anchor=use_kl_anchor,
            warmstart_lr_scale=warmstart_lr_scale,
            warmstart_entropy_coeff=warmstart_entropy_coeff,
            per_cluster_policies=per_cluster_policies,
            centralized_critic=centralized_critic,
            market_window=train_window,
            use_nonlinear_degradation=use_nonlinear_degradation, seed=ppo_seed,
            directional_services=directional_services,
            enable_commitments=enable_commitments,
            commitment_lead_time=commitment_lead_time,
            penalty_k=penalty_k,
            full_foresight=full_foresight,
            overcommit_penalty=overcommit_penalty,
            expected_reward_training=expected_reward_training,
            sustain=sustain,
            duration_features=duration_features,
            coupling=coupling,
        )
        ppo_training_metrics["ppo_bc"] = [
            float(m.get("episode_reward_mean", 0.0) or 0.0)
            for m in getattr(algo_w, "_lorenzo_training_metrics", [])
        ]
        print(f"[pipeline] PPO BC-warmstart on test window...")
        ppo_w_eval = evaluate_policy_multi_day(
            fleet, test_window,
            policy_fn=make_rllib_policy_fn(
                algo_w, directional_services=directional_services,
                policy_id=(list(CLUSTER_POLICY_IDS) if per_cluster_policies
                           else SHARED_POLICY_ID)),
            progress_label="PPO BC-warm",
            use_nonlinear_degradation=use_nonlinear_degradation,
            initial_soc=test_initial_soc,
            initial_fce_cumulative=test_initial_fce,
            initial_capacity_lost=test_initial_cap_lost,
            env_seed=eval_seed + 1000,
            record_trajectory=True,
            directional_services=directional_services,
            enable_commitments=enable_commitments,
            commitment_lead_time=commitment_lead_time,
            penalty_k=penalty_k,
            full_foresight=full_foresight,
            overcommit_penalty=overcommit_penalty,
            sustain=sustain,
            duration_features=duration_features,
            coupling=coupling,
            centralized_critic=centralized_critic,
        )
        # Evaluate the SAME policy both ways, always. Evaluation takes the
        # per-axis argmax while training samples from the distribution, and
        # for a policy that has not yet concentrated its mass those two give
        # completely different answers: measured on the smoke run, argmax
        # scored 0.00 EUR and sampling scored 1,374.89 EUR for one and the
        # same set of weights. Which one is "the" result is a protocol choice
        # that has to be stated and defended, not left implicit, so both go
        # into the record and the paper can report the pair.
        print("[eval] same policy, stochastic (sampled) evaluation...")
        try:
            _stoch = evaluate_policy_multi_day(
                fleet, test_window,
                policy_fn=make_rllib_policy_fn(
                    algo_w, directional_services=directional_services,
                    deterministic=False,
                    policy_id=(list(CLUSTER_POLICY_IDS) if per_cluster_policies
                               else SHARED_POLICY_ID)),
                progress_label="PPO BC-warm (stochastic)",
                use_nonlinear_degradation=use_nonlinear_degradation,
                initial_soc=test_initial_soc,
                initial_fce_cumulative=test_initial_fce,
                initial_capacity_lost=test_initial_cap_lost,
                env_seed=eval_seed + 1000,
                record_trajectory=False,
                directional_services=directional_services,
                enable_commitments=enable_commitments,
                commitment_lead_time=commitment_lead_time,
                penalty_k=penalty_k,
                full_foresight=full_foresight,
                overcommit_penalty=overcommit_penalty,
                sustain=sustain,
                duration_features=duration_features,
                coupling=coupling,
                centralized_critic=centralized_critic,
            )
            ppo_stochastic["ppo_bc"] = float(_stoch.total_profit_eur)
            print(f"[eval] ppo_bc  argmax {ppo_w_eval.total_profit_eur:,.2f} EUR"
                  f"  |  sampled {_stoch.total_profit_eur:,.2f} EUR")
        except Exception as _exc:
            print(f"[eval] stochastic evaluation failed: {_exc}")

        # Action statistics explain a large gap between the two.
        try:
            _traj = (ppo_w_eval.trajectory or {}).get("observations")
            if _traj is not None and len(_traj):
                diagnose_zero_policy(algo_w, _traj, directional_services,
                                     bc_net=bc_net, label="ppo_bc")
        except Exception as _exc:
            print(f"[eval] action statistics unavailable: {_exc}")

        ppo_results["ppo_bc"] = ppo_w_eval
        try:
            algo_w.stop()
        except Exception:
            pass

    eval_time = time.time() - t_eval0

    # ---- Print four-way comparison if PPO ran ----
    if ppo_results:
        print("\n" + "=" * 70)
        print("FOUR-WAY COMPARISON (env-realised, apples-to-apples)")
        print("=" * 70)
        print(f"  MILP objective (continuous): {test_demo.total_milp_profit:>14,.2f} EUR  <- upper bound")
        print(f"  MILP realised (discrete):    {test_demo.total_env_realized_profit:>14,.2f} EUR")
        print(f"  BC policy:                   {bc_eval.total_profit_eur:>14,.2f} EUR")
        for tag, ev in ppo_results.items():
            print(f"  {tag:<29s}{ev.total_profit_eur:>14,.2f} EUR")
        print(f"  Random baseline:             {random_eval.total_profit_eur:>14,.2f} EUR")
        print("\n  Per-service decomposition (arbitrage / flex / degradation):")
        def _row(tag, ev):
            print(f"    {tag:<26s} arb {ev.total_arbitrage:>12,.0f} | "
                  f"flex {ev.total_flex_revenue:>12,.0f} | deg {ev.total_degradation:>12,.0f}")
        _row("BC", bc_eval)
        for tag, ev in ppo_results.items():
            _row(tag, ev)
        _row("random", random_eval)
        print("=" * 70)

    # Headline numbers: BC and random are env-realized; MILP comparison uses
    # the env-realized value (applying MILP-derived actions through the env
    # with the same seed) for apples-to-apples comparison. The MILP optimizer
    # objective is also reported for reference.
    milp_test_realized = test_demo.total_env_realized_profit
    milp_test_objective = test_demo.total_milp_profit
    bc_test_profit = bc_eval.total_profit_eur
    random_test_profit = random_eval.total_profit_eur
    gap_pct = ((milp_test_realized - bc_test_profit)
               / max(abs(milp_test_realized), 1.0)) * 100
    if random_test_profit > 0:
        uplift_factor = bc_test_profit / random_test_profit
    elif random_test_profit < 0 and bc_test_profit > 0:
        uplift_factor = float("inf")
    else:
        uplift_factor = 0.0
    train_optimal_count = sum(1 for s in train_demo.daily_milp_status if s == "Optimal")

    # Decomposition dict for export (arb / flex / deg per policy)
    decomposition = {
        "milp_realised": {
            "arbitrage": float(sum(test_demo.daily_env_realized_profits)),
            # MILP daily_env_realized is the net; we don't decompose it here.
            # If you need the per-service split for MILP, run a dedicated env
            # rollout under the MILP action sequence with eval accounting.
            "note": "MILP env-realised total only; per-service split not tracked",
        },
        "bc": {
            "arbitrage": float(bc_eval.total_arbitrage),
            "flex": float(bc_eval.total_flex_revenue),
            "degradation": float(bc_eval.total_degradation),
        },
        "random": {
            "arbitrage": float(random_eval.total_arbitrage),
            "flex": float(random_eval.total_flex_revenue),
            "degradation": float(random_eval.total_degradation),
        },
    }
    for tag, ev in ppo_results.items():
        decomposition[tag] = {
            "arbitrage": float(ev.total_arbitrage),
            "flex": float(ev.total_flex_revenue),
            "degradation": float(ev.total_degradation),
        }

    return FullPipelineResult(
        n_batteries=N,
        train_days=train_days,
        test_days=test_days,
        use_nonlinear_degradation=use_nonlinear_degradation,
        demo_generation_seconds=train_demo.elapsed_seconds,
        bc_training_seconds=bc_time,
        evaluation_seconds=eval_time,
        train_total_milp_profit_eur=train_demo.total_milp_profit,
        train_milp_optimal_day_count=train_optimal_count,
        test_total_milp_profit_eur=milp_test_realized,
        test_milp_daily_profits=test_demo.daily_env_realized_profits,
        bc_dataset_size=int(train_demo.obs.shape[0]),
        bc_final_train_loss=float(history[-1]['train_loss']),
        bc_final_val_accuracy_per_axis=list(history[-1]['val_acc_per_axis']),
        test_total_bc_profit_eur=bc_test_profit,
        test_bc_daily_profits=bc_eval.daily_profits,
        test_total_random_profit_eur=random_test_profit,
        bc_gap_to_milp_percent=gap_pct,
        bc_uplift_over_random_factor=uplift_factor,
        test_milp_objective_eur=milp_test_objective,
        test_ppo_vanilla_profit_eur=(
            ppo_results["ppo_vanilla"].total_profit_eur
            if "ppo_vanilla" in ppo_results else None
        ),
        test_ppo_vanilla_profit_stochastic_eur=ppo_stochastic.get("ppo_vanilla"),
        test_ppo_bc_warmstart_profit_stochastic_eur=ppo_stochastic.get("ppo_bc"),
        test_ppo_bc_warmstart_profit_eur=(
            ppo_results["ppo_bc"].total_profit_eur
            if "ppo_bc" in ppo_results else None
        ),
        test_ppo_vanilla_daily_profits=(
            ppo_results["ppo_vanilla"].daily_profits
            if "ppo_vanilla" in ppo_results else None
        ),
        test_ppo_bc_warmstart_daily_profits=(
            ppo_results["ppo_bc"].daily_profits
            if "ppo_bc" in ppo_results else None
        ),
        decomposition=decomposition,
        trajectories={
            "milp": test_demo.trajectory,
            "bc": bc_eval.trajectory,
            "random": random_eval.trajectory,
            **({"ppo_vanilla": ppo_results["ppo_vanilla"].trajectory}
               if "ppo_vanilla" in ppo_results else {}),
            **({"ppo_bc": ppo_results["ppo_bc"].trajectory}
               if "ppo_bc" in ppo_results else {}),
        },
        ppo_training_metrics=ppo_training_metrics if ppo_training_metrics else None,
        bc_history=history,
        train_demo_obs=train_demo.obs,
        train_demo_actions=train_demo.actions,
        bc_train_predictions=bc_train_predictions,
    )


# ============================================================================
# Preset configurations
# ============================================================================

def smoke_test_config() -> Dict[str, Any]:
    """Tiny config for the unit tests: N=3, 7 train days + 3 test days."""
    return dict(
        train_start=datetime(2024, 4, 1),
        train_days=7,
        test_days=3,
        bc_n_epochs=30,
        bc_batch_size=32,
        bc_lr=3e-3,
    )


def production_config() -> Dict[str, Any]:
    """Paper-run config: full year on the heterogeneous N=50 fleet.

    Train: Jan-Sep 2024 (273 days). Test: Oct-Dec 2024 (92 days).
    Expected runtime: 15-30 minutes on a decent multi-core CPU.
    """
    return dict(
        train_start=datetime(2024, 1, 1),
        train_days=273,
        test_days=92,
        bc_n_epochs=50,
        bc_batch_size=128,
        bc_lr=1e-3,
    )