# -*- coding: utf-8 -*-
"""
marl_dagger.py — Dataset Aggregation (DAgger) against the MILP expert.

WHY
---
The behavioural clone reaches 98-99% of the MILP on the held-out year and is
robust to fleet heterogeneity, while PPO fine-tuning degrades it. So the route
to closing the last one or two percent is better imitation, not reinforcement.

What is left is not network capacity, it is DISTRIBUTION SHIFT. The clone is
trained on the states the MILP visits, but at roll-out it visits its own
states. A small error moves the state of charge, the next observation is
slightly outside the training distribution, the next error is a little larger,
and the deviation compounds over the day and across days. This is the standard
failure mode of behavioural cloning and DAgger is its standard remedy:

    roll out the LEARNER, label the states it actually reaches with the
    EXPERT's action, aggregate, retrain, repeat.

The expert here is queryable at will, which is what makes DAgger applicable:
the MILP can be re-solved from any state of charge.

GRANULARITY, AND WHY IT IS NOT PER-HOUR
---------------------------------------
Textbook DAgger re-queries the expert at every visited state. Here that means
solving, at each hour t, a MILP over the remaining horizon starting from the
learner's state of charge at t: twenty-four solves per day instead of one.
At the measured ~50 s per 50-unit day that is twenty minutes per day and about
five days per DAgger iteration over a year. Not affordable.

`requery_stride_hours` therefore controls the granularity:

    24  (default)  one solve per day, from the state of charge the learner
                   carried into that day. Same cost as generating the original
                   demonstrations. This captures the dominant shift channel in
                   this problem, which is the SoC carried ACROSS days: the
                   clone's intra-day trajectory stays close to the expert's,
                   but the day-boundary SoC drifts and compounds.
    6              four solves per day, catching intra-day drift as well.
    1              textbook DAgger. Correct and unaffordable at fleet scale.

The stride is a stated approximation, not a hidden one: `DAggerResult` records
it so a results file always says how the labels were produced.

WHAT THIS MODULE DOES AND DOES NOT TEST
---------------------------------------
The expert re-query, the aggregation and the bookkeeping are exercised by
`test_dagger.py`. Retraining the clone needs torch and is delegated to
`marl_bc.train_bc`, so the first end-to-end validation happens on the machine
that has it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fleet_coupling import FleetCoupling
from market_constants import SustainDurations, DEFAULT_SUSTAIN


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DAggerConfig:
    """How the aggregation loop is run."""

    n_iterations: int = 3
    # Hours between expert re-queries. 24 = once per day (see module docstring).
    requery_stride_hours: int = 24
    # Probability of following the EXPERT rather than the learner while
    # collecting. Classic DAgger anneals beta from 1 to 0; iteration 0 is the
    # plain behavioural clone, so beta is 0 from iteration 1 onward by default
    # (pure learner roll-out), which is the usual practical choice and the one
    # that actually exposes the learner's own state distribution.
    beta_schedule: Tuple[float, ...] = (0.0,)
    # Keep every past iteration's data (true aggregation) rather than training
    # on the newest batch alone. Aggregation is what gives DAgger its
    # no-regret guarantee; training on the last batch only is a different and
    # weaker algorithm.
    aggregate: bool = True
    # Cap on the aggregated dataset, in samples. None keeps everything.
    max_samples: Optional[int] = None

    def beta(self, iteration: int) -> float:
        if not self.beta_schedule:
            return 0.0
        idx = min(iteration, len(self.beta_schedule) - 1)
        return float(self.beta_schedule[idx])


@dataclass
class DAggerIteration:
    """One pass of collect-label-aggregate-retrain."""
    iteration: int
    beta: float
    new_samples: int
    total_samples: int
    expert_queries: int
    collect_seconds: float = 0.0
    train_seconds: float = 0.0
    bc_final_loss: float = float("nan")
    bc_acc_per_axis: List[float] = field(default_factory=list)
    eval_profit_eur: float = float("nan")
    milp_profit_eur: float = float("nan")

    @property
    def share_of_milp(self) -> float:
        if self.milp_profit_eur and abs(self.milp_profit_eur) > 1e-9:
            return self.eval_profit_eur / self.milp_profit_eur
        return float("nan")


@dataclass
class DAggerResult:
    iterations: List[DAggerIteration] = field(default_factory=list)
    requery_stride_hours: int = 24
    final_obs: Optional[np.ndarray] = None
    final_actions: Optional[np.ndarray] = None
    total_seconds: float = 0.0

    def summary_table(self) -> str:
        rows = ["| iter | beta | new | total | queries | BC loss | "
                "profit [M EUR] | share of MILP |",
                "|---|---|---|---|---|---|---|---|"]
        for it in self.iterations:
            share = ("n/a" if np.isnan(it.share_of_milp)
                     else f"{it.share_of_milp:.4f}")
            prof = ("n/a" if np.isnan(it.eval_profit_eur)
                    else f"{it.eval_profit_eur/1e6:.3f}")
            loss = ("n/a" if np.isnan(it.bc_final_loss)
                    else f"{it.bc_final_loss:.4f}")
            rows.append(f"| {it.iteration} | {it.beta:.2f} | {it.new_samples} "
                        f"| {it.total_samples} | {it.expert_queries} | {loss} "
                        f"| {prof} | {share} |")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Expert re-query
# ---------------------------------------------------------------------------

def solve_expert_from_state(
    fleet,
    prices_day: List[float],
    services_day: List[Any],
    soc: List[float],
    *,
    horizon: int = 24,
    start_hour: int = 0,
    directional_services: bool = True,
    sustain: SustainDurations = DEFAULT_SUSTAIN,
    coupling: FleetCoupling = FleetCoupling.uncoupled(),
    milp_mode: str = "discrete",
    use_nonlinear_degradation: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], str, float]:
    """Ask the MILP what to do from an ARBITRARY state of charge.

    This is the operation DAgger needs and behavioural cloning does not: the
    expert has to be answerable at the learner's state, not only along its own
    trajectory. Returns the discretised action sequence for hours
    [start_hour, start_hour + horizon), the solver status, and the objective.

    The state of charge is clamped into the MILP's feasible interior before
    solving. The environment allows SoC in [0, 1] while the MILP requires
    [soc_min, soc_max]; after stochastic activations the learner can sit a
    hair outside, and an infeasible expert query would silently become a
    zero-action label, which is worse than a slightly projected one.
    """
    from marl_milp_continuous import MultiBESSMILPOptimizer
    from marl_milp_discrete import MultiBESSMILPOptimizerDiscrete
    from marl_env import N_ACTION_BINS
    from action_projection import (_discretise_milp_action_directional,
                                   _discretise_milp_action)

    N = fleet.n_batteries
    soc_clamped = []
    for i in range(N):
        bp = fleet.batteries[i]
        lo, hi = bp.soc_min + 1e-4, bp.soc_max - 1e-4
        soc_clamped.append(float(np.clip(soc[i], lo, hi)))

    common = dict(
        use_nonlinear_degradation=use_nonlinear_degradation,
        directional_services=directional_services,
        coupling=coupling,
        **sustain.as_milp_kwargs(),
    )
    if milp_mode == "discrete":
        opt = MultiBESSMILPOptimizerDiscrete(
            fleet, None, n_action_bins=N_ACTION_BINS, **common)
    elif milp_mode == "continuous":
        opt = MultiBESSMILPOptimizer(fleet, None, **common)
    else:
        raise ValueError(f"milp_mode must be 'continuous' or 'discrete', "
                         f"got {milp_mode!r}")

    opt.current_soc = soc_clamped
    p = list(prices_day)[start_hour:start_hour + horizon]
    s = list(services_day)[start_hour:start_hour + horizon]
    result = opt.optimize(len(p), p, s)

    if result.solver_status != "Optimal":
        n_axes = 7 if directional_services else 5
        zeros = [{f"bess_{i}": np.zeros(n_axes, dtype=np.int64)
                  for i in range(N)} for _ in range(len(p))]
        return zeros, result.solver_status, 0.0

    seq: List[Dict[str, np.ndarray]] = []
    for t in range(len(p)):
        actions_t = {}
        for i in range(N):
            P_max = fleet.batteries[i].max_power_mw
            P_ch = opt.variables['P_charge'][(i, t)].varValue
            P_dis = opt.variables['P_discharge'][(i, t)].varValue
            R_fcr = opt.variables['R_fcr'][(i, t)].varValue
            if directional_services:
                actions_t[f"bess_{i}"] = _discretise_milp_action_directional(
                    P_ch, P_dis, R_fcr,
                    opt.variables['R_afrr_up'][(i, t)].varValue,
                    opt.variables['R_afrr_dn'][(i, t)].varValue,
                    opt.variables['R_mfrr_up'][(i, t)].varValue,
                    opt.variables['R_mfrr_dn'][(i, t)].varValue,
                    P_max,
                )
            else:
                actions_t[f"bess_{i}"] = _discretise_milp_action(
                    P_ch, P_dis, R_fcr,
                    opt.variables['R_afrr'][(i, t)].varValue,
                    opt.variables['R_mfrr'][(i, t)].varValue,
                    P_max,
                )
        seq.append(actions_t)
    return seq, result.solver_status, float(result.net_profit)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_labelled_batch(
    fleet,
    market_window,
    policy_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    *,
    n_days: int,
    beta: float = 0.0,
    # Episode length. 24 is the operating day; shorter values exist so the
    # mechanics can be exercised without paying for a full-day MILP solve on
    # a machine without a commercial solver.
    hours_per_day: int = 24,
    requery_stride_hours: int = 24,
    initial_soc: Optional[List[float]] = None,
    env_seed: int = 0,
    directional_services: bool = True,
    sustain: SustainDurations = DEFAULT_SUSTAIN,
    coupling: FleetCoupling = FleetCoupling.uncoupled(),
    duration_features: bool = True,
    centralized_critic: bool = False,
    milp_mode: str = "discrete",
    use_nonlinear_degradation: bool = False,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Roll the learner, label the states it visits with the expert's action.

    The state advances under whichever action is EXECUTED (the learner's, or
    the expert's with probability beta), while the recorded label is always
    the EXPERT's. That asymmetry is the whole point: the dataset ends up
    covering the learner's state distribution with correct actions on it.

    `policy_fn` of None means the expert drives, which reproduces the original
    demonstration generation and is the natural iteration 0.
    """
    from marl_env import MultiBESSEnv

    available = int(getattr(market_window, "n_days", n_days))
    if n_days > available:
        raise ValueError(
            f"n_days={n_days} exceeds the window's {available} days; slice or "
            f"extend the market window instead of over-reading it")

    rng = rng or np.random.default_rng(env_seed)
    N = fleet.n_batteries
    soc = list(initial_soc) if initial_soc else [0.5] * N
    fce = [0.0] * N
    cap_lost = [0.0] * N

    obs_all: List[np.ndarray] = []
    act_all: List[np.ndarray] = []
    n_queries = 0
    n_expert_steps = 0
    n_learner_steps = 0
    statuses: List[str] = []
    t0 = time.time()

    for day in range(n_days):
        prices_day, services_day = market_window.slice_day(day)
        prices_day, services_day = list(prices_day), list(services_day)

        env = MultiBESSEnv(
            fleet, episode_hours=hours_per_day,
            prices=prices_day[:hours_per_day],
            services=services_day[:hours_per_day],
            use_nonlinear_degradation=use_nonlinear_degradation,
            fce_cumulative_initial=fce, capacity_lost_initial=cap_lost,
            soc_init=0.5, seed=env_seed + day,
            directional_services=directional_services,
            sustain_hours=sustain, coupling=coupling,
            duration_features=duration_features,
        )
        env.reset(seed=env_seed + day)
        for i in range(N):
            env._socs[i] = float(soc[i])
        obs = {a: env._build_observation(a) for a in env.agents}

        expert_seq: List[Dict[str, np.ndarray]] = []
        next_requery = 0

        for t in range(hours_per_day):
            # Re-query the expert whenever the stride says so, always from the
            # state of charge the roll-out has actually reached.
            if t >= next_requery:
                seq, status, _ = solve_expert_from_state(
                    fleet, prices_day, services_day,
                    [float(env._socs[i]) for i in range(N)],
                    horizon=hours_per_day - t, start_hour=t,
                    directional_services=directional_services,
                    sustain=sustain, coupling=coupling, milp_mode=milp_mode,
                    use_nonlinear_degradation=use_nonlinear_degradation,
                )
                expert_seq = seq
                statuses.append(status)
                n_queries += 1
                next_requery = t + max(1, int(requery_stride_hours))
                seq_offset = t

            expert_t = expert_seq[t - seq_offset]

            # Record the state the roll-out is in, labelled with the expert.
            for a in env.agents:
                obs_all.append(np.asarray(obs[a], dtype=np.float32))
                act_all.append(np.asarray(expert_t[a], dtype=np.int64))

            # Decide what to EXECUTE.
            use_expert = (policy_fn is None) or (rng.random() < beta)
            if use_expert:
                exec_t = expert_t
                n_expert_steps += 1
            else:
                exec_t = {a: np.asarray(policy_fn(obs[a]), dtype=np.int64)
                          for a in env.agents}
                n_learner_steps += 1

            obs, _, term, trunc, _ = env.step(exec_t)
            if (isinstance(term, dict) and term.get("__all__")) or \
               (isinstance(trunc, dict) and trunc.get("__all__")):
                break

        for i in range(N):
            soc[i] = float(env._socs[i])
            fce[i] = float(env._lfp_states[i].fce_cumulative)
            cap_lost[i] = float(env._lfp_states[i].capacity_lost_fraction)

        if verbose and (day + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [dagger] day {day+1}/{n_days} | {el:.0f}s "
                  f"({el/(day+1):.1f}s/day) | {len(obs_all)} samples | "
                  f"{n_queries} expert queries", flush=True)

    info = {
        "expert_queries": n_queries,
        "expert_steps": n_expert_steps,
        "learner_steps": n_learner_steps,
        "n_optimal": sum(1 for s in statuses if s == "Optimal"),
        "n_queries_total": len(statuses),
        "seconds": time.time() - t0,
        "final_soc": list(soc),
    }
    return (np.asarray(obs_all, dtype=np.float32),
            np.asarray(act_all, dtype=np.int64), info)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_dagger(
    fleet,
    train_window,
    *,
    config: DAggerConfig = DAggerConfig(),
    n_train_days: int = 365,
    hours_per_day: int = 24,
    # The evaluation window governs its own length: evaluate_policy_multi_day
    # reads market_window.n_days rather than taking a day count, so slice the
    # window before passing it here if a shorter evaluation is wanted.
    eval_window=None,
    bc_epochs: int = 50,
    bc_seed: int = 0,
    env_seed: int = 0,
    directional_services: bool = True,
    sustain: SustainDurations = DEFAULT_SUSTAIN,
    coupling: FleetCoupling = FleetCoupling.uncoupled(),
    duration_features: bool = True,
    centralized_critic: bool = False,
    milp_mode: str = "discrete",
    use_nonlinear_degradation: bool = False,
    verbose: bool = True,
) -> DAggerResult:
    """Iterate collect -> label -> aggregate -> retrain.

    Iteration 0 is the plain behavioural clone: the expert drives, so the
    dataset is exactly what `generate_multi_day_expert_demos` produces. Every
    later iteration rolls the clone and labels ITS states, which is what
    closes the compounding-error gap.
    """
    from marl_bc import train_bc
    from marl_pipeline import make_bc_policy_fn, evaluate_policy_multi_day

    res = DAggerResult(requery_stride_hours=config.requery_stride_hours)
    t_start = time.time()
    obs_agg: Optional[np.ndarray] = None
    act_agg: Optional[np.ndarray] = None
    bc_net = None

    for it in range(config.n_iterations):
        beta = 1.0 if it == 0 else config.beta(it)
        policy_fn = None if bc_net is None else make_bc_policy_fn(
            bc_net, deterministic=True,
            centralized_critic=centralized_critic)

        if verbose:
            print(f"\n[dagger] iteration {it}: "
                  f"{'expert roll-out (plain BC)' if policy_fn is None else f'learner roll-out, beta={beta:.2f}'}",
                  flush=True)

        obs_new, act_new, info = collect_labelled_batch(
            fleet, train_window, policy_fn,
            n_days=n_train_days, beta=beta,
            requery_stride_hours=config.requery_stride_hours,
            hours_per_day=hours_per_day,
            env_seed=env_seed, directional_services=directional_services,
            sustain=sustain, coupling=coupling,
            duration_features=duration_features, milp_mode=milp_mode,
            use_nonlinear_degradation=use_nonlinear_degradation,
            verbose=verbose,
        )

        if obs_agg is None or not config.aggregate:
            obs_agg, act_agg = obs_new, act_new
        else:
            obs_agg = np.concatenate([obs_agg, obs_new], axis=0)
            act_agg = np.concatenate([act_agg, act_new], axis=0)

        if config.max_samples and len(obs_agg) > config.max_samples:
            # Keep the most recent samples: they are the ones drawn from the
            # current learner's distribution, which is what the next clone has
            # to be correct on.
            obs_agg = obs_agg[-config.max_samples:]
            act_agg = act_agg[-config.max_samples:]

        t_train = time.time()
        bc_net, hist = train_bc(obs_agg, act_agg, n_epochs=bc_epochs,
                                seed=bc_seed + it)
        train_s = time.time() - t_train

        rec = DAggerIteration(
            iteration=it, beta=beta, new_samples=int(len(obs_new)),
            total_samples=int(len(obs_agg)),
            expert_queries=int(info["expert_queries"]),
            collect_seconds=float(info["seconds"]), train_seconds=train_s,
        )
        # train_bc returns (net, history) with history a LIST of per-epoch
        # dicts. Reading it as a dict of lists raised TypeError, and a bare
        # except turned that into a silent NaN in the report, which is worse
        # than a crash: the table would have looked complete and been empty.
        if hist:
            last = hist[-1]
            rec.bc_final_loss = float(last.get("train_loss", float("nan")))
            acc = last.get("val_acc_per_axis")
            rec.bc_acc_per_axis = [float(x) for x in acc] if acc else []

        if eval_window is not None:
            ev = evaluate_policy_multi_day(
                fleet, eval_window,
                policy_fn=make_bc_policy_fn(
                    bc_net, deterministic=True,
                    centralized_critic=centralized_critic),
                env_seed=env_seed + 1000,
                directional_services=directional_services,
                sustain=sustain, coupling=coupling,
                duration_features=duration_features,
                use_nonlinear_degradation=use_nonlinear_degradation,
                centralized_critic=centralized_critic,
            )
            rec.eval_profit_eur = float(ev.total_profit_eur)

        res.iterations.append(rec)
        if verbose:
            print(f"[dagger] iteration {it}: {rec.new_samples} new samples, "
                  f"{rec.total_samples} total, "
                  f"BC loss {rec.bc_final_loss:.4f}, "
                  f"profit {rec.eval_profit_eur:,.0f} EUR", flush=True)

    res.final_obs, res.final_actions = obs_agg, act_agg
    res.total_seconds = time.time() - t_start
    return res


__all__ = [
    "DAggerConfig", "DAggerIteration", "DAggerResult",
    "solve_expert_from_state", "collect_labelled_batch", "run_dagger",
]
