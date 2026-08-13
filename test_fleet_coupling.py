# -*- coding: utf-8 -*-
"""Fleet-coupling regression tests: shared grid connection capacity.

Run:  python test_fleet_coupling.py

WHAT THESE LOCK DOWN
--------------------
The coupling has to hold in TWO places at once: the MILP plans against it and
the environment enforces it at roll-out. Twice already in this project a
constraint meant one thing to the optimizer and another to the environment
(tau_FCR defined in seven places with three live values; per-service sustain
caps that summed to three times the SoC budget), and both times the benchmark
and the policy were solving different problems without anyone noticing.

So the decisive test here is PARITY: a schedule the MILP considers feasible
must pass the environment's projection untouched. If that ever fails, the
comparison is broken no matter how good the numbers look.
"""

from __future__ import annotations

import sys

import numpy as np

from fleet_coupling import (FleetCoupling, direction_envelopes,
                            envelope_scales, project_to_connection)
from market_constants import DEFAULT_SUSTAIN
from milp_optimizer import BatteryParameters
from marl_milp_continuous import MultiBatteryParameters
from marl_env import MultiBESSEnv, N_ACTION_BINS

FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def _fleet(n=6, pmw=1.0, dur=2.0):
    bp = BatteryParameters(capacity_mwh=pmw * dur, max_power_mw=pmw,
                           degradation_cost_per_mwh=25000.0, cycle_life=8000)
    return MultiBatteryParameters([bp] * n)


def _env(fleet, coupling, soc=0.5):
    env = MultiBESSEnv(
        fleet, episode_hours=24, soc_init=soc, seed=0,
        directional_services=True, sustain_hours=DEFAULT_SUSTAIN,
        coupling=coupling,
    )
    env.reset(seed=0)
    return env


def _saturating(env):
    """All agents bid the top bin on every axis."""
    top = N_ACTION_BINS - 1
    return {a: np.full(env.action_space(a).shape, top, dtype=np.int64)
            for a in env.agents}


# ---------------------------------------------------------------------------
print("\n[1] envelope arithmetic")
# ---------------------------------------------------------------------------
s_up, s_dn = envelope_scales(up=10.0, dn=4.0, limit=5.0)
check(abs(s_up - 0.5) < 1e-12, "upward scale is limit/up when it binds")
check(s_dn == 1.0, "downward scale is 1 when it does not bind")

s_up, s_dn = envelope_scales(up=0.0, dn=0.0, limit=5.0)
check(s_up == 1.0 and s_dn == 1.0, "zero envelopes do not divide by zero")


# ---------------------------------------------------------------------------
print("\n[2] the environment enforces the connection limit")
# ---------------------------------------------------------------------------
fleet = _fleet(n=6, pmw=1.0)          # 6 MW of rated power
LIMIT = 2.5                            # deliberately far below the fleet

env_free = _env(fleet, FleetCoupling.uncoupled())
env_conn = _env(fleet, FleetCoupling.connection(LIMIT))

free = env_free._decode_clip_actions(_saturating(env_free), hour=0)
conn = env_conn._decode_clip_actions(_saturating(env_conn), hour=0)

up_f, dn_f = direction_envelopes(free)
up_c, dn_c = direction_envelopes(conn)
print(f"        uncoupled: up {up_f:.3f} MW, dn {dn_f:.3f} MW")
print(f"        coupled  : up {up_c:.3f} MW, dn {dn_c:.3f} MW  (limit {LIMIT})")

check(up_f > LIMIT or dn_f > LIMIT,
      "without coupling the saturating bid exceeds the connection")
check(up_c <= LIMIT + 1e-9, "upward envelope respects the connection limit")
check(dn_c <= LIMIT + 1e-9, "downward envelope respects the connection limit")
check(up_c > 0.0, "the projection does not simply zero the fleet")


# ---------------------------------------------------------------------------
print("\n[3] the projection scales down, never up, and keeps the mix")
# ---------------------------------------------------------------------------
a0 = env_conn.agents[0]
for k in ("P_charge", "P_discharge", "R_fcr", "R_afrr_up", "R_mfrr_dn"):
    check(conn[a0][k] <= free[a0][k] + 1e-9,
          f"{k} is not increased by the projection")

# proportional scaling preserves the ratio between two upward quantities
if free[a0]["R_afrr_up"] > 1e-9 and free[a0]["R_mfrr_up"] > 1e-9:
    r_free = free[a0]["R_afrr_up"] / free[a0]["R_mfrr_up"]
    r_conn = conn[a0]["R_afrr_up"] / max(conn[a0]["R_mfrr_up"], 1e-12)
    check(abs(r_free - r_conn) < 1e-6,
          "the upward bid mix is preserved (proportional scaling)")


# ---------------------------------------------------------------------------
print("\n[4] a bid within the limit passes through untouched")
# ---------------------------------------------------------------------------
small = {a: np.zeros(env_conn.action_space(a).shape, dtype=np.int64)
         for a in env_conn.agents}
# one unit only, one tenth of its rated power on the discharge axis
small[env_conn.agents[0]][1] = 1
before = env_conn._decode_clip_actions(small, hour=0)
up_s, dn_s = direction_envelopes(before)
check(up_s <= LIMIT and dn_s <= LIMIT, "the small bid is inside the envelope")
diag = project_to_connection({k: dict(v) for k, v in before.items()}, LIMIT)
check(diag["binding"] == 0.0, "the projection reports itself as non-binding")


# ---------------------------------------------------------------------------
print("\n[5] MILP/ENV PARITY: a MILP-feasible schedule survives the env")
# ---------------------------------------------------------------------------
from marl_milp_continuous import MultiBESSMILPOptimizer
from italian_market_data import make_synthetic_market_window
from datetime import datetime, timedelta

coupling = FleetCoupling.connection(LIMIT)
start = datetime(2024, 1, 1)
w = make_synthetic_market_window(start_date=start,
                                 end_date=start + timedelta(days=2), seed=0)
prices, services = w.slice_day(0)
prices, services = list(prices), list(services)

opt = MultiBESSMILPOptimizer(
    fleet, None, use_nonlinear_degradation=False,
    directional_services=True, coupling=coupling,
    **DEFAULT_SUSTAIN.as_milp_kwargs())
res = opt.optimize(24, prices, services)
print(f"        solver status: {res.solver_status}")

# power_trajectories[i][t] is the NET power: positive discharges, negative
# charges. flexibility_reservations is keyed by service name, each a
# (n_batteries, T) list of lists.
_RES = res.flexibility_reservations
_KEYMAP = {"R_fcr": "FCR", "R_afrr_up": "aFRR_up", "R_afrr_dn": "aFRR_dn",
           "R_mfrr_up": "mFRR_up", "R_mfrr_dn": "mFRR_dn"}

viol_milp = 0
viol_env = 0
for t in range(24):
    bid = {}
    for i in range(fleet.n_batteries):
        net = res.power_trajectories[i][t]
        b = {"P_discharge": max(0.0, net), "P_charge": max(0.0, -net)}
        for our_key, milp_key in _KEYMAP.items():
            b[our_key] = _RES[milp_key][i][t]
        bid[f"bess_{i}"] = b
    up, dn = direction_envelopes(bid)
    if up > LIMIT + 1e-6 or dn > LIMIT + 1e-6:
        viol_milp += 1
    # the environment's projection must leave a MILP-feasible bid alone
    replay = {k: dict(v) for k, v in bid.items()}
    d = project_to_connection(replay, LIMIT)
    if d["binding"] != 0.0:
        viol_env += 1

check(viol_milp == 0,
      "the MILP schedule respects the connection limit at all 24 hours")
check(viol_env == 0,
      "the env projection is a NO-OP on the MILP schedule (matched worlds)")


# ---------------------------------------------------------------------------
print("\n[5b] the tolerance is tight enough to still catch real violations")
# ---------------------------------------------------------------------------
# The tolerance exists for machine epsilon, not to let real violations
# through. A bid one part in ten thousand over the limit must still be caught.
probe = {"bess_0": {"P_discharge": LIMIT * 1.0001, "P_charge": 0.0,
                    "R_fcr": 0.0, "R_afrr_up": 0.0, "R_afrr_dn": 0.0,
                    "R_mfrr_up": 0.0, "R_mfrr_dn": 0.0}}
d = project_to_connection(probe, LIMIT)
check(d["binding"] == 1.0, "a 0.01% overshoot is still projected")
check(abs(probe["bess_0"]["P_discharge"] - LIMIT) < 1e-9,
      "the projection lands on the TRUE limit, not the tolerant one")

eps = {"bess_0": {"P_discharge": LIMIT + 4.4e-16, "P_charge": 0.0,
                  "R_fcr": 0.0, "R_afrr_up": 0.0, "R_afrr_dn": 0.0,
                  "R_mfrr_up": 0.0, "R_mfrr_dn": 0.0}}
d = project_to_connection(eps, LIMIT)
check(d["binding"] == 0.0, "machine-epsilon overshoot is NOT treated as binding")


# ---------------------------------------------------------------------------
print("\n[6] the constraint actually binds (it is not decorative)")
# ---------------------------------------------------------------------------
opt_free = MultiBESSMILPOptimizer(
    fleet, None, use_nonlinear_degradation=False,
    directional_services=True, coupling=FleetCoupling.uncoupled(),
    **DEFAULT_SUSTAIN.as_milp_kwargs())
res_free = opt_free.optimize(24, prices, services)
print(f"        MILP profit uncoupled: {res_free.net_profit:>12,.2f} EUR")
print(f"        MILP profit coupled  : {res.net_profit:>12,.2f} EUR")
check(res.net_profit < res_free.net_profit - 1e-6,
      "the connection limit costs the fleet profit, so it genuinely binds")


# ---------------------------------------------------------------------------
print("\n[7] uncoupled reproduces the previous behaviour exactly")
# ---------------------------------------------------------------------------
env_a = _env(fleet, FleetCoupling.uncoupled())
env_b = MultiBESSEnv(fleet, episode_hours=24, soc_init=0.5, seed=0,
                     directional_services=True, sustain_hours=DEFAULT_SUSTAIN)
env_b.reset(seed=0)
da = env_a._decode_clip_actions(_saturating(env_a), hour=0)
db = env_b._decode_clip_actions(_saturating(env_b), hour=0)
same = all(abs(da[a][k] - db[a][k]) < 1e-12
           for a in da for k in da[a])
check(same, "an explicit uncoupled setting equals the default (no regression)")


# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all fleet-coupling tests passed")
print("=" * 62)
