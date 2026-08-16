# -*- coding: utf-8 -*-
"""Relative-position feature tests: can the fleet break its own symmetry?

Run:  python test_relative_features.py

WHY THESE EXIST
---------------
Measured on the full run with coupling and a centralised critic:

    MILP realised     7,278,644      100%
    clone (BC)        6,313,634     86.7%
    PPO (CTDE)        6,029,240     82.8%
    vf_explained_var      0.617

The critic works: it went from 0.018 without the global state to 0.617 with
it, so the value function has learned what the joint situation is worth. The
ACTOR still loses to a plain clone. The diagnosis is that it cannot express
what the critic can score: under a binding shared connection the optimal
allocation is asymmetric — some units at full band, the rest held — while a
shared policy conditioned only on its own absolute state makes two identical
units bid identically, and the projection then scales everyone down uniformly.

The global-state block cannot fix that, because it is the SAME vector for
every agent: it supports a collective response, not a differentiated one.

These features are each unit's rank within the fleet. The properties below are
what make them capable of breaking the symmetry, and what stops them from
leaking fleet-wide state into the actor by accident.
"""

from __future__ import annotations

import sys

import numpy as np

from fleet_coupling import FleetCoupling
from market_constants import DEFAULT_SUSTAIN
from milp_optimizer import BatteryParameters
from marl_milp_continuous import MultiBatteryParameters
from marl_env import (MultiBESSEnv, OBS_DIM, GLOBAL_STATE_DIM,
                      RELATIVE_STATE_DIM, RELATIVE_STATE_SCHEMA,
                      mask_global_block)

FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def _uniform_fleet(n=8, dur=2.0):
    """Identical units: the hard case, where only the state can differ."""
    bp = BatteryParameters(capacity_mwh=1.0 * dur, max_power_mw=1.0,
                           degradation_cost_per_mwh=25000.0, cycle_life=8000)
    return MultiBatteryParameters([bp] * n)


def _env(fleet, rel, cc, soc=0.5):
    e = MultiBESSEnv(
        fleet, episode_hours=24, soc_init=soc, seed=0,
        directional_services=True, sustain_hours=DEFAULT_SUSTAIN,
        coupling=FleetCoupling.connection(3.0),
        relative_features=rel, centralized_critic=cc)
    e.reset(seed=0)
    return e


def _rel_block(env, agent):
    o = env._build_observation(agent)
    tail = env._tail_width()
    base = len(o) - tail
    return o[base:base + RELATIVE_STATE_DIM]


# ---------------------------------------------------------------------------
print(f"\n[1] the block is {RELATIVE_STATE_DIM} features")
# ---------------------------------------------------------------------------
for k in RELATIVE_STATE_SCHEMA:
    print(f"        {k}")
check(len(RELATIVE_STATE_SCHEMA) == RELATIVE_STATE_DIM,
      "the schema documents exactly as many features as the block holds")

fleet = _uniform_fleet()
off = _env(fleet, rel=False, cc=False)
on = _env(fleet, rel=True, cc=False)
check(on.observation_space("bess_0").shape[0]
      == off.observation_space("bess_0").shape[0] + RELATIVE_STATE_DIM,
      "enabling the block widens the observation by exactly its size")
check(np.allclose(off._build_observation("bess_0"),
                  on._build_observation("bess_0")[:OBS_DIM]),
      "the existing features are untouched: on/off is a controlled pair")


# ---------------------------------------------------------------------------
print("\n[2] ORDER: relative first, global last")
# ---------------------------------------------------------------------------
# The global block must stay last, because mask_global_block zeroes the final
# GLOBAL_STATE_DIM entries. If the relative block were appended after it, the
# mask would erase the wrong features and do so silently.
both = _env(fleet, rel=True, cc=True)
o = both._build_observation("bess_0")
check(len(o) == OBS_DIM + RELATIVE_STATE_DIM + GLOBAL_STATE_DIM,
      "with both blocks the width is base + relative + global")
masked = mask_global_block(o, True)
check(np.allclose(masked[-GLOBAL_STATE_DIM:], 0.0),
      "masking zeroes the global block")
check(np.allclose(masked[-(GLOBAL_STATE_DIM + RELATIVE_STATE_DIM):
                         -GLOBAL_STATE_DIM],
                  o[-(GLOBAL_STATE_DIM + RELATIVE_STATE_DIM):
                    -GLOBAL_STATE_DIM]),
      "masking leaves the relative block INTACT: the actor may read its own rank")


# ---------------------------------------------------------------------------
print("\n[3] the block DIFFERS per agent once states diverge")
# ---------------------------------------------------------------------------
# This is the whole point. The global block is identical for everyone; if this
# one were too, it could not break any symmetry.
env = _env(_uniform_fleet(), rel=True, cc=True)
at_start = np.array([_rel_block(env, a) for a in env.agents])
print(f"        identical units at the same SoC: "
      f"{len(set(map(tuple, np.round(at_start, 6))))} distinct rows of "
      f"{env.n_agents}")
check(len(set(map(tuple, np.round(at_start, 6)))) == 1,
      "identical units in an identical state tie, as they must "
      "(an arbitrary index would fake an asymmetry the state does not hold)")

rng = np.random.default_rng(0)
for i in range(env.n_agents):
    env._socs[i] = float(rng.uniform(0.15, 0.85))
after = np.array([_rel_block(env, a) for a in env.agents])
n_distinct = len(set(map(tuple, np.round(after, 6))))
print(f"        after the states diverge      : {n_distinct} distinct rows")
check(n_distinct == env.n_agents,
      "every unit carries a distinct rank once the states differ")


# ---------------------------------------------------------------------------
print("\n[4] the ranks are ranks: ordering and bounds")
# ---------------------------------------------------------------------------
socs = [float(env._socs[i]) for i in range(env.n_agents)]
order_soc = np.argsort(socs)
order_pct = np.argsort(after[:, 0])          # usable discharge percentile
check(np.array_equal(order_soc, order_pct),
      "on a uniform fleet the usable-energy rank orders exactly like SoC")
check(after.min() >= 0.0 and after.max() <= 1.0,
      "every percentile lies in [0, 1]")
check(abs(after[:, 0].mean() - 0.5) < 0.05,
      "the mean rank is one half, as a percentile must be")

# headroom is the complement of usable energy on a uniform fleet
check(np.corrcoef(after[:, 0], after[:, 1])[0, 1] < -0.9,
      "the charge-headroom rank is the mirror of the discharge one")


# ---------------------------------------------------------------------------
print("\n[5] N-INVARIANT: the width does not depend on fleet size")
# ---------------------------------------------------------------------------
widths = {n: _env(_uniform_fleet(n), rel=True, cc=True)
          .observation_space("bess_0").shape[0] for n in (3, 8, 20)}
print(f"        widths by fleet size: {widths}")
check(len(set(widths.values())) == 1,
      "a policy trained at one fleet size can be evaluated at another")


# ---------------------------------------------------------------------------
print("\n[6] heterogeneous durations separate the clusters at step zero")
# ---------------------------------------------------------------------------
# The same state of charge buys different HOURS on a 1 h and a 4 h unit, so on
# a heterogeneous fleet the ranks differ between clusters immediately, before
# any action is taken.
het = MultiBatteryParameters.heterogeneous_fleet(seed=0, durations=(4., 2., 1.))
e_het = MultiBESSEnv(
    het, episode_hours=24, soc_init=0.5, seed=0, directional_services=True,
    sustain_hours=DEFAULT_SUSTAIN, coupling=FleetCoupling.connection(33.0),
    relative_features=True, centralized_critic=True)
e_het.reset(seed=0)
by_cluster = {}
for i, a in enumerate(e_het.agents):
    by_cluster.setdefault(het.cluster_of(i), []).append(_rel_block(e_het, a)[0])
for c, vals in by_cluster.items():
    print(f"        {c:>11}: usable-energy rank {np.mean(vals):.3f}")
check(len({round(np.mean(v), 4) for v in by_cluster.values()}) == 3,
      "the three clusters occupy three distinct ranks from step zero")


# ---------------------------------------------------------------------------
print("\n[7] the toggle is reversible")
# ---------------------------------------------------------------------------
e = _env(_uniform_fleet(), rel=False, cc=False)
w0 = e.observation_space("bess_0").shape[0]
e.configure_relative_features(True)
w1 = e.observation_space("bess_0").shape[0]
e.configure_relative_features(False)
w2 = e.observation_space("bess_0").shape[0]
check(w1 == w0 + RELATIVE_STATE_DIM and w2 == w0,
      "configure_relative_features rebuilds the space both ways")
check(len(e._build_observation("bess_0")) == w2,
      "the built observation follows the rebuilt space")


# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all relative-position tests passed")
print("=" * 62)
