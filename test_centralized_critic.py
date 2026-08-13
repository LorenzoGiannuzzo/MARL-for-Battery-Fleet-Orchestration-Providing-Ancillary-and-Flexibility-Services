# -*- coding: utf-8 -*-
"""Centralised-critic tests: the global-state block (Reviewer 1, point 3).

Run:  python test_centralized_critic.py

SCOPE
-----
The environment side is fully testable and is what this file covers. The
RLModule that slices the block off the actor lives in
marl_centralized_critic.py and needs RLlib, so it is validated on the first
smoke run rather than here.

The properties pinned below are the ones that make the block usable:

  APPENDED, NOT OVERWRITING. The block sits at the end so every existing index
  keeps its meaning. An earlier version allocated the observation at the base
  width and wrote obs[-6:] into it, which silently destroyed the duration
  features instead of adding to them. That is exactly the failure this file
  exists to prevent recurring.

  IDENTICAL ACROSS AGENTS. It is the fleet's state, not the agent's. If it
  differed per agent it would not be a global state and the critic would be
  learning something else.

  N-INVARIANT. Six summary statistics rather than a concatenation of all
  agents' observations, so a critic trained at one fleet size can be evaluated
  at another and the scalability study does not need a network per size.

  CONTROLLED PAIR. With the block off, the observation is bit-identical to
  what it was before the feature existed, so on/off is a clean comparison.
"""

from __future__ import annotations

import sys

import numpy as np

from fleet_coupling import FleetCoupling
from market_constants import DEFAULT_SUSTAIN
from milp_optimizer import BatteryParameters
from marl_milp_continuous import MultiBatteryParameters
from marl_env import (MultiBESSEnv, OBS_DIM, GLOBAL_STATE_DIM,
                      GLOBAL_STATE_SCHEMA, N_ACTION_BINS)

FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def _fleet(n=6, dur=2.0):
    bp = BatteryParameters(capacity_mwh=1.0 * dur, max_power_mw=1.0,
                           degradation_cost_per_mwh=25000.0, cycle_life=8000)
    return MultiBatteryParameters([bp] * n)


def _env(fleet, cc, soc=0.5, limit=3.0):
    e = MultiBESSEnv(
        fleet, episode_hours=24, soc_init=soc, seed=0,
        directional_services=True, sustain_hours=DEFAULT_SUSTAIN,
        coupling=(FleetCoupling.connection(limit) if limit
                  else FleetCoupling.uncoupled()),
        centralized_critic=cc)
    e.reset(seed=0)
    return e


TOP = N_ACTION_BINS - 1


def _fcr(env):
    return {a: np.array([0, 0, TOP, 0, 0, 0, 0], dtype=np.int64)
            for a in env.agents}


# ---------------------------------------------------------------------------
print(f"\n[1] the block is {GLOBAL_STATE_DIM} features, appended at the end")
# ---------------------------------------------------------------------------
for k in GLOBAL_STATE_SCHEMA:
    print(f"        {k}")
check(len(GLOBAL_STATE_SCHEMA) == GLOBAL_STATE_DIM,
      "the schema documents exactly as many features as the block holds")

fleet = _fleet()
off = _env(fleet, cc=False)
on = _env(fleet, cc=True)
d_off = off.observation_space("bess_0").shape[0]
d_on = on.observation_space("bess_0").shape[0]
print(f"        observation width: {d_off} without, {d_on} with")
check(d_on == d_off + GLOBAL_STATE_DIM,
      "enabling the block widens the observation by exactly its size")


# ---------------------------------------------------------------------------
print("\n[2] APPENDED, not overwriting (the bug this file exists for)")
# ---------------------------------------------------------------------------
o_off = off._build_observation("bess_0")
o_on = on._build_observation("bess_0")
check(len(o_off) == d_off and len(o_on) == d_on,
      "the built observation matches the declared space in both cases")
check(np.allclose(o_off, o_on[:d_off]),
      "the first OBS_DIM features are bit-identical: a controlled pair")
print(f"        duration features (26-28) without block: "
      f"{np.round(o_off[26:29], 4)}")
print(f"        duration features (26-28) with block   : "
      f"{np.round(o_on[26:29], 4)}")
check(np.allclose(o_off[26:29], o_on[26:29]),
      "the duration features survive intact instead of being overwritten")


# ---------------------------------------------------------------------------
print("\n[3] IDENTICAL across agents: it is the fleet's state")
# ---------------------------------------------------------------------------
on.step(_fcr(on))
blocks = {a: on._build_observation(a)[-GLOBAL_STATE_DIM:] for a in on.agents}
ref = blocks[on.agents[0]]
check(all(np.allclose(blocks[a], ref) for a in on.agents),
      f"all {len(on.agents)} agents see the same global block")
print("        after one step:")
for k, v in zip(GLOBAL_STATE_SCHEMA, ref):
    print(f"          {k:>32}: {v:.4f}")


# ---------------------------------------------------------------------------
print("\n[4] N-INVARIANT: the width does not depend on fleet size")
# ---------------------------------------------------------------------------
widths = {}
for n in (3, 6, 20):
    e = _env(_fleet(n), cc=True)
    widths[n] = e.observation_space("bess_0").shape[0]
print(f"        widths by fleet size: {widths}")
check(len(set(widths.values())) == 1,
      "a critic trained at one fleet size can be evaluated at another")


# ---------------------------------------------------------------------------
print("\n[5] the block actually tracks the fleet")
# ---------------------------------------------------------------------------
# Mean state of charge must follow the fleet, and dispersion must be zero when
# every unit sits at the same level and positive when they do not.
lo = _env(_fleet(), cc=True, soc=0.3)
hi = _env(_fleet(), cc=True, soc=0.8)
g_lo = lo._build_observation("bess_0")[-GLOBAL_STATE_DIM:]
g_hi = hi._build_observation("bess_0")[-GLOBAL_STATE_DIM:]
check(abs(g_lo[2] - 0.3) < 1e-5 and abs(g_hi[2] - 0.8) < 1e-5,
      "fleet_mean_soc reports the fleet's mean state of charge")
check(abs(g_lo[3]) < 1e-6,
      "dispersion is zero when every unit sits at the same level")

mixed = _env(_fleet(), cc=True)
for i in range(mixed.n_agents):
    mixed._socs[i] = 0.2 + 0.1 * i
g_mix = mixed._build_observation("bess_0")[-GLOBAL_STATE_DIM:]
check(g_mix[3] > 0.05, "dispersion is positive when the units differ")
print(f"        dispersion: uniform {g_lo[3]:.4f}, spread {g_mix[3]:.4f}")

# Contention: a bidding fleet loads the connection, an idle one does not.
idle = _env(_fleet(), cc=True)
idle.step({a: np.zeros(7, dtype=np.int64) for a in idle.agents})
g_idle = idle._build_observation("bess_0")[-GLOBAL_STATE_DIM:]
print(f"        upward envelope: idle {g_idle[0]:.3f}, bidding {ref[0]:.3f}")
check(ref[0] > g_idle[0],
      "the envelope feature rises when the fleet competes for the connection")


# ---------------------------------------------------------------------------
print("\n[6] the block is bounded, so the observation space stays valid")
# ---------------------------------------------------------------------------
extremes = []
for soc in (0.0, 0.1, 0.5, 0.9, 1.0):
    e = _env(_fleet(), cc=True, soc=soc, limit=0.5)   # deliberately tiny limit
    e.step(_fcr(e))
    extremes.append(e._build_observation("bess_0")[-GLOBAL_STATE_DIM:])
arr = np.asarray(extremes)
print(f"        min {arr.min():.3f}, max {arr.max():.3f} "
      f"(observation space is [-2, 2])")
check(arr.min() >= -2.0 and arr.max() <= 2.0,
      "every global feature stays inside the declared observation bounds")


# ---------------------------------------------------------------------------
print("\n[6b] the actor's view: masked, not shortened")
# ---------------------------------------------------------------------------
# The first implementation SLICED the tail off before the policy head. The
# encoder is built from the declared observation space, so its first layer was
# sized for the full width and the shortened input raised
# "mat1 and mat2 shapes cannot be multiplied (1x29 and 35x512)".
# Masking keeps the width and still leaves the actor blind, because a constant
# carries no information. This reproduces the masking without needing RLlib.
full = on._build_observation("bess_0").copy()
masked = full.copy()
masked[-GLOBAL_STATE_DIM:] = 0.0
check(len(masked) == len(full),
      "the actor's view has the SAME width as the critic's")
check(np.allclose(masked[:-GLOBAL_STATE_DIM], full[:-GLOBAL_STATE_DIM]),
      "everything outside the global block is untouched")
check(np.allclose(masked[-GLOBAL_STATE_DIM:], 0.0),
      "the global block is zero in the actor's view")
check(not np.allclose(full[-GLOBAL_STATE_DIM:], 0.0),
      "the critic's view still carries the real fleet state")


# ---------------------------------------------------------------------------
print("\n[7] the toggle is reversible after construction")
# ---------------------------------------------------------------------------
e = _env(_fleet(), cc=False)
w0 = e.observation_space("bess_0").shape[0]
e.configure_centralized_critic(True)
w1 = e.observation_space("bess_0").shape[0]
e.configure_centralized_critic(False)
w2 = e.observation_space("bess_0").shape[0]
check(w1 == w0 + GLOBAL_STATE_DIM and w2 == w0,
      "configure_centralized_critic rebuilds the space both ways")
check(len(e._build_observation("bess_0")) == w2,
      "the built observation follows the rebuilt space")


# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all centralised-critic (environment side) tests passed")
print("=" * 62)
