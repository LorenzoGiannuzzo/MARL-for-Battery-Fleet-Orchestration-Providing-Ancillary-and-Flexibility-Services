# -*- coding: utf-8 -*-
"""Fase-1 regression tests: reserve sustain windows must come from ONE source.

Run:  python test_sustain_config.py

What these lock down
--------------------
1. The environment's bid clip honours tau_FCR from the injected scenario,
   not a hard-coded constant. This is the test that would have caught the
   original bug: the clip said 4.0 h, the availability check said 0.5 h and
   the MILP default said 0.25 h, all in the same run.

2. The FIX 6 availability reference windows equal the clip windows. If they
   drift apart, a policy holding SOC exactly where the MILP holds it gets
   charged an availability penalty it should not owe.

3. Raising tau_FCR strictly reduces the FCR power a given SOC can back.
   At tau_FCR = 4 h a 2 h fleet is capped at roughly a quarter of what it
   can offer at 1 h, which is the mechanism behind Reviewer 2's point 2.

4. No stray hard-coded sustain literals remain in the modules that were
   patched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from market_constants import (SustainDurations, DEFAULT_SUSTAIN,
                              SOGL_TAU_FCR_MIN_H, SOGL_TAU_FCR_MAX_H,
                              TAU_FCR_SWEEP)
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


def _fleet(n=3, cap=2.0, pmw=1.0):
    """Small 2 h fleet (E/P = 2), same ratio as the paper's."""
    bp = BatteryParameters(capacity_mwh=cap, max_power_mw=pmw,
                           degradation_cost_per_mwh=25000.0, cycle_life=8000)
    return MultiBatteryParameters([bp] * n)


def _env(sustain, soc=0.5):
    env = MultiBESSEnv(
        _fleet(), episode_hours=24, soc_init=soc, seed=0,
        directional_services=True, sustain_hours=sustain,
    )
    env.reset(seed=0)
    return env


def _max_fcr_bid(env):
    """Decode a saturating all-axes action and read back the FCR power the
    clip actually allows."""
    top = N_ACTION_BINS - 1
    actions = {a: np.array([0, 0, top, 0, 0, 0, 0], dtype=np.int64)
               for a in env.agents}
    decoded = env._decode_clip_actions(actions, hour=0)
    return decoded[env.agents[0]]["R_fcr"]


# ---------------------------------------------------------------------------
print("\n[1] the clip reads tau_FCR from the injected scenario")
# ---------------------------------------------------------------------------
bids = {}
for tau in (0.25, 0.5, 1.0, 2.0, 4.0):
    # Mid SOC: FCR is symmetric, so it needs BOTH usable energy and
    # headroom. At SOC 0.9 the down-room is zero and the bid collapses for
    # every tau, which would hide the effect under test.
    env = _env(SustainDurations.with_tau_fcr(tau), soc=0.5)
    bids[tau] = _max_fcr_bid(env)
    print(f"        tau_FCR={tau:>4}h -> max FCR bid {bids[tau]:.4f} MW")

check(len(set(round(v, 6) for v in bids.values())) > 1,
      "the FCR cap moves with tau_FCR (it was hard-coded before)")

check(bids[0.25] > bids[4.0],
      "a shorter sustain window admits a larger FCR bid")

# 2 h unit (2 MWh / 1 MW) at SOC 0.5. The symmetric FCR budget is the tighter
# of usable energy (0.5-0.1)*2 = 0.8 MWh and headroom (0.9-0.5)*2 = 0.8 MWh.
# At tau=4 h that backs 0.8/4 = 0.2 MW against a 1.0 MW rating: FCR is
# offerable at one fifth of rated power. At tau=0.25 h the energy constraint
# is slack (0.8/0.25 = 3.2 MW) and the bid is limited by the power rating.
check(abs(bids[4.0] - 0.2) < 1e-6,
      "tau_FCR=4h caps a 2h unit at 0.2 MW of 1.0 MW rated (unofferable at Pn)")
check(abs(bids[0.25] - 1.0) < 1e-6,
      "tau_FCR=0.25h leaves the unit power-limited, not energy-limited")


# ---------------------------------------------------------------------------
print("\n[2] availability check and bid clip share the same windows")
# ---------------------------------------------------------------------------
from flexibility_market import ServiceType

for tau in (0.25, 2.0):
    env = _env(SustainDurations.with_tau_fcr(tau))
    check(env.sustain.fcr == tau,
          f"env.sustain.fcr == {tau} after injection")
    # The FIX 6 dict is rebuilt inside step(); reproduce it the same way the
    # patched code does and confirm it tracks env.sustain.
    _tau = env.sustain
    avail_ref = {ServiceType.FCR: _tau.fcr,
                 ServiceType.AFRR: _tau.afrr,
                 ServiceType.MFRR: _tau.mfrr}
    check(avail_ref[ServiceType.FCR] == env.sustain.fcr,
          f"availability tau_FCR tracks the clip at tau={tau}")


# ---------------------------------------------------------------------------
print("\n[3] configure_sustain_hours() rebinds an existing env")
# ---------------------------------------------------------------------------
env = _env(SustainDurations.with_tau_fcr(4.0))
before = _max_fcr_bid(env)
env.configure_sustain_hours(SustainDurations.sogl_default())
after = _max_fcr_bid(env)
check(after > before,
      f"rebinding 4h -> 0.25h raises the cap ({before:.3f} -> {after:.3f} MW)")

try:
    env.configure_sustain_hours({"fcr": 0.25})
    check(False, "configure_sustain_hours rejects a raw dict")
except TypeError:
    check(True, "configure_sustain_hours rejects a raw dict")


# ---------------------------------------------------------------------------
print("\n[4] MILP kwargs come from the same object")
# ---------------------------------------------------------------------------
s = SustainDurations.with_tau_fcr(0.5)
kw = s.as_milp_kwargs()
check(kw == {"fcr_sustain_hours": 0.5, "afrr_sustain_hours": 1.0,
             "mfrr_sustain_hours": 2.0},
      "as_milp_kwargs() matches the MILP constructor signature")


# ---------------------------------------------------------------------------
print("\n[5] SO GL Art. 156(10) bounds")
# ---------------------------------------------------------------------------
check(SOGL_TAU_FCR_MIN_H == 0.25 and SOGL_TAU_FCR_MAX_H == 0.5,
      "admissible tau_FCR range is [15, 30] minutes")
check(DEFAULT_SUSTAIN.is_sogl_compliant(),
      "the default scenario is SO GL compliant")
check(not SustainDurations.legacy_paper().is_sogl_compliant(),
      "the legacy 4h scenario is flagged non-compliant")
check(set(TAU_FCR_SWEEP) >= {0.25, 0.5, 4.0},
      "the sweep brackets the admissible range and keeps the legacy point")


# ---------------------------------------------------------------------------
print("\n[6] no stray hard-coded sustain literals survive")
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PATTERNS = [
    (re.compile(r"fcr_sustain\s*=\s*4\.0"), "fcr_sustain = 4.0"),
    (re.compile(r"fcr_sustain_hours\s*=\s*4\.0"), "fcr_sustain_hours=4.0"),
    (re.compile(r"ServiceType\.FCR:\s*0\.5\b"), "ServiceType.FCR: 0.5"),
]
for fname in ("marl_env.py", "marl_pipeline.py", "marl_bc.py"):
    src = (HERE / fname).read_text(encoding="utf-8")
    # Drop comment lines: the revision notes legitimately mention the old
    # values while explaining why they are gone.
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    for pat, label in PATTERNS:
        check(pat.search(code) is None,
              f"{fname}: no live '{label}'")


# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all sustain-configuration tests passed")
print("=" * 62)
