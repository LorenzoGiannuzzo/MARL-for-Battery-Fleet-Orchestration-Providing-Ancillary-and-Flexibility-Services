# -*- coding: utf-8 -*-
"""
commitments.py — Step A of the realistic market model: the commitment table.

CONTEXT (what this is and why)
------------------------------
The current MultiBESSEnv treats every hour as an isolated auction: you bid a
service at hour t, an award/activation dice is rolled in that same hour t, and
at hour t+1 everything resets — the reserved capacity is NOT frozen. That is
unrealistic. In a real balancing market the chain is:

    bid (ex-ante, day-ahead) -> probabilistic award -> if won, the capacity is
    LOCKED (frozen) on the battery for the delivery hour -> during delivery the
    service may be activated (stochastically) -> if activated and the battery
    cannot deliver (insufficient SoC), a non-delivery PENALTY applies.

This module implements the STATE that makes all of that possible: a per-agent
table of ACTIVE COMMITMENTS. Everything else (locking the power, drawing the
activation, charging the penalty) is just operations on this table, added in
Steps B/C/D inside the env.

DESIGN DECISIONS (fixed with Lorenzo, "day-ahead ROLLING" variant)
------------------------------------------------------------------
  * Bid timing (fase 1): DAY-AHEAD ROLLING. The agent decides at hour t a bid
    for DELIVERY at hour t + lead_time. The hourly step is preserved (7 action
    axes intact, BC/transfer untouched). The realism — being blind to the SoC
    you'll actually have at delivery — comes for free: when delivery hour
    arrives, the SoC is whatever the day turned into.
  * Award (fase 2): PROBABILISTIC, exogenous award_probability (already in the
    MSD-conditioned service profiles). No endogenous clearing price.
  * Lock duration (fase 3): HOURLY product (option C, configurable per service).
    The Italian MSD procures reserves per hourly period, so a commitment locks
    capacity for `lock_hours` delivery hours (default 1 = one delivery hour).
    NOTE the distinction we agreed on: the lock DURATION is hourly; the 4h/1h/2h
    sustain figures are a separate SoC QUALIFICATION requirement enforced
    elsewhere (the env's fcr/afrr/mfrr_sustain caps), not the lock length.
  * Activation (fase 4): drawn in the DELIVERY hour (not the bid hour), once
    per locked commitment that is active in that hour.
  * Penalty (fase 4): non-delivery costs `penalty_k * energy_price` per MWh of
    UNDELIVERED energy. We use penalty_k = 1.5 by default. Rationale: real
    balancing settlement penalises non-delivery at the imbalance price times an
    ex-ante coefficient k > 1 (e.g. RTE's k coefficient, set in M-3); k in
    [1, 2] is the common range in the literature. 1.5 is a defensible midpoint
    and is configurable.

This module has NO dependency on the env or RLlib — it is pure bookkeeping, so
it can be unit-tested on its own (see __main__ at the bottom for a smoke test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Default lock duration per service KEY, in delivery hours. Hourly product =>
# 1 hour. Exposed so a future ablation can lengthen it (e.g. toward 4h blocks
# or 15-min TIDE granularity) without touching the logic.
DEFAULT_LOCK_HOURS: Dict[str, int] = {
    "R_fcr": 1,
    "R_afrr": 1, "R_afrr_up": 1, "R_afrr_dn": 1,
    "R_mfrr": 1, "R_mfrr_up": 1, "R_mfrr_dn": 1,
}

# Default non-delivery penalty multiplier (k). Undelivered energy is charged
# penalty_k * energy_price. See module docstring for the rationale (k in [1,2],
# 1.5 midpoint).
DEFAULT_PENALTY_K: float = 1.5


@dataclass
class Commitment:
    """A single awarded market commitment held by one agent.

    A commitment is created when a bid placed at `bid_hour` for delivery at
    `delivery_hour` is AWARDED. From `delivery_hour` (inclusive) for
    `lock_hours` hours, `mw` of the battery's power is frozen toward `service`
    and may be activated.

    Fields
    ------
    service : str
        Service key, e.g. 'R_fcr', 'R_afrr_up'. Matches the env's action axes.
    mw : float
        Awarded reserved power in MW (already discretised/clipped at bid time).
    bid_hour : int
        Absolute hour index at which the bid was decided.
    delivery_hour : int
        Absolute hour index at which delivery (and possible activation) starts.
    lock_hours : int
        Number of consecutive delivery hours the capacity stays frozen.
    """
    service: str
    mw: float
    bid_hour: int
    delivery_hour: int
    lock_hours: int = 1

    def is_active_at(self, hour: int) -> bool:
        """True if this commitment freezes capacity during `hour`."""
        return self.delivery_hour <= hour < self.delivery_hour + self.lock_hours

    def is_expired_at(self, hour: int) -> bool:
        """True once `hour` is past the commitment's delivery window."""
        return hour >= self.delivery_hour + self.lock_hours


class CommitmentTable:
    """Per-agent ledger of active market commitments.

    Pure bookkeeping. The env will, each hour:
      1. expire() old commitments,
      2. read locked_power(agent, hour) to shrink available P_max BEFORE
         decoding the agent's arbitrage/new-bid action (Step B),
      3. add() new awarded commitments for future delivery (Step C),
      4. iterate active_commitments(agent, hour) to draw activation and apply
         SoC deltas / penalties (Step D).
    """

    def __init__(
        self,
        agents: List[str],
        lock_hours: Optional[Dict[str, int]] = None,
    ):
        self._lock_hours = dict(DEFAULT_LOCK_HOURS)
        if lock_hours:
            self._lock_hours.update(lock_hours)
        # agent -> list of Commitment
        self._table: Dict[str, List[Commitment]] = {a: [] for a in agents}

    # -- lifecycle -----------------------------------------------------------

    def reset(self, agents: List[str]) -> None:
        """Clear the table (call from env.reset)."""
        self._table = {a: [] for a in agents}

    def lock_hours_for(self, service: str) -> int:
        return int(self._lock_hours.get(service, 1))

    def add(
        self, agent: str, service: str, mw: float,
        bid_hour: int, delivery_hour: int,
    ) -> Optional[Commitment]:
        """Register a newly AWARDED commitment. Returns it, or None if mw<=0."""
        if mw <= 0.0:
            return None
        c = Commitment(
            service=service, mw=float(mw),
            bid_hour=int(bid_hour), delivery_hour=int(delivery_hour),
            lock_hours=self.lock_hours_for(service),
        )
        self._table[agent].append(c)
        return c

    def expire(self, hour: int) -> None:
        """Drop commitments whose delivery window has fully passed by `hour`."""
        for a in self._table:
            self._table[a] = [c for c in self._table[a]
                              if not c.is_expired_at(hour)]

    # -- queries -------------------------------------------------------------

    def active_commitments(self, agent: str, hour: int) -> List[Commitment]:
        """Commitments freezing capacity for `agent` during `hour`."""
        return [c for c in self._table[agent] if c.is_active_at(hour)]

    def locked_power(self, agent: str, hour: int) -> float:
        """Total MW frozen for `agent` at `hour` across all active commitments.

        This is what Step B subtracts from P_max before the agent may bid new
        capacity or do arbitrage in `hour`.
        """
        return float(sum(c.mw for c in self.active_commitments(agent, hour)))

    def locked_power_by_service(self, agent: str, hour: int) -> Dict[str, float]:
        """Per-service breakdown of frozen MW at `hour` (for diagnostics)."""
        out: Dict[str, float] = {}
        for c in self.active_commitments(agent, hour):
            out[c.service] = out.get(c.service, 0.0) + c.mw
        return out

    def total_active(self, hour: int) -> int:
        """Count of active commitments across all agents (diagnostics)."""
        return sum(len(self.active_commitments(a, hour)) for a in self._table)

    def snapshot(self) -> Dict[str, List[Commitment]]:
        """Shallow view of the whole table (diagnostics/tests)."""
        return {a: list(v) for a, v in self._table.items()}


# ---------------------------------------------------------------------------
# Standalone smoke test (Step A is testable in isolation, by design)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agents = ["bess_0", "bess_1"]
    ct = CommitmentTable(agents, lock_hours={"R_fcr": 1, "R_afrr_dn": 1})

    # bess_0 bids FCR at hour 5 for delivery at hour 29 (t + 24, day-ahead lead)
    ct.add("bess_0", "R_fcr", mw=2.0, bid_hour=5, delivery_hour=29)
    # bess_0 also bids aFRR_dn at hour 6 for delivery at hour 30
    ct.add("bess_0", "R_afrr_dn", mw=1.5, bid_hour=6, delivery_hour=30)

    assert ct.locked_power("bess_0", 29) == 2.0, "FCR should be locked at h29"
    assert ct.locked_power("bess_0", 30) == 1.5, "aFRR_dn locked at h30"
    assert ct.locked_power("bess_0", 28) == 0.0, "nothing locked before delivery"
    assert ct.locked_power("bess_1", 29) == 0.0, "other agent unaffected"

    # after hour 30 the FCR commitment (delivery 29, lock 1) is expired
    ct.expire(hour=31)
    assert ct.locked_power("bess_0", 29) == 0.0, "expired FCR cleared"
    assert ct.total_active(30) == 0, "all expired by h31"

    print("commitments.py smoke test: OK")
    print("  locked_power_by_service example:",
          CommitmentTable(agents).locked_power_by_service("bess_0", 0))