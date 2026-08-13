"""
Multi-BESS PettingZoo ParallelEnv for BSP-aggregated MAPPO training.

Step 5 of the multi-agent track. This module exposes a fleet of N BESS
units to N homogeneous PPO agents under PettingZoo's ParallelEnv API.
Each agent owns ONE BESS and bids independently into the Italian energy
market (PUN arbitrage) and the three ancillary services (FCR, aFRR,
mFRR). A single BSP aggregator collects the bids and the env enforces
ARERA's 1-MW per-service aggregate minimum: services whose aggregate bid
falls short pay zero revenue to all participants.

The environment is designed so that:

  - Every observation an agent sees is LOCAL (its own BESS state + the
    market state visible to all). The aggregate state is not exposed,
    so the same policy can be deployed at any fleet size without retraining
    on global structure. The CENTRALIZED critic in MAPPO (Step 6) will
    augment this with full-fleet state at training time only.

  - The reward is local: an agent's reward equals its own arbitrage
    revenue plus its proportional share of awarded flex revenue minus
    its own degradation cost. Summing over agents yields the fleet net
    profit, so credit assignment is exact at the team level by construction.

  - The same physical model as the MILP is used: discrete 1-hour SOC
    dynamics with round-trip efficiency; the same flexibility services;
    the same award and activation probability draws. This is intentional
    so that the DRL trained here can be compared head-to-head against the
    MILP from Step 4 on identical scenarios.

  - The non-linear LFP degradation model from Step 3 is integrated through
    the `use_nonlinear_degradation` flag. When enabled, each step's
    degradation cost is computed by advancing per-BESS LFPBatteryState
    instances with the realised charge/discharge/DoD/SOC of that hour,
    using the FULL Schmalstieg+Naumann formulas (including the 4th-power
    DoD stress factor and the SOC-dependent calendar aging). This is the
    physical fidelity advantage the DRL has over the MILP - the env can
    feed the agent gradients through the true non-linear cost surface
    that the MILP's linear approximation cannot represent.

The MultiBESSEnv conforms to PettingZoo 1.24+ ParallelEnv API:

    env = MultiBESSEnv(...)
    obs, infos = env.reset(seed=0)
    while env.agents:
        actions = {agent: policy(obs[agent]) for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from milp_optimizer import BatteryParameters
from marl_milp_continuous import MultiBatteryParameters
from flexibility_market import FlexibilityService, ServiceType
from degradation_model import LFPBatteryState, LFPDegradationParameters
from market_constants import SustainDurations, DEFAULT_SUSTAIN
from fleet_coupling import FleetCoupling, project_to_connection


# ============================================================================
# Action discretisation
# ============================================================================

# Each per-agent action axis is a fraction of the BESS's max_power, in
# 11 bins from 0.0 (no power) to 1.0 (full power). Five axes:
#   0: charge fraction        (0..1)
#   1: discharge fraction     (0..1)
#   2: FCR capacity bid frac  (0..1)
#   3: aFRR capacity bid frac (0..1)
#   4: mFRR capacity bid frac (0..1)
N_ACTION_BINS = 11
ACTION_AXES = 5
ACTION_AXES_DIRECTIONAL = 7  # [charge, discharge, FCR, aFRR_up, aFRR_dn, mFRR_up, mFRR_dn]
_BIN_VALUES = np.linspace(0.0, 1.0, N_ACTION_BINS)  # [0.0, 0.1, ..., 1.0]

# One-time per-process announce flag for the FIX A log line.
_JOINT_SUSTAIN_ANNOUNCED = False


def _decode_action_to_fractions(action: np.ndarray) -> np.ndarray:
    """Convert MultiDiscrete action into a length-N array of fractions in [0,1]."""
    return _BIN_VALUES[np.asarray(action, dtype=np.int64)]


# Mapping between env action-axis service KEYS ('R_fcr', ...) and ServiceType,
# used by the commitment path (Steps C/D) to register and settle commitments.
_KEY_TO_ST = {
    'R_fcr': ServiceType.FCR,
    'R_afrr': ServiceType.AFRR, 'R_mfrr': ServiceType.MFRR,
    'R_afrr_up': ServiceType.AFRR_UP, 'R_afrr_dn': ServiceType.AFRR_DN,
    'R_mfrr_up': ServiceType.MFRR_UP, 'R_mfrr_dn': ServiceType.MFRR_DN,
}
_ST_TO_KEY = {v: k for k, v in _KEY_TO_ST.items()}

# Fixed ordering of the 5 directional service axes, used to lay out the
# pending-commitment observation features (obs[21:26]) deterministically.
# These are the actual flexibility services (charge/discharge are arbitrage,
# not services, so they are excluded).
_ORDERED_SERVICE_KEYS = [
    'R_fcr', 'R_afrr_up', 'R_afrr_dn', 'R_mfrr_up', 'R_mfrr_dn',
]


def _service_key_for(st: "ServiceType") -> str:
    return _ST_TO_KEY[st]


def _service_type_for(key: str) -> "ServiceType":
    return _KEY_TO_ST[key]


# ============================================================================
# Observation packing
# ============================================================================

# 26 core + commitment features, plus 3 duration-relative features (26-28).
#
# The three added slots exist because a shared policy cannot behave correctly
# across assets of different storage duration when duration is only implicit
# in the state. The sustain constraint the environment enforces is
#
#     R_s * tau_s <= (SOC - SOC_min) * capacity
#
# and dividing through by P_max turns the left side into exactly the fraction
# the policy emits (actions are already relative to each unit's rated power)
# and the right side into usable energy expressed in HOURS AT RATED POWER.
# In those coordinates the constraint has the same form for every asset, so a
# single policy can generalise across a continuum of durations instead of
# memorising three clusters. Features 26-28 supply those quantities directly
# rather than leaving the network to divide capacity_mwh by max_power_mw on
# its own — a division it had no reason to learn while every unit shared an
# energy-to-power ratio of 2 h.
#
# The slots are always present so that enabling and disabling the information
# leaves the network shape and parameter count untouched: the ablation then
# varies information content only. See configure_duration_features.
OBS_DIM = 29  # see _build_observation for the schema
# 21 base features + 5 pending-commitment features (one per directional service:
# FCR, aFRR up/dn, mFRR up/dn). The 5 extra features expose how much capacity the
# agent has ALREADY committed for delivery in the near future, per service. This
# is the bridge that lets the PPO connect a day-ahead bid to its eventual
# activation: the commitment persists in the observation between bid and
# delivery, restoring the Markov property that the 24h lead otherwise breaks.
# With commitments disabled these 5 features are always zero (legacy behaviour
# preserved bit-for-bit except for the larger, zero-padded observation vector).

# ---- Full-foresight (diagnostic) observation ----
# When full_foresight=True the agent is given EVERYTHING the MILP sees: the true
# PUN prices and the true service (cap_price, energy_price, award_prob) for ALL
# 24 hours of the day, not just the current hour + 6h forecast. This removes the
# PPO's information disadvantage entirely, turning it into a perfect-foresight
# agent like the MILP. It is a DIAGNOSTIC test ("how close does the DRL get to
# the MILP at equal information?"), NOT a deployable configuration. OBS grows by
# 24 PUN + 24*5*3 service features = 24 + 360 = 384 extra dims.
_FF_HOURS = 24
_FF_N_SERVICES = 5  # FCR, aFRR_up, aFRR_dn, mFRR_up, mFRR_dn
OBS_DIM_FULL_FORESIGHT = OBS_DIM + _FF_HOURS + _FF_HOURS * _FF_N_SERVICES * 3  # 26 + 384 = 410


# ---------------------------------------------------------------------------
# Centralised-critic global state (CTDE)
# ---------------------------------------------------------------------------
# Reviewer 1, point 3: the submission "does not describe a centralised critic
# or another explicit credit-assignment mechanism, making the implementation
# closer to parameter-shared independent PPO than to a fully evaluated
# centralised-training and decentralised-execution architecture."
#
# A centralised critic needs to see what makes the JOINT value differ from the
# sum of the individual ones. Here that is the contention for the shared grid
# connection: with the coupling active, one unit's admissible bid depends on
# what the others bid, and a critic that only sees its own unit cannot
# represent that.
#
# APPENDED AT THE END, deliberately. Every existing index keeps its meaning,
# the full-foresight block keeps its offsets, and the actor's slice is simply
# "everything except the tail". Inserting in the middle would have renumbered
# the duration features and the foresight block for no benefit.
#
# N-INVARIANT BY CONSTRUCTION. The obvious global state, the concatenation of
# all agents' observations, has a dimension that depends on fleet size, so a
# critic trained at N=50 could not be evaluated at N=100 and the scalability
# study would need a different network per fleet size. These six summary
# statistics describe the same contention at any N.
GLOBAL_STATE_DIM = 6
def mask_global_block(obs, has_block: bool):
    """Return the observation as a DECENTRALISED policy must see it.

    The global block is the fleet's state. A centralised critic reads it; an
    actor deployed on a single unit cannot, and neither can the behavioural
    clone, which is a decentralised policy by definition.

    Getting this wrong is not loud. With the block left in, the clone trains
    on 35 features and learns to use all of them, then its weights are
    transferred into an actor that masks the last six: verify_transfer fell
    from 1.000 to 0.604, meaning the warm start no longer starts where the
    clone was. The clone also becomes a stronger baseline than it should be,
    since it competes using information the policy it is compared against
    does not have.

    Masking rather than truncating, for the same reason as in
    marl_centralized_critic: the width stays constant, so no network needs
    resizing and nothing downstream has to know whether the block is there.

    `has_block` must be stated, not inferred. Guessing from the width was the
    first attempt and it is wrong: 29 and 35 are both valid observation widths
    depending on the flag, so a 29-wide vector without a block had its last six
    real features zeroed. The caller always knows the flag; asking for it costs
    a keyword and removes the guess.
    """
    import numpy as _np
    a = _np.asarray(obs, dtype=_np.float32)
    if not has_block or a.shape[-1] <= GLOBAL_STATE_DIM:
        return a
    out = a.copy()
    out[..., -GLOBAL_STATE_DIM:] = 0.0
    return out


GLOBAL_STATE_SCHEMA = (
    "fleet_up_envelope_frac",    # committed upward power / connection limit
    "fleet_dn_envelope_frac",    # committed downward power / connection limit
    "fleet_mean_soc",
    "fleet_soc_dispersion",      # std of SoC across units
    "frac_units_energy_limited_up",
    "frac_units_energy_limited_dn",
)


# ============================================================================
# Environment
# ============================================================================

class MultiBESSEnv(ParallelEnv):
    """Multi-BESS BSP aggregator under PettingZoo ParallelEnv API.

    Episodes are typically 24 hours (1 day) or 168 hours (1 week). Each
    hour all agents act simultaneously; the env aggregates the bids, draws
    the random outcomes (award and activation), applies SOC dynamics, and
    returns per-agent rewards plus next observation.

    Parameters
    ----------
    multi_params : MultiBatteryParameters
        Fleet definition (sizing, cycle life, efficiency per BESS).
    episode_hours : int
        Number of hours per episode.
    prices : list of float, optional
        Energy prices per hour, length episode_hours. If None, a synthetic
        daily curve is generated at reset (useful for tests).
    services : list of list of FlexibilityService, optional
        Per-hour flexibility service catalog, length episode_hours. If
        None, a constant catalog is used.
    use_nonlinear_degradation : bool
        If True, per-step degradation cost uses the full Schmalstieg+
        Naumann LFP physical model (DoD-dependent, SOC-dependent calendar
        aging integrated step by step). If False, the legacy amortised
        rate from BatteryParameters is used (matches the linear MILP).
    nonlinear_replacement_cost : float
        EUR per BESS replacement at end of life, used to monetise the
        physical capacity-loss fraction. Only used in non-linear mode.
    nonlinear_temperature_K : float
        Ambient temperature used for calendar aging. Default 298.15 K.
    soc_init : float
        Initial SOC for every BESS at reset. Default 0.5.
    soc_min : float
        Lower SOC bound; charge/discharge actions are clipped to respect it.
    soc_max : float
        Upper SOC bound; charge/discharge actions are clipped to respect it.
    seed : int, optional
        Default seed for the env's RNG (used for award/activation draws).
    """

    metadata = {"render_modes": ["human"], "name": "multi_bess_v0"}

    def __init__(
        self,
        multi_params: MultiBatteryParameters,
        episode_hours: int = 24,
        prices: Optional[List[float]] = None,
        services: Optional[List[List[FlexibilityService]]] = None,
        use_nonlinear_degradation: bool = False,
        nonlinear_replacement_cost: float = 200000.0,
        nonlinear_temperature_K: float = 298.15,
        soc_init: float = 0.5,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        fce_cumulative_initial: Optional[List[float]] = None,
        seed: Optional[int] = None,
        directional_services: bool = False,
        capacity_lost_initial: Optional[List[float]] = None,
        sustain_hours: Optional[SustainDurations] = None,
        duration_features: bool = True,
        coupling: Optional[FleetCoupling] = None,
        centralized_critic: bool = False,
    ):
        super().__init__()
        self.multi_params = multi_params
        self.n_agents = multi_params.n_batteries
        self.episode_hours = int(episode_hours)
        self._fixed_prices = prices
        self._fixed_services = services
        self.use_nonlinear_degradation = use_nonlinear_degradation
        self.nonlinear_replacement_cost = float(nonlinear_replacement_cost)
        self.nonlinear_temperature_K = float(nonlinear_temperature_K)
        self.soc_init = float(soc_init)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.directional_services = bool(directional_services)
        # Reserve sustain windows (MILP Eq. 4-5 parity). Single source of
        # truth: market_constants.SustainDurations. Before this was
        # introduced the env used tau_FCR=4.0 h in the bid clip and 0.5 h
        # in the FIX 6 availability check while the MILP defaulted to
        # 0.25 h, so the three stages of one run disagreed. Now every
        # consumer reads self.sustain.
        self.sustain: SustainDurations = (
            sustain_hours if sustain_hours is not None else DEFAULT_SUSTAIN)
        # Duration-relative observation features (indices 26-28). When False
        # the three slots are written as zeros, so the observation vector keeps
        # its shape and only the information differs.
        self.expose_duration_features = bool(duration_features)
        # Fleet-level coupling (shared connection capacity). Must be the SAME
        # object the MILP was built with, or the optimizer plans against a
        # constraint the environment does not enforce and the comparison stops
        # being matched. See fleet_coupling.
        self.coupling: FleetCoupling = (
            coupling if coupling is not None else FleetCoupling.uncoupled())
        # Last fleet-level envelope, refreshed each step. Seeds the global
        # state block; zero before the first step of an episode.
        self._last_fleet_envelope: Tuple[float, float] = (0.0, 0.0)
        # Number of action axes depends on directional mode:
        #  legacy 5-axis: [charge, discharge, FCR, aFRR, mFRR]
        #  directional 7-axis: [charge, discharge, FCR, aFRR_up, aFRR_dn, mFRR_up, mFRR_dn]
        self.n_action_axes = (ACTION_AXES_DIRECTIONAL if self.directional_services
                              else ACTION_AXES)
        self.fce_cumulative_initial = (
            list(fce_cumulative_initial)
            if fce_cumulative_initial is not None
            else [0.0] * self.n_agents
        )
        # Capacity already lost at the start of this episode (fraction in [0,1]),
        # to be carried over across daily env re-instantiations so SOH does not
        # reset to 1.0 each day. None or list of zeros = fresh battery.
        self.capacity_lost_initial = (
            list(capacity_lost_initial)
            if capacity_lost_initial is not None
            else [0.0] * self.n_agents
        )
        self._default_seed = seed

        # ---- PettingZoo agent registry ----
        self.possible_agents: List[str] = [f"bess_{i}" for i in range(self.n_agents)]
        self.agents: List[str] = self.possible_agents.copy()
        self.agent_name_mapping: Dict[str, int] = {
            name: i for i, name in enumerate(self.possible_agents)
        }

        # ---- Spaces per agent ----
        self._action_space = spaces.MultiDiscrete([N_ACTION_BINS] * self.n_action_axes)
        # Full-foresight diagnostic flag: when on, the agent sees all 24h of true
        # prices/services (everything the MILP sees). Read via getattr so it can
        # be set before reset() like the commitment flags; default off.
        self.full_foresight: bool = bool(getattr(self, "full_foresight", False))
        # Centralised-critic global state, appended at the end when enabled.
        self.centralized_critic: bool = bool(centralized_critic)
        _obs_dim = OBS_DIM_FULL_FORESIGHT if self.full_foresight else OBS_DIM
        if self.centralized_critic:
            _obs_dim += GLOBAL_STATE_DIM
        # Observation bounds are loose, normalised features in [-2, 2]
        self._observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(_obs_dim,), dtype=np.float32,
        )

        # ---- Per-agent runtime state ----
        # All LFPBatteryState instances even in linear mode, so we can track
        # SOH for diagnostics; they are only USED for cost in non-linear mode.
        self._lfp_states: List[LFPBatteryState] = []
        self._socs: np.ndarray = np.full(self.n_agents, self.soc_init, dtype=np.float64)
        self._hour: int = 0
        self._prices_episode: List[float] = []
        self._services_episode: List[List[FlexibilityService]] = []
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._last_info: Dict[str, Any] = {}

        # ---- Realistic market model (Steps A-D): commitment table ----
        # When enable_commitments=True, awarded bids freeze battery power for
        # their delivery hours (capacity locking), instead of every hour being
        # an isolated auction. See commitments.py for the data structure and
        # the design rationale (day-ahead rolling, hourly lock, k=1.5 penalty).
        # Default False keeps the legacy single-hour behaviour bit-identical,
        # so existing runs are unaffected until the feature is switched on.
        from commitments import CommitmentTable, DEFAULT_PENALTY_K
        self.enable_commitments: bool = bool(
            getattr(self, "enable_commitments", False))
        # Day-ahead rolling lead time for SERVICE bids only: a service bid
        # decided at hour t is for delivery at hour (t + lead_time) wrapped into
        # the 24h episode, i.e. (t + lead_time) % episode_hours. Arbitrage
        # (charge/discharge) executes immediately at t (energy near-real-time);
        # only the MSD service reservations are bid a day ahead. Default 24h
        # reflects the real MSD ex-ante (bids placed the day before delivery).
        # The modulo wrap keeps deliveries inside the 24h episode while
        # preserving the full-day lead structure and the 7-axis action space.
        self.commitment_lead_time: int = int(
            getattr(self, "commitment_lead_time", 24))
        self.penalty_k: float = float(
            getattr(self, "penalty_k", DEFAULT_PENALTY_K))
        # Cure 1+2: immediate per-hour penalty on RAW overcommit (bids beyond
        # available power, measured pre-clip in _decode_clip_actions as a
        # fraction of P_max). Default 0.0 = OFF, so existing runs are unaffected.
        # When > 0, step() subtracts overcommit_penalty * excess_fraction * P_max
        # from each agent's reward IN THE SAME HOUR, an immediate, attributable
        # signal against saturating all axes. MILP and BC never overcommit, so
        # their penalty is always 0 and the comparison stays fair.
        self.overcommit_penalty: float = float(
            getattr(self, "overcommit_penalty", 0.0))
        # FIX 6: availability-penalty coefficient. In expected_reward_mode
        # (training only) the capacity payment is scaled by the deliverable
        # headroom; this coefficient (default 1.0 = full withholding of the
        # undeliverable capacity) lets the strength be softened. 0.0 disables
        # the fix. Eval / MILP replay / BC sample, so they are unaffected.
        self.availability_penalty_coeff: float = float(
            getattr(self, "availability_penalty_coeff", 1.0))
        # FIX A: JOINT sustain-budget enforcement at action decode (mirrors
        # MILP Eq. 4-5). The per-service sustain caps in _decode_clip_actions
        # each check the FULL SoC headroom independently, so stacking
        # FCR + aFRR + mFRR can jointly commit up to ~3x the deliverable
        # energy while every axis passes its own check; the capacity payment
        # is then earned on energy that cannot exist (capacity farming).
        # When True (default), the SUM of committed energy per direction is
        # additionally capped by the same SoC budget the MILP uses, in BOTH
        # training and evaluation (the decode is shared), closing the exploit
        # at the source. MILP-feasible bids always pass (budgets are chosen
        # >= the MILP's own limits), so MILP replay and BC are no-ops.
        self.enforce_joint_sustain: bool = bool(
            getattr(self, "enforce_joint_sustain", True))
        global _JOINT_SUSTAIN_ANNOUNCED
        if not _JOINT_SUSTAIN_ANNOUNCED:
            print(f"  [env] FIX A joint sustain budget: "
                  f"{'ENFORCED' if self.enforce_joint_sustain else 'OFF'} "
                  f"(MILP Eq. 4-5 parity, train+eval)", flush=True)
            _JOINT_SUSTAIN_ANNOUNCED = True
        # EXPECTED-REWARD MODE (training-only alignment with the MILP). See
        # configure_expected_reward(). When True, step() books award/activation
        # revenues in EXPECTATION instead of sampling the two Bernoulli draws.
        # Evaluation envs keep this False and sample the real market outcomes.
        self.expected_reward_mode: bool = bool(
            getattr(self, "expected_reward_mode", False))
        self._commitments: CommitmentTable = CommitmentTable(
            list(self.possible_agents))

    # ------------------------------------------------------------------
    # PettingZoo required: per-agent space queries
    # ------------------------------------------------------------------

    def configure_commitments(self, enable: bool = True,
                              lead_time: int = 24, penalty_k: float = 1.5):
        """Enable/parameterise the realistic commitment market model.

        Call AFTER construction (and before reset) to switch on Steps A-D:
        day-ahead rolling bids, capacity locking, deferred activation, and the
        non-delivery penalty. Returns self for chaining.

        Parameters
        ----------
        enable : bool
            If False, the env behaves exactly as the legacy single-hour
            auction (bit-identical), which is the default.
        lead_time : int
            Hours between a service bid and its delivery, wrapped modulo the
            episode length. 24 = day-ahead.
        penalty_k : float
            Non-delivery penalty multiplier (undelivered energy is charged
            penalty_k * energy_price). 1.5 by default.
        """
        self.enable_commitments = bool(enable)
        self.commitment_lead_time = int(lead_time)
        self.penalty_k = float(penalty_k)
        return self

    def configure_overcommit_penalty(self, coeff: float = 0.0):
        """Set the immediate overcommit penalty coefficient (Cure 1+2).
        0.0 disables it (default). When > 0, each agent is charged
        coeff * overcommit_excess_fraction * P_max in the same hour it bids
        beyond available power. Returns self.
        """
        self.overcommit_penalty = float(coeff)
        return self

    def configure_expected_reward(self, enable: bool = True):
        """Enable expected-reward mode (training-only). When enabled, step()
        books reserve award/activation revenues in expectation (weighted by
        their probabilities), matching the MILP objective, instead of sampling
        the two Bernoulli draws. Use for the PPO/BC TRAINING env; keep it
        disabled (default) for the EVALUATION env so it samples the true
        realized market outcomes. Returns self.
        """
        self.expected_reward_mode = bool(enable)
        return self

    def configure_availability_penalty(self, coeff: float = 1.0):
        """Set the FIX 6 availability-penalty coefficient (training only).
        In expected_reward_mode the capacity payment of each commitment is
        scaled by the fraction of committed power the battery can actually
        deliver at the delivery hour (over the service sustain window);
        `coeff` multiplies the withheld part. 1.0 (default) withholds all
        undeliverable capacity, 0.0 disables the fix. Evaluation envs sample
        and are unaffected regardless. Returns self.
        """
        self.availability_penalty_coeff = float(coeff)
        return self

    def configure_joint_sustain(self, enable: bool = True):
        """Enable/disable the JOINT sustain-budget enforcement (FIX A).
        Default ON. Applies at action decode in BOTH training and evaluation,
        capping the summed committed energy per direction to the SoC budget
        (MILP Eq. 4-5 parity). Disable only to reproduce legacy runs where
        the per-service caps alone allowed joint over-commitment. Returns
        self.
        """
        self.enforce_joint_sustain = bool(enable)
        return self

    def configure_coupling(self, coupling: FleetCoupling):
        """Set the fleet-level coupling. Returns self."""
        if not isinstance(coupling, FleetCoupling):
            raise TypeError(
                "coupling must be a fleet_coupling.FleetCoupling, got "
                f"{type(coupling).__name__}")
        self.coupling = coupling
        return self

    def configure_duration_features(self, enable: bool = True):
        """Turn the duration-relative observation features (26-28) on or off.

        Off writes zeros into the three slots instead of shrinking the vector,
        which keeps the network architecture identical between the two arms of
        the comparison. Returns self.
        """
        self.expose_duration_features = bool(enable)
        return self

    def configure_sustain_hours(self, sustain_hours: SustainDurations):
        """Set the reserve sustain windows used by the bid clip, the joint
        sustain budget (FIX A) and the FIX 6 availability check.

        Must be the SAME object handed to the MILP (via
        `sustain_hours.as_milp_kwargs()`), otherwise the optimizer plans
        against constraints the environment does not enforce and the
        comparison is no longer matched. Call after construction and
        before reset. Returns self.
        """
        if not isinstance(sustain_hours, SustainDurations):
            raise TypeError(
                "sustain_hours must be a market_constants.SustainDurations, "
                f"got {type(sustain_hours).__name__}")
        self.sustain = sustain_hours
        return self

    def configure_full_foresight(self, enable: bool = True):
        """Enable the diagnostic full-foresight observation (all 24h true
        prices/services, everything the MILP sees). Resizes the observation
        space. Call after construction and before reset. Returns self.
        Diagnostic only — turns the PPO into a perfect-foresight agent.
        """
        self.full_foresight = bool(enable)
        self._rebuild_observation_space()
        return self

    def _rebuild_observation_space(self):
        """Recompute the observation space after a flag change.

        Both flags that change the observation width go through here, so the
        two cannot drift apart: a space rebuilt for full foresight but not for
        the global state would silently truncate the tail.
        """
        _obs_dim = OBS_DIM_FULL_FORESIGHT if self.full_foresight else OBS_DIM
        if getattr(self, "centralized_critic", False):
            _obs_dim += GLOBAL_STATE_DIM
        self._observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(_obs_dim,), dtype=np.float32,
        )
        return self

    def configure_centralized_critic(self, enable: bool = True):
        """Append the global-state block to every agent's observation.

        The block is IDENTICAL across agents: it is the fleet's state, not the
        agent's. A centralised critic reads it; a decentralised actor must not,
        which is what marl_centralized_critic.CentralisedCriticModule enforces
        by slicing it off before the policy head.
        """
        self.centralized_critic = bool(enable)
        return self._rebuild_observation_space()

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._default_seed is not None:
            self._rng = np.random.default_rng(self._default_seed)

        self.agents = self.possible_agents.copy()
        self._hour = 0
        self._socs = np.full(self.n_agents, self.soc_init, dtype=np.float64)
        self._commitments.reset(list(self.possible_agents))
        self._lfp_states = []
        for i, bp in enumerate(self.multi_params.batteries):
            s = LFPBatteryState(capacity_mwh=bp.capacity_mwh)
            s.fce_cumulative = float(self.fce_cumulative_initial[i])
            # Carry forward prior capacity loss so SOH does not reset each
            # episode (e.g. across daily env re-instantiations in the
            # multi-day rolling pipeline). We attribute it to calendar loss
            # by convention; downstream computations use the property
            # capacity_lost_fraction = _Q_loss_cal + _Q_loss_cyc which
            # treats the two channels symmetrically.
            s._Q_loss_cal = float(self.capacity_lost_initial[i])
            self._lfp_states.append(s)

        self._prices_episode = (
            list(self._fixed_prices)
            if self._fixed_prices is not None
            else self._default_prices(self.episode_hours)
        )
        self._services_episode = (
            list(self._fixed_services)
            if self._fixed_services is not None
            else self._default_services(self.episode_hours)
        )

        obs = {a: self._build_observation(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        if not self.agents:
            raise RuntimeError("step() called after the episode terminated")

        t = self._hour
        price = self._prices_episode[t]
        services_now = self._services_episode[t]
        services_by_type = {s.service_type: s for s in services_now}

        # STEP B: drop commitments whose delivery window has passed, so
        # locked_power(t) reflects only currently-active commitments.
        if self.enable_commitments:
            self._commitments.expire(t)

        # ---- Decode and clip per-agent actions ----
        per_agent = self._decode_clip_actions(actions, hour=t)

        # ---- Per-agent SOC dynamics from arbitrage (services applied later) ----
        rewards: Dict[str, float] = {}
        per_agent_arb: Dict[str, float] = {}
        per_agent_deg: Dict[str, float] = {}
        per_agent_overcommit_pen: Dict[str, float] = {}
        per_agent_flex: Dict[str, float] = {a: 0.0 for a in self.agents}
        per_agent_throughput: Dict[str, float] = {}

        for a in self.agents:
            i = self.agent_name_mapping[a]
            bp = self.multi_params.batteries[i]
            P_ch = per_agent[a]['P_charge']
            P_dis = per_agent[a]['P_discharge']
            arb = (P_dis - P_ch) * price
            per_agent_arb[a] = arb
            eff = bp.efficiency
            cap = bp.capacity_mwh
            d_soc = (eff * P_ch / cap) - (P_dis / (eff * cap))
            self._socs[i] = float(np.clip(self._socs[i] + d_soc, 0.0, 1.0))
            per_agent_throughput[a] = P_ch + P_dis

        # ---- Aggregate flex bids per service type ----
        if self.directional_services:
            service_keys = ('R_fcr', 'R_afrr_up', 'R_afrr_dn', 'R_mfrr_up', 'R_mfrr_dn')
            service_types = (
                ServiceType.FCR, ServiceType.AFRR_UP, ServiceType.AFRR_DN,
                ServiceType.MFRR_UP, ServiceType.MFRR_DN,
            )
            # Activation direction: True if upward (BESS discharges), False if downward,
            # FCR is symmetric (no SOC impact).
            service_directions = {
                ServiceType.FCR:     'sym',
                ServiceType.AFRR_UP: 'up', ServiceType.AFRR_DN: 'dn',
                ServiceType.MFRR_UP: 'up', ServiceType.MFRR_DN: 'dn',
            }
        else:
            service_keys = ('R_fcr', 'R_afrr', 'R_mfrr')
            service_types = (ServiceType.FCR, ServiceType.AFRR, ServiceType.MFRR)
            service_directions = {
                ServiceType.FCR: 'sym', ServiceType.AFRR: 'sym', ServiceType.MFRR: 'sym',
            }

        agg_bids = {st: 0.0 for st in service_types}
        per_agent_bid = {a: {} for a in self.agents}
        for a in self.agents:
            for st, key in zip(service_types, service_keys):
                bid = per_agent[a][key]
                per_agent_bid[a][st] = bid
                agg_bids[st] += bid

        # ---- For each service: ARERA aggregate min, award draw, activation draw ----
        flex_throughput_per_agent = {a: 0.0 for a in self.agents}
        # Track per-agent SOC change from activations (downward charges; upward discharges)
        # so we can apply it AFTER the arbitrage SOC update above.
        per_agent_dsoc_act = {a: 0.0 for a in self.agents}
        # FIX 6: per-agent availability penalty (capacity remuneration withheld
        # for committed power the battery cannot back at delivery). Nonzero only
        # in expected_reward_mode (training); eval / MILP replay / BC keep it 0.
        per_agent_avail_pen = {a: 0.0 for a in self.agents}

        # In directional mode, draw a single CATEGORICAL outcome per hour for
        # each service root (aFRR, mFRR) that determines whether Terna calls
        # upward, downward, or neither. This enforces operational mutex: for
        # any hour the system can only be called in ONE direction, matching
        # MSD reality. award_probability for up/dn was already mutex-normalised
        # in italian_market_data.py so P_up_excl + P_dn_excl <= 1.
        directional_awarded_services = None
        if self.directional_services:
            directional_awarded_services = set()
            for root, st_up, st_dn in [
                ('aFRR', ServiceType.AFRR_UP, ServiceType.AFRR_DN),
                ('mFRR', ServiceType.MFRR_UP, ServiceType.MFRR_DN),
            ]:
                s_up = services_by_type.get(st_up)
                s_dn = services_by_type.get(st_dn)
                p_up = s_up.award_probability if s_up is not None else 0.0
                p_dn = s_dn.award_probability if s_dn is not None else 0.0
                # Categorical draw: outcome in {'up', 'dn', 'none'}
                # with P(up)=p_up_excl, P(dn)=p_dn_excl, P(none)=1-p_up-p_dn.
                u = self._rng.random()
                if u < p_up:
                    directional_awarded_services.add(st_up)
                elif u < p_up + p_dn:
                    directional_awarded_services.add(st_dn)
                # else: neither awarded this hour

        # Per-service total revenue (aggregate across BESS), for trajectory
        # recording and per-policy decomposition charts.
        per_service_revenue: Dict["ServiceType", float] = {}

        if not self.enable_commitments:
            # =============== LEGACY PATH (single-hour auction) ===============
            # Unchanged: bid, award, activation and payment all resolve in the
            # same hour t. Kept bit-identical for backward compatibility.
            for st in service_types:
                s = services_by_type.get(st)
                if s is None or agg_bids[st] < 1.0:
                    continue
                # Award weight: 1.0/0.0 when sampled, award_probability in
                # EXPECTED-REWARD MODE (training only; matches the MILP).
                if directional_awarded_services is not None and st != ServiceType.FCR:
                    aw_w = 1.0 if (st in directional_awarded_services) else 0.0
                elif self.expected_reward_mode:
                    aw_w = float(s.award_probability)
                else:
                    aw_w = 1.0 if (self._rng.random() < s.award_probability) else 0.0
                if aw_w <= 0.0:
                    continue
                # Activation weight: 1.0/0.0 when sampled, activation_probability
                # in EXPECTED-REWARD MODE.
                if self.expected_reward_mode:
                    act_w = float(s.activation_probability)
                else:
                    act_w = 1.0 if (self._rng.random() < s.activation_probability) else 0.0
                total_cap_payment = s.capacity_price * agg_bids[st] * aw_w
                total_energy_payment = s.energy_price * agg_bids[st] * aw_w * act_w
                direction = service_directions[st]
                per_service_revenue[st] = total_cap_payment + total_energy_payment

                for a in self.agents:
                    share = (per_agent_bid[a][st] / agg_bids[st]
                              if agg_bids[st] > 0 else 0.0)
                    per_agent_flex[a] += share * (total_cap_payment + total_energy_payment)
                    act_mwh = per_agent_bid[a][st] * aw_w * act_w
                    if act_mwh > 0:
                        i = self.agent_name_mapping[a]
                        bp = self.multi_params.batteries[i]
                        cap = bp.capacity_mwh
                        eff = bp.efficiency
                        flex_throughput_per_agent[a] += act_mwh
                        if direction == 'up':
                            per_agent_dsoc_act[a] -= act_mwh / (eff * cap)
                        elif direction == 'dn':
                            per_agent_dsoc_act[a] += (eff * act_mwh) / cap
        else:
            # =============== COMMITMENT PATH (Steps C + D) ===================
            # Two temporally separated flows happen in hour t:
            #
            #  (C) BID @ t: the service bids decoded this hour are NOT delivered
            #      now. For each, draw the award; if won, pay the CAPACITY
            #      payment immediately (you are paid for being committed) and
            #      register a Commitment for delivery at (t + lead) % 24.
            #
            #  (D) DELIVERY @ t: commitments registered `lead` hours ago whose
            #      delivery_hour == t are now active. For each, draw activation;
            #      if activated, pay the ENERGY payment, move SoC, and — if the
            #      battery cannot physically deliver (insufficient SoC/headroom)
            #      — charge a non-delivery PENALTY of penalty_k * energy_price
            #      on the undelivered MWh.
            delivery_hour = (t + self.commitment_lead_time) % self.episode_hours

            # ---- (C) award + register future commitment ----
            # Option 3: capacity is NO LONGER paid here at bid time. Both the
            # capacity payment and the energy payment are now booked together at
            # DELIVERY (Step D). This collapses the reward of a service into a
            # single coherent event at the delivery hour, instead of splitting an
            # immediate capacity reward (which rewarded over-committing) from a
            # delayed penalty. Combined with the pending-commitment observation
            # features, it gives the PPO a single, attributable consequence.
            for st in service_types:
                s = services_by_type.get(st)
                if s is None or agg_bids[st] < 1.0:
                    continue
                # EXPECTED-REWARD MODE (training only). FIX 5: register the
                # commitment at FULL reserved size, exactly like a won bid, so
                # locked_power / P_free reflect the true MILP power constraint
                # (P_ch + P_dis + sum R_s <= P_max). The award-probability
                # weighting is applied to the PAYMENT at delivery (Step D), NOT
                # to the reservation. Registering the scaled size here was the
                # leak: it under-counted locked power and let the PPO stack
                # phantom reserve that only exists in expectation.
                if self.expected_reward_mode:
                    for a in self.agents:
                        mw = per_agent_bid[a][st]
                        if mw > 0:
                            self._commitments.add(
                                a, _service_key_for(st), mw,
                                bid_hour=t, delivery_hour=delivery_hour,
                            )
                    continue
                if directional_awarded_services is not None and st != ServiceType.FCR:
                    awarded = st in directional_awarded_services
                else:
                    awarded = self._rng.random() < s.award_probability
                if not awarded:
                    continue
                for a in self.agents:
                    # register the won capacity as a future commitment; payment
                    # happens at delivery (Step D).
                    mw = per_agent_bid[a][st]
                    if mw > 0:
                        self._commitments.add(
                            a, _service_key_for(st), mw,
                            bid_hour=t, delivery_hour=delivery_hour,
                        )

            # ---- (D) deliver + activate commitments due THIS hour ----
            # FIX (SOC bound safety): multiple commitments can activate in the
            # same hour. Each one's deliverable must be computed against the SoC
            # AS IT EVOLVES within the hour, not the hour's starting SoC, or the
            # cumulative SoC delta can push past [soc_min, soc_max]. We track a
            # projected SoC `soc_proj` that starts at the current SoC and is
            # updated after every activation, so the headroom each commitment
            # sees already accounts for the ones processed before it. This makes
            # bound violations structurally impossible (MILP/BC stayed in bounds
            # only because they bid sanely; overcommitting policies exposed the
            # missing within-hour clamp).
            # FIX 6 reference sustain durations for the availability check.
            # These MUST be the same windows the bid-time decode clip and the
            # MILP headroom use, or a policy that keeps SoC exactly where the
            # MILP does would still be charged an availability penalty. They
            # previously did not match (0.5 h here vs 4.0 h in the clip vs
            # 0.25 h in the MILP default); all three now read self.sustain.
            _tau = self.sustain
            _avail_ref_h = {
                ServiceType.FCR: _tau.fcr,
                ServiceType.AFRR: _tau.afrr,
                ServiceType.MFRR: _tau.mfrr,
            }
            if self.directional_services:
                _avail_ref_h.update({
                    ServiceType.AFRR_UP: _tau.afrr, ServiceType.AFRR_DN: _tau.afrr,
                    ServiceType.MFRR_UP: _tau.mfrr, ServiceType.MFRR_DN: _tau.mfrr,
                })
            for a in self.agents:
                i = self.agent_name_mapping[a]
                bp = self.multi_params.batteries[i]
                cap = bp.capacity_mwh
                eff = bp.efficiency
                soc_proj = self._socs[i]  # evolves as activations are applied
                for c in self._commitments.active_commitments(a, t):
                    st = _service_type_for(c.service)
                    s = services_by_type.get(st)
                    if s is None:
                        continue
                    direction = service_directions.get(st, 'sym')
                    requested_mwh = c.mw  # MW * 1h
                    energy_price = s.energy_price
                    # Deliverable headroom given the PROJECTED SoC (already
                    # reflects earlier activations in this same hour). Computed
                    # BEFORE the capacity payment so the payment can be scaled by
                    # what the battery can actually back this hour.
                    if direction == 'up':
                        deliverable = max(0.0, (soc_proj - self.soc_min) * eff * cap)
                    elif direction == 'dn':
                        deliverable = max(0.0, (self.soc_max - soc_proj) * cap / eff)
                    else:  # FCR symmetric: limited by the tighter of the two rooms
                        deliverable = max(0.0, min((soc_proj - self.soc_min),
                                                   (self.soc_max - soc_proj)) * cap)

                    # FIX 5: commitment is registered at FULL reserved size, so
                    # value it by its award probability HERE (aw_w = award_prob
                    # in expected-reward training, 1.0 in eval where it exists
                    # only because it was actually won).
                    if self.expected_reward_mode:
                        aw_w = float(s.award_probability)
                    else:
                        aw_w = 1.0
                    # Capacity payment booked at DELIVERY (Option 3): paid for
                    # being available this hour, regardless of activation.
                    cap_pay = s.capacity_price * c.mw * aw_w
                    per_agent_flex[a] += cap_pay
                    per_service_revenue[st] = per_service_revenue.get(st, 0.0) + cap_pay
                    # FIX 6 (TRAINING ONLY): capacity remuneration requires being
                    # able to DELIVER the committed power. Scale it by the
                    # deliverable fraction over the service sustain window and
                    # withhold the rest. Gated on expected_reward_mode so eval /
                    # MILP replay / BC (which sample) stay byte-identical and the
                    # reported baseline never moves. Net capacity contribution
                    # stays in [0, cap_pay], so it cannot destabilise training.
                    # This removes the capacity-farming incentive that let the
                    # PPO stack bids while parking SoC at an extreme to dodge
                    # cycling and degradation.
                    if self.expected_reward_mode and c.mw > 1e-9:
                        ref_h = _avail_ref_h.get(st, 1.0)
                        required = c.mw * ref_h
                        avail_frac = (float(np.clip(deliverable / required, 0.0, 1.0))
                                      if required > 1e-9 else 0.0)
                        avail_pen = (self.availability_penalty_coeff * cap_pay
                                     * (1.0 - avail_frac))
                        if avail_pen != 0.0:
                            per_agent_flex[a] -= avail_pen
                            per_service_revenue[st] = (
                                per_service_revenue.get(st, 0.0) - avail_pen)
                            per_agent_avail_pen[a] += avail_pen
                    # activation in the DELIVERY hour. EXPECTED-REWARD MODE:
                    # weight by activation_probability instead of drawing the
                    # Bernoulli; eval keeps sampling (bit-identical there).
                    if self.expected_reward_mode:
                        act_w = float(s.activation_probability)
                    else:
                        act_w = 1.0 if (self._rng.random() < s.activation_probability) else 0.0
                    if act_w <= 0.0:
                        continue
                    delivered = min(requested_mwh, deliverable)
                    undelivered = max(0.0, requested_mwh - delivered)

                    # energy payment weighted by award AND activation
                    # expectation (w = 1.0 in eval).
                    w = aw_w * act_w
                    pay = energy_price * delivered * w
                    per_agent_flex[a] += pay
                    per_service_revenue[st] = per_service_revenue.get(st, 0.0) + pay
                    eff_delivered = delivered * w
                    flex_throughput_per_agent[a] += eff_delivered
                    # Update BOTH the accumulator (applied later) and the local
                    # projection (so the next commitment sees the new headroom).
                    if direction == 'up':
                        dsoc = -eff_delivered / (eff * cap)
                    elif direction == 'dn':
                        dsoc = (eff * eff_delivered) / cap
                    else:
                        dsoc = 0.0
                    per_agent_dsoc_act[a] += dsoc
                    soc_proj = min(self.soc_max, max(self.soc_min, soc_proj + dsoc))

                    # (D) NON-DELIVERY PENALTY on the shortfall, weighted by w.
                    if undelivered > 0:
                        penalty = self.penalty_k * energy_price * undelivered * w
                        per_agent_flex[a] -= penalty
                        per_service_revenue[st] = per_service_revenue.get(st, 0.0) - penalty

        # Apply SOC delta from activations
        for a in self.agents:
            i = self.agent_name_mapping[a]
            self._socs[i] = float(np.clip(self._socs[i] + per_agent_dsoc_act[a],
                                           0.0, 1.0))

        # ---- Per-agent degradation cost ----
        for a in self.agents:
            i = self.agent_name_mapping[a]
            bp = self.multi_params.batteries[i]
            base_tp = per_agent_throughput[a]
            flex_tp = flex_throughput_per_agent[a]
            total_tp = base_tp + flex_tp

            if self.use_nonlinear_degradation:
                soc_now = self._socs[i]
                P_ch = per_agent[a]['P_charge']
                P_dis = per_agent[a]['P_discharge']
                # ---- DoD proxy fix for directional flex services ----
                # The old formula summed arbitrage + ALL flex throughput as if
                # it were one big cycle, giving dod_proxy ≈ 1.0 whenever flex
                # was active. In directional mode, with aggressive aFRR_dn
                # bidding, this pushed the non-linear LFP model into the full-
                # cycle regime where cyclic aging is 2–4× higher, causing
                # SOH to drop ~37%/year instead of the realistic ~12-15%.
                #
                # The corrected formula separates contributions:
                #   - Arbitrage: max(charge, discharge) MWh per hour → swing in one direction
                #   - Flex: many micro-cycles (multiple service activations per hour
                #     with partially cancelling SoC swings), each effective DoD ~10-15%.
                arb_dod = max(abs(P_ch), abs(P_dis)) / bp.capacity_mwh
                flex_micro_dod_factor = 0.15
                flex_dod = (flex_tp / bp.capacity_mwh) * flex_micro_dod_factor
                dod_proxy = float(np.clip(arb_dod + flex_dod, 0.05, 1.0))
                # ---- Attribute flex throughput between charge and discharge ----
                # aFRR_dn/mFRR_dn activate as battery CHARGE; aFRR_up/mFRR_up
                # and FCR activate as DISCHARGE. We split flex_tp using
                # per_agent_dsoc_act sign as a proxy.
                # Net SoC change from activations: positive = charge dominant.
                soc_delta_act = per_agent_dsoc_act.get(a, 0.0)
                if soc_delta_act >= 0:
                    flex_ch_share = min(1.0, abs(soc_delta_act) * bp.capacity_mwh
                                         / max(flex_tp, 1e-6))
                else:
                    flex_ch_share = max(0.0, 1.0 - abs(soc_delta_act) * bp.capacity_mwh
                                         / max(flex_tp, 1e-6))
                flex_ch_share = float(np.clip(flex_ch_share, 0.0, 1.0))
                state = self._lfp_states[i]
                cap_loss_before = state.capacity_lost_fraction
                state.advance(
                    hours=1.0,
                    soc_avg=soc_now,
                    charge_mwh=P_ch + flex_tp * flex_ch_share,
                    discharge_mwh=P_dis + flex_tp * (1.0 - flex_ch_share),
                    dod=dod_proxy,
                    T_kelvin=self.nonlinear_temperature_K,
                )
                cap_loss_after = state.capacity_lost_fraction
                d_loss = cap_loss_after - cap_loss_before
                deg = d_loss * self.nonlinear_replacement_cost
            else:
                # ---- Linear degradation (matches the MILP's internal model) ----
                rate = (bp.degradation_cost_per_mwh / (2.0 * bp.cycle_life))
                deg = rate * total_tp
                # SOH tracking CONSISTENT with the linear cost. Key fix: cycle_life
                # is the number of equivalent full cycles to END-OF-LIFE, defined
                # as 80% SOH = 20% capacity loss. So one equivalent full cycle
                # costs (0.20 / cycle_life) of capacity, NOT (1.0 / cycle_life).
                # The old code advanced the non-linear model with a fixed dod=0.5,
                # which (a) ignored this 20% convention and (b) overstated fade ~5x,
                # crashing the plotted SOH (e.g. Random to ~53%).
                EOL_CAPACITY_LOSS = 0.20  # 80% SOH end-of-life convention
                state = self._lfp_states[i]
                eq_full_cycles = total_tp / (2.0 * max(bp.capacity_mwh, 1e-6))
                lin_cap_loss = eq_full_cycles * (EOL_CAPACITY_LOSS
                                                 / max(bp.cycle_life, 1.0))
                try:
                    state._Q_loss_cyc += lin_cap_loss
                except Exception:
                    pass

            per_agent_deg[a] = deg
            # Cure 1+2: immediate per-hour penalty on raw overcommit. The excess
            # (fraction of P_max bid beyond available power, captured pre-clip)
            # is converted to a power-equivalent and charged at the configured
            # coefficient. Zero when overcommit_penalty == 0 (default) or when
            # the agent did not overcommit (always the case for MILP/BC), so the
            # cross-policy comparison stays fair — only saturating policies pay.
            oc_excess = per_agent[a].get('overcommit', 0.0)
            bp_a = self.multi_params.batteries[self.agent_name_mapping[a]]
            oc_pen = self.overcommit_penalty * oc_excess * bp_a.max_power_mw
            per_agent_overcommit_pen[a] = oc_pen
            rewards[a] = per_agent_arb[a] + per_agent_flex[a] - deg - oc_pen

        # ---- Advance hour ----
        self._hour += 1
        terminated = {a: False for a in self.agents}
        truncated = {a: (self._hour >= self.episode_hours) for a in self.agents}

        obs = {a: self._build_observation(a) for a in self.agents}

        infos = {
            a: {
                'arbitrage': per_agent_arb[a],
                'flex_revenue': per_agent_flex[a],
                'degradation': per_agent_deg[a],
                'availability_penalty': float(per_agent_avail_pen[a]),
                'throughput': per_agent_throughput[a] + flex_throughput_per_agent[a],
                'soc': float(self._socs[self.agent_name_mapping[a]]),
                'soh': float(self._lfp_states[self.agent_name_mapping[a]].soh),
                'fce': float(self._lfp_states[self.agent_name_mapping[a]].fce_cumulative),
                # FIX 1 diagnostics: reserve-stacking & overcommit signals so we
                # can log Sigma(bid)/P_max, how much was clipped, and locked MW.
                'reserve_bid_mw': float(sum(per_agent_bid[a].values())),
                'reserve_frac_pmax': float(sum(per_agent_bid[a].values())
                    / max(self.multi_params.batteries[
                        self.agent_name_mapping[a]].max_power_mw, 1e-9)),
                'overcommit_excess': float(per_agent[a].get('overcommit', 0.0)),
                'overcommit_penalty': float(per_agent_overcommit_pen[a]),
                'locked_power': float(self._commitments.locked_power(a, t)
                    if self.enable_commitments else 0.0),
            } for a in self.agents
        }
        # Aggregate diagnostic dict reflecting directional or legacy keys
        agg_diag = {'hour': t, 'price': price, 'total_reward': sum(rewards.values())}
        if self.directional_services:
            agg_diag.update({
                'aggregate_bid_fcr':     agg_bids[ServiceType.FCR],
                'aggregate_bid_afrr_up': agg_bids[ServiceType.AFRR_UP],
                'aggregate_bid_afrr_dn': agg_bids[ServiceType.AFRR_DN],
                'aggregate_bid_mfrr_up': agg_bids[ServiceType.MFRR_UP],
                'aggregate_bid_mfrr_dn': agg_bids[ServiceType.MFRR_DN],
                'rev_fcr':     per_service_revenue.get(ServiceType.FCR, 0.0),
                'rev_afrr_up': per_service_revenue.get(ServiceType.AFRR_UP, 0.0),
                'rev_afrr_dn': per_service_revenue.get(ServiceType.AFRR_DN, 0.0),
                'rev_mfrr_up': per_service_revenue.get(ServiceType.MFRR_UP, 0.0),
                'rev_mfrr_dn': per_service_revenue.get(ServiceType.MFRR_DN, 0.0),
            })
        else:
            agg_diag.update({
                'aggregate_bid_fcr':  agg_bids[ServiceType.FCR],
                'aggregate_bid_afrr': agg_bids[ServiceType.AFRR],
                'aggregate_bid_mfrr': agg_bids[ServiceType.MFRR],
                'rev_fcr':  per_service_revenue.get(ServiceType.FCR, 0.0),
                'rev_afrr': per_service_revenue.get(ServiceType.AFRR, 0.0),
                'rev_mfrr': per_service_revenue.get(ServiceType.MFRR, 0.0),
            })
        self.last_aggregate_info = agg_diag
        self._last_info = infos

        if any(truncated.values()):
            self.agents = []

        return obs, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Action decoding + feasibility projection
    # ------------------------------------------------------------------

    def _decode_clip_actions(self, actions: Dict[str, np.ndarray],
                             hour: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Decode discrete actions to MW values, then project to feasible set.

        In legacy mode (5-axis): [charge, discharge, FCR, aFRR, mFRR].
        In directional mode (7-axis): [charge, discharge, FCR, aFRR_up,
        aFRR_dn, mFRR_up, mFRR_dn] with three new projections:
          a) Per-service direction mutex: for aFRR and mFRR, only one of
             the two directions can be > 0. If both > 0, keep the LARGER
             and zero the other (matches MILP B_dir mutual exclusion).
          b) Sustain pre-check: upward reserves are capped so SOC can
             sustain the activation for `sustain_hours` (1h aFRR, 2h mFRR).
             Downward reserves are capped so SOC has headroom to absorb.
          c) Total capacity balance: same as legacy but extended to 7 axes.

        STEP B (capacity locking): when commitments are enabled, the power
        already frozen by commitments active at `hour` is subtracted from the
        battery's available P_max BEFORE the capacity balance, so new bids and
        arbitrage must fit in the REMAINING headroom. This is what makes
        over-committing costly: locked power is not free to reuse.
        """
        out = {}
        # Sustain windows from the single source of truth (see
        # market_constants). Handed in at construction or via
        # configure_sustain_hours(); identical to what the MILP was built
        # with, so a MILP-feasible bid always survives this clip.
        fcr_sustain = self.sustain.fcr
        afrr_sustain = self.sustain.afrr
        mfrr_sustain = self.sustain.mfrr
        _use_lock = (self.enable_commitments and hour is not None)

        for a in self.agents:
            i = self.agent_name_mapping[a]
            bp = self.multi_params.batteries[i]
            fracs = _decode_action_to_fractions(np.asarray(actions[a]))
            P_max = bp.max_power_mw
            cap = bp.capacity_mwh
            eff = bp.efficiency
            soc = self._socs[i]

            # STEP B: shrink available power by what is already locked at `hour`.
            # Fractions were decoded against the nominal P_max, so we scale the
            # CAP (the capacity-balance budget) down to the free headroom. The
            # locked capacity itself is handled separately (it is honoured/
            # activated via the commitment table in Step D), so here we only
            # ensure NEW bids + arbitrage fit in what is left.
            locked = (self._commitments.locked_power(a, hour)
                      if _use_lock else 0.0)
            P_free = max(0.0, P_max - locked)

            if self.directional_services:
                (P_ch, P_dis, R_fcr,
                 R_afrr_up, R_afrr_dn,
                 R_mfrr_up, R_mfrr_dn) = (f * P_max for f in fracs)

                # 1) Mutex charge/discharge (arbitrage)
                if P_ch > 0 and P_dis > 0:
                    if P_ch >= P_dis: P_dis = 0.0
                    else: P_ch = 0.0

                # 1b) Direction mutex per service
                if R_afrr_up > 0 and R_afrr_dn > 0:
                    if R_afrr_up >= R_afrr_dn: R_afrr_dn = 0.0
                    else: R_afrr_up = 0.0
                if R_mfrr_up > 0 and R_mfrr_dn > 0:
                    if R_mfrr_up >= R_mfrr_dn: R_mfrr_dn = 0.0
                    else: R_mfrr_up = 0.0

                # 2) Capacity balance: total <= P_free (= P_max - locked).
                # With commitments enabled, P_free is the headroom left after
                # honouring power already frozen by active commitments; this is
                # the mechanism that penalises over-committing.
                total = (P_ch + P_dis + R_fcr
                         + R_afrr_up + R_afrr_dn + R_mfrr_up + R_mfrr_dn)
                # Cure 2: record the RAW overcommit excess (before scaling),
                # normalised by P_max. This is the amount the agent tried to bid
                # beyond its available power. A reward penalty proportional to
                # this is applied in step(), giving an IMMEDIATE, attributable
                # signal against saturating all axes — the gradient that the
                # mere proportional rescale below does NOT provide (rescaling
                # keeps the full allocation, so saturating was previously free).
                overcommit_excess = max(0.0, total - P_free) / max(P_max, 1e-9)
                if total > P_free and total > 0:
                    scale = P_free / total
                    P_ch *= scale; P_dis *= scale; R_fcr *= scale
                    R_afrr_up *= scale; R_afrr_dn *= scale
                    R_mfrr_up *= scale; R_mfrr_dn *= scale

                # 3) Sustain pre-check on services. Upward needs energy
                # to discharge for sustain hours; downward needs headroom.
                # FCR is symmetric, so it needs both up- and down-room for
                # tau_FCR.
                #
                # REVISION NOTE (Reviewer 2, point 2): this was hard-coded to
                # 4.0 h on the argument that Terna requires a 4 h sustain to
                # qualify. That is wrong. SO GL (Reg. EU 2017/1485) Art.
                # 156(10) bounds the minimum activation period for FCR
                # providers between 15 and 30 minutes, and Art. 156(9) sets
                # 15 minutes as the fallback for limited-energy reservoirs,
                # which is what a BESS is. At 4 h a 2 h fleet cannot offer
                # FCR at rated power at all, which is what pushed the
                # policies onto the restoration products. The window now
                # comes from self.sustain (0.25 h by default) and is swept
                # explicitly in the tau_FCR sensitivity.
                fcr_energy_cap = max(0.0, (soc - self.soc_min) * cap) / max(fcr_sustain, 1e-9)
                fcr_room_cap = max(0.0, (self.soc_max - soc) * cap) / max(fcr_sustain, 1e-9)
                R_fcr = min(R_fcr, fcr_energy_cap, fcr_room_cap)

                afrr_up_cap = max(0.0, (soc - self.soc_min) * cap) / max(afrr_sustain, 1e-9)
                R_afrr_up = min(R_afrr_up, afrr_up_cap)
                afrr_dn_cap = max(0.0, (self.soc_max - soc) * cap) / max(afrr_sustain, 1e-9)
                R_afrr_dn = min(R_afrr_dn, afrr_dn_cap)

                mfrr_up_cap = max(0.0, (soc - self.soc_min) * cap) / max(mfrr_sustain, 1e-9)
                R_mfrr_up = min(R_mfrr_up, mfrr_up_cap)
                mfrr_dn_cap = max(0.0, (self.soc_max - soc) * cap) / max(mfrr_sustain, 1e-9)
                R_mfrr_dn = min(R_mfrr_dn, mfrr_dn_cap)

                # 3b) FIX A: JOINT sustain budget per direction (MILP Eq. 4-5
                # parity). The caps above are PER-SERVICE: each one checks the
                # full SoC headroom on its own, so the same energy budget is
                # counted once per service and the stacked commitment can
                # reach ~3x what the battery can sustain. Here the SUM of
                # committed energy per direction is capped by the SoC budget;
                # FCR (symmetric) counts on both sides. Proportional scaling
                # preserves the bid mix (no arbitrary service priority).
                # Budget conventions are chosen to DOMINATE the MILP's own
                # sustain constraints, so any MILP-feasible bid passes
                # untouched (up: MILP uses <= headroom*eta_d <= headroom;
                # dn: matches the Step-D absorbable-energy convention
                # headroom/eff): MILP replay and BC stay no-ops by
                # construction. Applies in BOTH training and evaluation.
                if self.enforce_joint_sustain:
                    E_up_budget = max(0.0, (soc - self.soc_min) * cap)
                    E_dn_budget = max(0.0, (self.soc_max - soc) * cap / eff)
                    need_up = (R_fcr * fcr_sustain
                               + R_afrr_up * afrr_sustain
                               + R_mfrr_up * mfrr_sustain)
                    if need_up > E_up_budget and need_up > 1e-12:
                        s_up = E_up_budget / need_up
                        R_fcr *= s_up
                        R_afrr_up *= s_up
                        R_mfrr_up *= s_up
                    need_dn = (R_fcr * fcr_sustain
                               + R_afrr_dn * afrr_sustain
                               + R_mfrr_dn * mfrr_sustain)
                    if need_dn > E_dn_budget and need_dn > 1e-12:
                        s_dn = E_dn_budget / need_dn
                        R_fcr *= s_dn
                        R_afrr_dn *= s_dn
                        R_mfrr_dn *= s_dn

                # 4) Arbitrage SOC feasibility
                P_ch_max = max(0.0, (self.soc_max - soc) * cap / eff)
                P_ch = min(P_ch, P_ch_max)
                P_dis_max = max(0.0, (soc - self.soc_min) * eff * cap)
                P_dis = min(P_dis, P_dis_max)

                out[a] = {
                    'P_charge': P_ch, 'P_discharge': P_dis, 'R_fcr': R_fcr,
                    'R_afrr_up': R_afrr_up, 'R_afrr_dn': R_afrr_dn,
                    'R_mfrr_up': R_mfrr_up, 'R_mfrr_dn': R_mfrr_dn,
                    'overcommit': overcommit_excess,
                }
                continue

            # ---- Legacy 5-axis (back-compat) ----
            P_ch, P_dis, R_fcr, R_afrr, R_mfrr = (f * P_max for f in fracs)
            if P_ch > 0 and P_dis > 0:
                if P_ch >= P_dis: P_dis = 0.0
                else: P_ch = 0.0
            total = P_ch + P_dis + R_fcr + R_afrr + R_mfrr
            overcommit_excess = max(0.0, total - P_free) / max(P_max, 1e-9)
            if total > P_free and total > 0:
                scale = P_free / total
                P_ch *= scale; P_dis *= scale
                R_fcr *= scale; R_afrr *= scale; R_mfrr *= scale
            P_ch_max = max(0.0, (self.soc_max - soc) * cap / eff)
            P_ch = min(P_ch, P_ch_max)
            P_dis_max = max(0.0, (soc - self.soc_min) * eff * cap)
            P_dis = min(P_dis, P_dis_max)
            out[a] = {
                'P_charge': P_ch, 'P_discharge': P_dis,
                'R_fcr': R_fcr, 'R_afrr': R_afrr, 'R_mfrr': R_mfrr,
                'overcommit': overcommit_excess,
            }

        # ---- FLEET PASS: shared connection envelope ----------------------
        # Everything above is per unit and independent. This is the only step
        # that makes one unit's admissible bid depend on what the others bid,
        # and it is therefore what turns fifty independent problems into a
        # coordination problem. Runs after the per-unit clip because scaling
        # DOWN never breaks a per-unit constraint (less power needs less
        # sustaining energy and less headroom), so the two passes compose
        # without iterating.
        self._last_coupling_diag = project_to_connection(
            out, self.coupling.connection_limit_mw)
        # Remember the fleet envelope so the global-state block can report the
        # contention the NEXT observation is formed under.
        from fleet_coupling import direction_envelopes as _envelopes
        self._last_fleet_envelope = _envelopes(out)
        return out

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _global_state(self) -> np.ndarray:
        """Fleet-level summary for the centralised critic.

        Six N-invariant statistics describing the contention for the shared
        connection and the fleet's energy posture. Identical for every agent,
        because it is the fleet's state and not the agent's.

        The envelope is normalised by the connection limit when one is set, so
        the feature reads directly as "how much of the shared cable is already
        spoken for". Without a coupling there is no contention to report and
        the total rated power is used instead, which keeps the feature bounded
        and meaningful rather than undefined.
        """
        g = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)
        n = max(self.n_agents, 1)

        limit = self.coupling.connection_limit_mw
        if limit is None or limit <= 0.0:
            limit = max(sum(b.max_power_mw
                            for b in self.multi_params.batteries), 1e-9)
        up, dn = self._last_fleet_envelope
        g[0] = min(up / limit, 2.0)
        g[1] = min(dn / limit, 2.0)

        socs = np.asarray(self._socs[:n], dtype=np.float64)
        g[2] = float(socs.mean())
        g[3] = float(socs.std())

        # A unit counts as energy-limited in a direction when the energy it can
        # still move is less than one hour at rated power: it is then the
        # binding side of the fleet's ability to serve an activation.
        lim_up = lim_dn = 0
        for i in range(n):
            bp = self.multi_params.batteries[i]
            hours_up = (socs[i] - self.soc_min) * bp.capacity_mwh / max(
                bp.max_power_mw, 1e-9)
            hours_dn = (self.soc_max - socs[i]) * bp.capacity_mwh / max(
                bp.max_power_mw, 1e-9)
            lim_up += 1 if hours_up < 1.0 else 0
            lim_dn += 1 if hours_dn < 1.0 else 0
        g[4] = lim_up / n
        g[5] = lim_dn / n
        return g

    def _build_observation(self, agent: str) -> np.ndarray:
        """Per-agent observation.

        Schema (core block; see OBS_DIM for the duration features at 26-28):
          0:  own SOC (in [0, 1])
          1:  own FCE_cumulative normalised by 8000 (capped at 2.0)
          2:  own capacity_mwh / 5.0  (utility=1.0 reference)
          3:  own max_power_mw / 2.5
          4:  hour / 24                         (current hour of day, in [0, 1])
          5:  current PUN price / 200            (normalised energy price)
          6-11: PUN price forecast next 6 hours / 200
          12:  FCR capacity price / 50
          13:  aFRR capacity price / 100
          14:  mFRR capacity price / 50
          15:  FCR energy price / 200
          16:  aFRR energy price / 300
          17:  mFRR energy price / 200
          18:  FCR award probability
          19:  aFRR award probability
          20:  mFRR award probability
        """
        i = self.agent_name_mapping[agent]
        bp = self.multi_params.batteries[i]
        t = min(self._hour, self.episode_hours - 1)

        # Allocate from the DECLARED observation space, not from a locally
        # recomputed width. Recomputing it here once cost the global-state
        # block six slots it did not own: writing obs[-6:] into a vector sized
        # without the block silently overwrote the duration features instead
        # of appending. One source of truth for the width, and the two cannot
        # disagree.
        obs = np.zeros(self._observation_space.shape[0], dtype=np.float32)
        obs[0] = self._socs[i]
        obs[1] = min(self._lfp_states[i].fce_cumulative / 8000.0, 2.0)
        obs[2] = bp.capacity_mwh / 5.0
        obs[3] = bp.max_power_mw / 2.5
        obs[4] = (self._hour % 24) / 24.0
        obs[5] = self._prices_episode[t] / 200.0
        # 6-hour forecast (clip to remaining horizon)
        for k in range(6):
            tt = min(t + 1 + k, self.episode_hours - 1)
            obs[6 + k] = self._prices_episode[tt] / 200.0

        services_by_type = {s.service_type: s for s in self._services_episode[t]}
        fcr = services_by_type.get(ServiceType.FCR)
        afrr = services_by_type.get(ServiceType.AFRR)
        mfrr = services_by_type.get(ServiceType.MFRR)
        if fcr is not None:
            obs[12] = fcr.capacity_price / 50.0
            obs[15] = fcr.energy_price / 200.0
            obs[18] = fcr.award_probability
        if afrr is not None:
            obs[13] = afrr.capacity_price / 100.0
            obs[16] = afrr.energy_price / 300.0
            obs[19] = afrr.award_probability
        if mfrr is not None:
            obs[14] = mfrr.capacity_price / 50.0
            obs[17] = mfrr.energy_price / 200.0
            obs[20] = mfrr.award_probability

        # ---- 21-27: pending-commitment features (Option 4) ----
        # For each directional service axis, how much capacity this agent has
        # already committed for delivery in the upcoming hours, normalised by
        # P_max. Zero when commitments are disabled. This makes the bid->
        # activation link observable: a bid placed now shows up here until it is
        # delivered, so the policy can (a) see the consequence building up when
        # it bids and (b) anticipate the activation when delivery approaches.
        if self.enable_commitments:
            pmax = max(bp.max_power_mw, 1e-9)
            # sum committed MW per service across all currently-pending
            # commitments for this agent (any delivery hour still in the future
            # or due now), so the agent sees its outstanding obligations.
            pend = {k: 0.0 for k in _ORDERED_SERVICE_KEYS}
            for c in self._commitments.snapshot().get(agent, []):
                if c.service in pend:
                    pend[c.service] += c.mw
            for j, k in enumerate(_ORDERED_SERVICE_KEYS):
                obs[21 + j] = min(pend[k] / pmax, 2.0)

        # ---- 26-28: duration-relative state (asset-invariant coordinates) --
        # 26: usable discharge energy, in hours at rated power
        # 27: charge headroom, in hours at rated power (efficiency-adjusted)
        # 28: the unit's storage duration, capacity / P_max, in hours
        #
        # These are the quantities the sustain constraints are written in once
        # both sides are divided by P_max, so they make the offerable fraction
        # of rated power directly readable from the state for ANY asset. Two
        # units with the same duration become the same decision problem; two
        # with different durations differ in one scalar the policy receives
        # explicitly and can condition on continuously, including durations
        # never seen in training.
        #
        # Normalised by a 4 h reference and clipped at 2.0, matching the
        # treatment of the other ratio features.
        if self.expose_duration_features:
            _pmax = max(bp.max_power_mw, 1e-9)
            _dur = bp.capacity_mwh / _pmax
            _eff = max(getattr(bp, "efficiency", 1.0), 1e-9)
            obs[26] = min((self._socs[i] - bp.soc_min) * _dur / 4.0, 2.0)
            obs[27] = min((bp.soc_max - self._socs[i]) * _dur / (4.0 * _eff),
                          2.0)
            obs[28] = min(_dur / 4.0, 2.0)

        # ---- 29+: full-foresight features (diagnostic) ----
        # Give the agent EVERYTHING the MILP sees: the true PUN price and the
        # true service (cap_price, energy_price, award_prob) for ALL 24 hours,
        # using the same normalisation as the per-hour features above. Layout:
        #   [26 : 26+24]                      -> 24h true PUN
        #   [26+24 : 26+24+24*5*3]            -> per hour h in 0..23, per service
        #                                        s in [FCR,aFRRup,aFRRdn,mFRRup,
        #                                        mFRRdn]: cap, energy, award
        # This makes the PPO a perfect-foresight agent like the MILP, isolating
        # "how close does the DRL get at equal information?" from the information
        # gap. Diagnostic only — not deployable.
        if self.centralized_critic:
            obs[-GLOBAL_STATE_DIM:] = self._global_state()

        if self.full_foresight:
            o = OBS_DIM  # start index
            H = self.episode_hours
            for h in range(_FF_HOURS):
                hh = min(h, H - 1)
                obs[o + h] = self._prices_episode[hh] / 200.0
            o += _FF_HOURS
            # service normalisation constants mirror the per-hour block
            cap_norm = {'R_fcr': 50.0, 'R_afrr_up': 100.0, 'R_afrr_dn': 100.0,
                        'R_mfrr_up': 50.0, 'R_mfrr_dn': 50.0}
            en_norm = {'R_fcr': 200.0, 'R_afrr_up': 300.0, 'R_afrr_dn': 300.0,
                       'R_mfrr_up': 200.0, 'R_mfrr_dn': 200.0}
            for h in range(_FF_HOURS):
                hh = min(h, H - 1)
                sbt = {s.service_type: s for s in self._services_episode[hh]}
                for si, key in enumerate(_ORDERED_SERVICE_KEYS):
                    st = _service_type_for(key)
                    s = sbt.get(st)
                    base_i = o + h * (_FF_N_SERVICES * 3) + si * 3
                    if s is not None:
                        obs[base_i + 0] = min(s.capacity_price / cap_norm[key], 2.0)
                        obs[base_i + 1] = min(s.energy_price / en_norm[key], 2.0)
                        obs[base_i + 2] = s.award_probability
        # Safety clip to the observation Box bounds. Several normalised features
        # (energy price /200, FCR cap price /50, etc.) can exceed 2.0 on real
        # Italian price spikes (PUN > 400 EUR/MWh, high MSD prices). Without this
        # clip the observation falls outside the declared Box[-2,2], which the
        # RLlib connector validates and can reject. Clipping preserves ordering
        # (a spike still reads as "max") while keeping the vector in-space. This
        # also fixes a latent issue in the BASE features that predates
        # full_foresight (indices 5-15 could hit 2.5-10 on price spikes).
        np.clip(obs, -2.0, 2.0, out=obs)
        return obs

    # ------------------------------------------------------------------
    # Defaults for prices and services (test-friendly synthetic data)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_prices(hours: int) -> List[float]:
        return [80.0 + 40.0 * math.sin(2 * math.pi * (t - 6) / 24) for t in range(hours)]

    @staticmethod
    def _default_services(hours: int) -> List[List[FlexibilityService]]:
        return [
            [
                FlexibilityService(
                    service_type=ServiceType.FCR, capacity_price=20.0,
                    energy_price=80.0, activation_probability=0.10,
                    response_time=30, min_capacity=1.0,
                    max_duration=4, min_duration=1, award_probability=1.0,
                ),
                FlexibilityService(
                    service_type=ServiceType.AFRR, capacity_price=55.0,
                    energy_price=120.0, activation_probability=0.30,
                    response_time=200, min_capacity=1.0,
                    max_duration=4, min_duration=1, award_probability=1.0,
                ),
                FlexibilityService(
                    service_type=ServiceType.MFRR, capacity_price=15.0,
                    energy_price=100.0, activation_probability=0.20,
                    response_time=900, min_capacity=1.0,
                    max_duration=4, min_duration=1, award_probability=1.0,
                ),
            ]
            for _ in range(hours)
        ]

    def render(self):  # PettingZoo expected hook; we don't render
        pass

    def close(self):
        pass