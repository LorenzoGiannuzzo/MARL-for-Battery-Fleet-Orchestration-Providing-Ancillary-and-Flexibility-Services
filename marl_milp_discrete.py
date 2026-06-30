"""
Discrete-action multi-BESS MILP (full-foresight oracle on the PPO action grid).

This is the THIRD baseline for the Energy Informatics paper, sitting between
the continuous-action MILP oracle and the warm-started PPO:

    1. MILP-continuous (full foresight)  -> theoretical upper bound, continuous
    2. MILP-discrete   (full foresight)  -> THIS FILE: oracle on the SAME
                                            discrete action space as the PPO
    3. warm-PPO        (no foresight)    -> closed-loop, discrete, stochastic

Why it exists
-------------
The pipeline (marl_pipeline.generate_multi_day_expert_demos) evaluates the
MILP by solving it in CONTINUOUS power and then PROJECTING the solution onto
the env's 40-bin grid via `_discretise_milp_action_directional`. That post-hoc
projection handicaps the MILP: the continuous optimum, once rounded to bins and
fed through the env's rescale/clip, is no longer optimal for the discrete world
it is scored in. A reviewer would (correctly) attribute part of the MILP->PPO
gap to this projection artefact rather than to the DRL.

This optimiser removes the artefact. It chooses, for every (BESS, hour, axis),
ONE bin from the exact PPO grid

        level_k = P_max_i * k / (N_ACTION_BINS - 1),   k in {0, ..., 39}

as native MILP one-hot variables, under perfect price foresight. The remaining
gap to the PPO is then attributable to closed-loop adaptation to realised
stochasticity, not to discretisation handicap.

Action-axis mapping (directional 7-axis, matching marl_env and
marl_bc._discretise_milp_action_directional):

    axis 0: charge      -> P_charge
    axis 1: discharge   -> P_discharge
    axis 2: FCR         -> R_fcr
    axis 3: aFRR_up     -> R_afrr_up
    axis 4: aFRR_dn     -> R_afrr_dn
    axis 5: mFRR_up     -> R_mfrr_up
    axis 6: mFRR_dn     -> R_mfrr_dn

Feasibility model
-----------------
The env (`MultiBESSEnv._decode_clip_actions`) accepts ANY combination of bins
and then PROJECTS it onto the feasible set (charge/discharge mutex, direction
mutex, capacity-balance rescale to P_free, and per-service sustain clips).

This optimiser instead enforces feasibility BY CONSTRUCTION: it only ever
selects bin combinations whose decoded power already satisfies the
capacity-balance and sustain constraints, so no env-side rescale/clip ever
fires on its actions. Consequence (state this in the paper): the discrete MILP
never proposes an infeasible action, so any residual gap to the PPO is NOT due
to infeasible-action projection. This is the conservative, defensible choice.

Because feasibility is enforced as MILP constraints, the decoded power that
this optimiser reports is exactly what the env would apply (the env's
projection is the identity on a feasible action), so a roll-out of these
actions reproduces the MILP's planned schedule up to the stochastic realisation
of awards/activations.

Computational note
------------------
This adds N * T * 7 * N_ACTION_BINS binary one-hot variables. For the 50-BESS
fleet over 24 h with 40 bins that is 50*24*7*40 = 336000 binaries per day,
which CBC will generally NOT solve to proven optimality inside the 300 s limit.
That is acceptable and, in fact, part of the story: report the best incumbent
and the optimality gap. Two ways to get a proven optimum if needed:
  - swap the solver for Gurobi (academic licence) via `solver=`;
  - run the exact discrete MILP on a reduced representative sub-fleet for the
    optimality argument, and use the 50-BESS fleet for the env roll-out.

Author: Lorenzo Giannuzzo, DENERG Politecnico di Torino (multi-agent branch).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pulp

from marl_milp_continuous import (
    MultiBatteryParameters,
    MultiBESSMILPOptimizer,
)

# Import the canonical grid constants from the env so the MILP grid and the
# PPO grid can NEVER drift apart. If the env changes N_ACTION_BINS, this file
# follows automatically.
from marl_env import N_ACTION_BINS


# ============================================================================
# Axis <-> MILP power-variable mapping
# ============================================================================

# Directional 7-axis layout. Order MUST match
# _discretise_milp_action_directional and _decode_clip_actions.
#   (axis_index, milp_power_variable_name)
_DIRECTIONAL_AXIS_TO_VAR: List[str] = [
    'P_charge',     # axis 0
    'P_discharge',  # axis 1
    'R_fcr',        # axis 2
    'R_afrr_up',    # axis 3
    'R_afrr_dn',    # axis 4
    'R_mfrr_up',    # axis 5
    'R_mfrr_dn',    # axis 6
]

# Legacy 5-axis layout (kept for completeness; this file targets directional).
_LEGACY_AXIS_TO_VAR: List[str] = [
    'P_charge', 'P_discharge', 'R_fcr', 'R_afrr', 'R_mfrr',
]


class MultiBESSMILPOptimizerDiscrete(MultiBESSMILPOptimizer):
    """Multi-BESS MILP whose 7 power axes are constrained to the PPO bin grid.

    Subclasses the continuous optimiser and overrides ONLY:
      * `_create_variables`     -> add per-(BESS, hour, axis) one-hot bin vars
      * `_add_battery_constraints` -> add the bin->power linking constraints,
                                       then delegate to the continuous model for
                                       every other constraint (SOC dynamics,
                                       capacity balance, sustain, direction
                                       mutex, A_s <= R_s, ...).

    Everything else (objective, ARERA aggregate, degradation, extract_results,
    the hard-timeout `optimize()` wrapper) is inherited UNCHANGED, so the
    discrete oracle optimises the *identical* objective and obeys the
    *identical* physics as the continuous oracle - only the admissible power
    levels differ.
    """

    def __init__(self, *args, n_action_bins: int = N_ACTION_BINS, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_action_bins = int(n_action_bins)
        if self.n_action_bins < 2:
            raise ValueError("n_action_bins must be >= 2")

        # Select the axis->variable map for the active mode. This file is
        # written for directional; legacy is supported for symmetry.
        if self.directional_services:
            self._axis_to_var = _DIRECTIONAL_AXIS_TO_VAR
        else:
            self._axis_to_var = _LEGACY_AXIS_TO_VAR

        # Sanity: every mapped variable must be a real power axis of the model.
        # P_charge/P_discharge always exist; the R_* must match _service_keys.
        valid_R = {f'R_{s}' for s in self._service_keys}
        for var_name in self._axis_to_var:
            if var_name in ('P_charge', 'P_discharge'):
                continue
            if var_name not in valid_R:
                raise ValueError(
                    f"Axis variable {var_name!r} is not an active service axis "
                    f"for this mode (active R vars: {sorted(valid_R)}). "
                    f"directional_services={self.directional_services}."
                )

    # ------------------------------------------------------------------
    # Bin grid helpers
    # ------------------------------------------------------------------

    def _bin_level_fraction(self, k: int) -> float:
        """Fraction of P_max for bin k, matching env `_BIN_VALUES`.

        Env uses np.linspace(0, 1, N_ACTION_BINS), i.e. k / (N_ACTION_BINS - 1).
        k = 0 -> 0.0 (do-nothing on that axis); k = N-1 -> 1.0 (full P_max).
        """
        return k / (self.n_action_bins - 1)

    # ------------------------------------------------------------------
    # Variable creation (override)
    # ------------------------------------------------------------------

    def _create_variables(self, time_horizon: int) -> Dict:
        # Build all the continuous-model variables first (SOC, P_*, R_*, A_*,
        # mutex binaries, direction binaries, aggregate binaries). The P_* and
        # R_* continuous vars are RETAINED and become the "decoded power" of
        # the chosen bins via the linking constraints added below.
        v = super()._create_variables(time_horizon)

        N = self.multi_params.n_batteries
        T = time_horizon
        Kn = self.n_action_bins

        # One-hot bin selectors per (BESS, hour, axis).
        #   b[(axis_var, i, t, k)] in {0,1},  sum_k b == 1.
        # We index by the axis VARIABLE NAME (e.g. 'R_fcr') so the linking
        # constraint can find the matching continuous power variable directly.
        for axis_var in self._axis_to_var:
            v[f'bin_{axis_var}'] = {
                (i, t, k): pulp.LpVariable(
                    f"bin_{axis_var}_{i}_{t}_{k}", cat='Binary'
                )
                for i in range(N) for t in range(T) for k in range(Kn)
            }

        return v

    # ------------------------------------------------------------------
    # Constraints (override): add bin<->power linking, then inherit the rest
    # ------------------------------------------------------------------

    def _add_battery_constraints(self, v: Dict, T: int):
        # 1) Link each power axis to its one-hot bin grid BEFORE the physics
        #    constraints, so the continuous P_*/R_* are pinned to grid levels.
        N = self.multi_params.n_batteries
        Kn = self.n_action_bins

        for i, bp in enumerate(self.multi_params.batteries):
            P_max = bp.max_power_mw
            for t in range(T):
                for axis_var in self._axis_to_var:
                    bin_vars = v[f'bin_{axis_var}']
                    power_var = v[axis_var][(i, t)]

                    # Exactly one bin selected per axis per (BESS, hour).
                    self.problem += (
                        pulp.lpSum(bin_vars[(i, t, k)] for k in range(Kn)) == 1
                    ), f"OneHot_{axis_var}_{i}_{t}"

                    # Decoded power = P_max * sum_k (k/(Kn-1)) * b_k.
                    # This pins the continuous power variable to the discrete
                    # grid level of the selected bin. Because k=0 maps to 0.0,
                    # selecting bin 0 means "do nothing on this axis", exactly
                    # like the PPO.
                    self.problem += (
                        power_var == P_max * pulp.lpSum(
                            self._bin_level_fraction(k) * bin_vars[(i, t, k)]
                            for k in range(Kn)
                        )
                    ), f"BinDecode_{axis_var}_{i}_{t}"

        # 2) Delegate to the continuous model for ALL remaining physics:
        #    SOC dynamics (directional), charge/discharge mutex, per-BESS
        #    capacity balance (sum of axes <= P_max), FCR symmetric sustain,
        #    directional sustain + direction mutex, and A_s <= R_s. These act
        #    on the SAME P_*/R_* variables we just pinned to the grid, so the
        #    feasible set is exactly "grid levels that also satisfy the physics"
        #    -> feasible-by-construction discrete actions.
        super()._add_battery_constraints(v, T)

        # NOTE on env parity: the env (_decode_clip_actions) would additionally
        # rescale an over-budget action down to P_free and clip service reserves
        # to the sustain caps. Here those projections can NEVER trigger, because
        # the capacity-balance and sustain constraints inherited above forbid
        # selecting any bin combination that would exceed them. The decoded
        # power is therefore identical to what the env would apply.


# ============================================================================
# Convenience factory
# ============================================================================

def build_discrete_oracle(
    multi_params: MultiBatteryParameters,
    *,
    directional_services: bool = True,
    fcr_sustain_hours: float = 4.0,
    afrr_sustain_hours: float = 1.0,
    mfrr_sustain_hours: float = 2.0,
    use_nonlinear_degradation: bool = False,
    enable_commitments: bool = False,
    penalty_k: float = 1.5,
    n_action_bins: int = N_ACTION_BINS,
    solver: Optional[object] = None,
) -> MultiBESSMILPOptimizerDiscrete:
    """Construct a discrete-action MILP oracle aligned with the directional env.

    The sustain defaults (FCR 4 h, aFRR 1 h, mFRR 2 h) match
    `marl_env._decode_clip_actions`, so the discrete MILP plans in the
    SAME world the env evaluates in. Pass a Gurobi `solver` here to get proven
    optimality on the discrete problem instead of CBC's best incumbent.
    """
    opt = MultiBESSMILPOptimizerDiscrete(
        multi_params,
        use_nonlinear_degradation=use_nonlinear_degradation,
        directional_services=directional_services,
        fcr_sustain_hours=fcr_sustain_hours,
        afrr_sustain_hours=afrr_sustain_hours,
        mfrr_sustain_hours=mfrr_sustain_hours,
        enable_commitments=enable_commitments,
        penalty_k=penalty_k,
        n_action_bins=n_action_bins,
    )
    if solver is not None:
        opt.solver = solver
    return opt