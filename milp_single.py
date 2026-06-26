"""
MILP Optimizer for Battery Energy Storage System
Implements Mixed Integer Linear Programming optimization for simultaneous
arbitrage and flexibility services according to Italian market regulations
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pulp
import copy
from markets import FlexibilityMarket, FlexibilityService, BatteryState, ServiceType
from market_config import ITALIAN_CONFIG


@dataclass
class BatteryParameters:
    """Battery system parameters for MILP optimization"""
    capacity_mwh: float = 4.0  # Battery capacity in MWh
    max_power_mw: float = 2.0  # Maximum charge/discharge power in MW
    efficiency: float = 0.95  # Round-trip efficiency
    soc_min: float = 0.1  # Minimum SOC (10%)
    soc_max: float = 0.9  # Maximum SOC (90%)
    initial_soc: float = 0.5  # Initial SOC (50%)
    degradation_cost_per_mwh: float = 25000.0  # EUR/MWh of replacement capacity cost
    cycle_life: int = 6000  # Equivalent full cycles to end-of-life (matches PPO env)
    # BUG FIX: marginal degradation cost = degradation_cost_per_mwh / (2 * cycle_life)
    # The factor (2 * cycle_life) amortizes the replacement cost over the total energy
    # throughput in lifetime (each full cycle = 2 * capacity_mwh of throughput).
    # The original MILP missed this factor and over-penalized degradation by 12000x
    # relative to the PPO environment in legacy_degradation.py.


@dataclass
class OptimizationResult:
    """Results from MILP optimization"""
    algorithm: str  # 'MILP'
    arbitrage_profit: float
    flexibility_revenue: float
    degradation_cost: float
    net_profit: float
    execution_time: float
    battery_utilization: float
    final_soh: float
    soc_trajectory: List[float]
    power_trajectory: List[float]
    flexibility_reservations: Dict[str, List[float]]
    flexibility_activations: Dict[str, List[float]]
    solver_status: str
    objective_value: float
    # Per-service realized flexibility revenue (capacity + expected activation
    # energy), computed with the same formula as `flexibility_revenue` so that
    # FCR + aFRR + mFRR == flexibility_revenue exactly. Added May 2026 to make
    # the MILP service breakdown consistent with the PPO side (which already
    # reports capacity + activation per service from the env roll-out).
    flexibility_revenue_by_service: Dict[str, float] = None


@dataclass
class RollingHorizonConfig:
    """Configuration for rolling horizon optimization"""
    horizon_length: int = 24  # Length of each optimization window (hours)
    overlap_length: int = 6  # Overlap between consecutive windows (hours)
    step_size: int = 18  # Step size between windows (hours) = horizon_length - overlap_length
    max_iterations: int = 100  # Maximum number of rolling horizon iterations
    convergence_tolerance: float = 0.01  # Convergence tolerance for objective value
    reoptimization_threshold: float = 0.05  # Threshold for triggering reoptimization due to forecast errors


@dataclass
class RollingHorizonResult:
    """Results from rolling horizon optimization"""
    total_iterations: int
    converged: bool
    final_objective_value: float
    execution_time: float
    optimization_results: List[OptimizationResult]
    forecast_updates: List[int]  # Time steps where forecast updates occurred
    reoptimizations: List[int]  # Time steps where reoptimization was triggered


class MILPOptimizer:
    """
    Mixed Integer Linear Programming optimizer for BESS
    Optimizes simultaneous arbitrage and flexibility services
    """

    def __init__(self, battery_params: BatteryParameters, markets: FlexibilityMarket):
        """
        Initialize MILP optimizer

        Args:
            battery_params: Battery system parameters
            markets: Flexibility market model (should be deterministic)
        """
        self.battery_params = battery_params
        self.markets = markets

        # Solver configuration
        self.solver = pulp.PULP_CBC_CMD(msg=0)  # Use CBC solver with no output

        # Optimization problem
        self.problem = None

        # Decision variables (will be created during optimization)
        self.variables = {}

        # Constraints tracking
        self.constraints = {}

        # Rolling horizon state
        self.current_soc = battery_params.initial_soc
        self.rolling_horizon_config = None

    def create_decision_variables(self, time_horizon: int) -> Dict:
        """
        Create MILP decision variables for the optimization problem

        Args:
            time_horizon: Number of time steps (hours) to optimize

        Returns:
            Dictionary containing all decision variables
        """
        variables = {}

        # Battery operation variables
        variables['P_charge'] = pulp.LpVariable.dicts(
            "P_charge", range(time_horizon),
            lowBound=0, upBound=self.battery_params.max_power_mw, cat='Continuous'
        )

        variables['P_discharge'] = pulp.LpVariable.dicts(
            "P_discharge", range(time_horizon),
            lowBound=0, upBound=self.battery_params.max_power_mw, cat='Continuous'
        )

        variables['SOC'] = pulp.LpVariable.dicts(
            "SOC", range(time_horizon + 1),  # +1 for initial SOC
            lowBound=self.battery_params.soc_min,
            upBound=self.battery_params.soc_max, cat='Continuous'
        )

        # Flexibility service reservation variables
        variables['R_fcr'] = pulp.LpVariable.dicts(
            "R_fcr", range(time_horizon),
            lowBound=0, upBound=self.battery_params.max_power_mw, cat='Continuous'
        )

        variables['R_afrr'] = pulp.LpVariable.dicts(
            "R_afrr", range(time_horizon),
            lowBound=0, upBound=self.battery_params.max_power_mw, cat='Continuous'
        )

        variables['R_mfrr'] = pulp.LpVariable.dicts(
            "R_mfrr", range(time_horizon),
            lowBound=0, upBound=self.battery_params.max_power_mw, cat='Continuous'
        )

        # Flexibility service activation variables
        variables['A_fcr'] = pulp.LpVariable.dicts(
            "A_fcr", range(time_horizon),
            lowBound=0, cat='Continuous'
        )

        variables['A_afrr'] = pulp.LpVariable.dicts(
            "A_afrr", range(time_horizon),
            lowBound=0, cat='Continuous'
        )

        variables['A_mfrr'] = pulp.LpVariable.dicts(
            "A_mfrr", range(time_horizon),
            lowBound=0, cat='Continuous'
        )

        # Binary variables for charge/discharge mutual exclusion
        variables['B_charge'] = pulp.LpVariable.dicts(
            "B_charge", range(time_horizon), cat='Binary'
        )

        variables['B_discharge'] = pulp.LpVariable.dicts(
            "B_discharge", range(time_horizon), cat='Binary'
        )

        # Block 4 (May 2026): binary participation indicators per service per
        # hour. Used to enforce ARERA minimum-capacity rules via a big-M
        # formulation in `add_market_constraints`:
        #   R_s,t  in  {0}  U  [min_cap_s, max_power]
        # implemented as
        #   R_s,t <= max_power * B_s,t
        #   R_s,t >= min_cap_s * B_s,t
        # so that B_s,t = 0 forces R_s,t = 0, while B_s,t = 1 forces
        # R_s,t >= min_cap_s.
        variables['B_fcr'] = pulp.LpVariable.dicts(
            "B_fcr", range(time_horizon), cat='Binary'
        )
        variables['B_afrr'] = pulp.LpVariable.dicts(
            "B_afrr", range(time_horizon), cat='Binary'
        )
        variables['B_mfrr'] = pulp.LpVariable.dicts(
            "B_mfrr", range(time_horizon), cat='Binary'
        )

        return variables

    def add_battery_constraints(self, variables: Dict, time_horizon: int):
        """
        Add battery physical and operational constraints

        Args:
            variables: Dictionary of decision variables
            time_horizon: Number of time steps
        """
        constraints = []

        # Initial SOC constraint
        constraints.append(
            variables['SOC'][0] == self.battery_params.initial_soc
        )

        # SOC evolution constraints (energy balance)
        for t in range(time_horizon):
            constraints.append(
                variables['SOC'][t + 1] == variables['SOC'][t] +
                (self.battery_params.efficiency * variables['P_charge'][t] -
                 variables['P_discharge'][t] / self.battery_params.efficiency) /
                self.battery_params.capacity_mwh
            )

        # Power capacity constraints considering flexibility reservations
        for t in range(time_horizon):
            # Total power usage cannot exceed battery capacity
            constraints.append(
                variables['P_charge'][t] + variables['R_fcr'][t] +
                variables['R_afrr'][t] + variables['R_mfrr'][t] <=
                self.battery_params.max_power_mw
            )

            constraints.append(
                variables['P_discharge'][t] + variables['R_fcr'][t] +
                variables['R_afrr'][t] + variables['R_mfrr'][t] <=
                self.battery_params.max_power_mw
            )

        # Mutual exclusion: cannot charge and discharge simultaneously
        for t in range(time_horizon):
            constraints.append(
                variables['B_charge'][t] + variables['B_discharge'][t] <= 1
            )

            # Link binary variables to continuous variables
            constraints.append(
                variables['P_charge'][t] <=
                self.battery_params.max_power_mw * variables['B_charge'][t]
            )

            constraints.append(
                variables['P_discharge'][t] <=
                self.battery_params.max_power_mw * variables['B_discharge'][t]
            )

        # Flexibility service activation constraints
        for t in range(time_horizon):
            constraints.append(variables['A_fcr'][t] <= variables['R_fcr'][t])
            constraints.append(variables['A_afrr'][t] <= variables['R_afrr'][t])
            constraints.append(variables['A_mfrr'][t] <= variables['R_mfrr'][t])

        # SOC constraints for flexibility services (symmetric capability)
        for t in range(time_horizon):
            # FCR requires symmetric capability - need SOC margin for both directions
            # If providing FCR, SOC must allow for potential activation in both directions
            max_fcr_energy = variables['R_fcr'][t] * 0.5  # Assume max 30min activation

            constraints.append(
                variables['SOC'][t] >= self.battery_params.soc_min +
                max_fcr_energy / self.battery_params.capacity_mwh
            )

            constraints.append(
                variables['SOC'][t] <= self.battery_params.soc_max -
                max_fcr_energy / self.battery_params.capacity_mwh
            )

        # Add all constraints to the problem
        for i, constraint in enumerate(constraints):
            self.problem += constraint, f"Battery_Constraint_{i}"

        self.constraints['battery'] = constraints

    def add_market_constraints(self, variables: Dict, time_horizon: int,
                               flexibility_services: List[List[FlexibilityService]]):
        """
        Add market-specific constraints for flexibility services.

        Block 4 (May 2026): the ARERA minimum-capacity rule
        (typically 1 MW per service) is now enforced via a big-M
        formulation. For each service s in {FCR, aFRR, mFRR} and each
        hour t:

            R_s,t in {0}  U  [min_cap_s, P_max]

        is expressed linearly as

            R_s,t <= P_max * B_s,t           (B_s,t = 0  =>  R_s,t = 0)
            R_s,t >= min_cap_s * B_s,t       (B_s,t = 1  =>  R_s,t >= min_cap_s)

        where B_s,t is the binary participation indicator created in
        `create_decision_variables`. When the service is unavailable in a
        given hour (no FlexibilityService entry for that t), we force
        B_s,t = 0, which in turn forces R_s,t = 0.

        Args:
            variables: Dictionary of decision variables.
            time_horizon: Number of time steps.
            flexibility_services: List of available services per hour.
        """
        constraints = []
        P_max = float(self.battery_params.max_power_mw)

        for t in range(time_horizon):
            services = flexibility_services[t]
            fcr_service  = next((s for s in services if s.service_type == ServiceType.FCR),  None)
            afrr_service = next((s for s in services if s.service_type == ServiceType.AFRR), None)
            mfrr_service = next((s for s in services if s.service_type == ServiceType.MFRR), None)

            # ---------------- FCR ----------------
            if fcr_service is not None:
                m_fcr = float(fcr_service.min_capacity)
                # Upper bound: R_fcr <= P_max * B_fcr
                constraints.append(
                    variables['R_fcr'][t] <= P_max * variables['B_fcr'][t]
                )
                # Lower bound: R_fcr >= min_cap * B_fcr
                constraints.append(
                    variables['R_fcr'][t] >= m_fcr * variables['B_fcr'][t]
                )
            else:
                # Service not available this hour: force R_fcr = 0
                constraints.append(variables['B_fcr'][t] == 0)
                constraints.append(variables['R_fcr'][t] == 0)

            # ---------------- aFRR ----------------
            if afrr_service is not None:
                m_afrr = float(afrr_service.min_capacity)
                constraints.append(
                    variables['R_afrr'][t] <= P_max * variables['B_afrr'][t]
                )
                constraints.append(
                    variables['R_afrr'][t] >= m_afrr * variables['B_afrr'][t]
                )
            else:
                constraints.append(variables['B_afrr'][t] == 0)
                constraints.append(variables['R_afrr'][t] == 0)

            # ---------------- mFRR ----------------
            if mfrr_service is not None:
                m_mfrr = float(mfrr_service.min_capacity)
                constraints.append(
                    variables['R_mfrr'][t] <= P_max * variables['B_mfrr'][t]
                )
                constraints.append(
                    variables['R_mfrr'][t] >= m_mfrr * variables['B_mfrr'][t]
                )
            else:
                constraints.append(variables['B_mfrr'][t] == 0)
                constraints.append(variables['R_mfrr'][t] == 0)

        # Register all constraints with the problem
        for i, constraint in enumerate(constraints):
            self.problem += constraint, f"Market_Constraint_{i}"

        self.constraints['market'] = constraints

    def create_objective_function(self, variables: Dict, time_horizon: int,
                                  energy_prices: List[float],
                                  flexibility_services: List[List[FlexibilityService]]):
        """
        Create the objective function for profit maximization
        Integrates arbitrage profits, flexibility service revenues, and battery degradation costs

        Args:
            variables: Dictionary of decision variables
            time_horizon: Number of time steps
            energy_prices: Energy prices for each time step (EUR/MWh)
            flexibility_services: Available flexibility services for each time step

        Returns:
            PuLP expression for the objective function
        """
        objective = 0

        for t in range(time_horizon):
            # ==================== ARBITRAGE COMPONENT ====================
            # Revenue from discharging (selling energy) minus cost of charging (buying energy)
            arbitrage_profit = (
                    variables['P_discharge'][t] * energy_prices[t] -  # Revenue from discharge
                    variables['P_charge'][t] * energy_prices[t]  # Cost of charge
            )
            objective += arbitrage_profit

            # ==================== FLEXIBILITY SERVICES REVENUE ====================
            services = flexibility_services[t]

            for service in services:
                # June 2026: capacity auctions are not guaranteed to clear.
                # Each per-hour service has an `award_probability` in [0,1]
                # that scales the EXPECTED capacity and energy revenue. The
                # capacity-balance constraints remain binding always (the bid
                # commits the power) so the MILP must trade off the expected
                # service revenue against the certain arbitrage opportunity
                # foregone by reserving the capacity.
                aw = service.award_probability
                if service.service_type == ServiceType.FCR:
                    capacity_revenue = variables['R_fcr'][t] * service.capacity_price * aw
                    energy_revenue = (variables['A_fcr'][t] * service.energy_price *
                                      service.activation_probability * aw)
                    objective += capacity_revenue + energy_revenue

                elif service.service_type == ServiceType.AFRR:
                    capacity_revenue = variables['R_afrr'][t] * service.capacity_price * aw
                    energy_revenue = (variables['A_afrr'][t] * service.energy_price *
                                      service.activation_probability * aw)
                    objective += capacity_revenue + energy_revenue

                elif service.service_type == ServiceType.MFRR:
                    capacity_revenue = variables['R_mfrr'][t] * service.capacity_price * aw
                    energy_revenue = (variables['A_mfrr'][t] * service.energy_price *
                                      service.activation_probability * aw)
                    objective += capacity_revenue + energy_revenue

            # ==================== BATTERY DEGRADATION COST ====================
            # Degradation cost based on total energy throughput.
            # BUG FIX: amortize replacement cost over (2 * cycle_life) MWh of throughput.
            # This matches the formula used in legacy_degradation.py and in the
            # extended/enhanced PPO environments:
            #     degradation_cost = MWh * degradation_cost_per_mwh / (2 * cycle_life)

            # Total energy throughput for this time step (MWh)
            energy_throughput = variables['P_charge'][t] + variables['P_discharge'][t]

            # Add flexibility service activations to throughput
            # (activations also contribute to battery wear)
            flexibility_throughput = (variables['A_fcr'][t] +
                                      variables['A_afrr'][t] +
                                      variables['A_mfrr'][t])

            total_throughput = energy_throughput + flexibility_throughput

            # Apply marginal degradation cost per MWh of throughput
            marginal_cost_per_mwh = (self.battery_params.degradation_cost_per_mwh /
                                     (2.0 * self.battery_params.cycle_life))
            degradation_cost = total_throughput * marginal_cost_per_mwh

            objective -= degradation_cost

        return objective

    def optimize(self, time_horizon: int, energy_prices: List[float],
                 flexibility_services: List[List[FlexibilityService]]) -> OptimizationResult:
        """
        Solve the MILP optimization problem

        Args:
            time_horizon: Number of time steps to optimize
            energy_prices: Energy prices for each time step (EUR/MWh)
            flexibility_services: Available flexibility services for each time step

        Returns:
            OptimizationResult with solution details
        """
        import time
        start_time = time.time()

        # Create optimization problem
        self.problem = pulp.LpProblem("BESS_Optimization", pulp.LpMaximize)

        # Create decision variables
        variables = self.create_decision_variables(time_horizon)
        self.variables = variables

        # Add constraints
        self.add_battery_constraints(variables, time_horizon)
        self.add_market_constraints(variables, time_horizon, flexibility_services)

        # Create and set objective function
        objective = self.create_objective_function(variables, time_horizon,
                                                   energy_prices, flexibility_services)
        self.problem += objective

        # Solve the problem
        self.problem.solve(self.solver)

        execution_time = time.time() - start_time

        # Extract results
        if self.problem.status == pulp.LpStatusOptimal:
            return self._extract_results(variables, time_horizon, execution_time,
                                         energy_prices, flexibility_services)
        else:
            # Return empty result if optimization failed
            return OptimizationResult(
                algorithm="MILP",
                arbitrage_profit=0.0,
                flexibility_revenue=0.0,
                degradation_cost=0.0,
                net_profit=0.0,
                execution_time=execution_time,
                battery_utilization=0.0,
                final_soh=1.0,
                soc_trajectory=[],
                power_trajectory=[],
                flexibility_reservations={},
                flexibility_activations={},
                solver_status=pulp.LpStatus[self.problem.status],
                objective_value=0.0,
                flexibility_revenue_by_service={'FCR': 0.0, 'aFRR': 0.0, 'mFRR': 0.0},
            )

    def _extract_results(self, variables: Dict, time_horizon: int,
                         execution_time: float, energy_prices: List[float],
                         flexibility_services: List[List[FlexibilityService]]) -> OptimizationResult:
        """
        Extract optimization results from solved problem

        Args:
            variables: Dictionary of decision variables
            time_horizon: Number of time steps
            execution_time: Time taken to solve
            energy_prices: Energy prices used in optimization
            flexibility_services: Flexibility services used in optimization

        Returns:
            OptimizationResult with extracted values
        """
        # Extract SOC trajectory
        soc_trajectory = [variables['SOC'][t].varValue for t in range(time_horizon + 1)]

        # Extract power trajectory (net power: positive = discharge, negative = charge)
        power_trajectory = [
            variables['P_discharge'][t].varValue - variables['P_charge'][t].varValue
            for t in range(time_horizon)
        ]

        # Extract flexibility reservations
        flexibility_reservations = {
            'FCR': [variables['R_fcr'][t].varValue for t in range(time_horizon)],
            'aFRR': [variables['R_afrr'][t].varValue for t in range(time_horizon)],
            'mFRR': [variables['R_mfrr'][t].varValue for t in range(time_horizon)]
        }

        # Extract flexibility activations
        flexibility_activations = {
            'FCR': [variables['A_fcr'][t].varValue for t in range(time_horizon)],
            'aFRR': [variables['A_afrr'][t].varValue for t in range(time_horizon)],
            'mFRR': [variables['A_mfrr'][t].varValue for t in range(time_horizon)]
        }

        # ==================== CALCULATE REVENUE COMPONENTS ====================

        # Calculate arbitrage profit
        arbitrage_profit = 0.0
        for t in range(time_horizon):
            discharge_revenue = variables['P_discharge'][t].varValue * energy_prices[t]
            charge_cost = variables['P_charge'][t].varValue * energy_prices[t]
            arbitrage_profit += discharge_revenue - charge_cost

        # Calculate flexibility service revenues (REALIZED payment, computed
        # against the TRUE settlement prices, consistently with the
        # imperfect-foresight regime for energy arbitrage).
        # We also accumulate the same quantity split per service so that the
        # downstream breakdown is exactly consistent with the total (the sum
        # of the three service entries equals flexibility_revenue).
        flexibility_revenue = 0.0
        flex_rev_by_service = {'FCR': 0.0, 'aFRR': 0.0, 'mFRR': 0.0}
        for t in range(time_horizon):
            services = flexibility_services[t]

            for service in services:
                # Use true settlement prices for the realized payment.
                # Fall back to forecast prices if true prices are not set
                # (backward compatibility with deterministic regime).
                true_cap = (service.true_capacity_price
                            if service.true_capacity_price is not None
                            else service.capacity_price)
                true_en = (service.true_energy_price
                           if service.true_energy_price is not None
                           else service.energy_price)
                # June 2026: capacity bids may not clear. The REALIZED MILP
                # revenue (used to compute flexibility_revenue and the
                # per-service breakdown) is the expected value over the
                # auction outcome, consistent with the MILP objective.
                aw = service.award_probability

                if service.service_type == ServiceType.FCR:
                    capacity_rev = variables['R_fcr'][t].varValue * true_cap * aw
                    energy_rev = (variables['A_fcr'][t].varValue * true_en *
                                  service.activation_probability * aw)
                    flexibility_revenue += capacity_rev + energy_rev
                    flex_rev_by_service['FCR'] += capacity_rev + energy_rev

                elif service.service_type == ServiceType.AFRR:
                    capacity_rev = variables['R_afrr'][t].varValue * true_cap * aw
                    energy_rev = (variables['A_afrr'][t].varValue * true_en *
                                  service.activation_probability * aw)
                    flexibility_revenue += capacity_rev + energy_rev
                    flex_rev_by_service['aFRR'] += capacity_rev + energy_rev

                elif service.service_type == ServiceType.MFRR:
                    capacity_rev = variables['R_mfrr'][t].varValue * true_cap * aw
                    energy_rev = (variables['A_mfrr'][t].varValue * true_en *
                                  service.activation_probability * aw)
                    flexibility_revenue += capacity_rev + energy_rev
                    flex_rev_by_service['mFRR'] += capacity_rev + energy_rev

        # Calculate degradation cost
        # BUG FIX: same marginal formula as in the objective function
        total_energy_throughput = 0.0
        for t in range(time_horizon):
            # Arbitrage throughput
            arbitrage_throughput = (variables['P_charge'][t].varValue +
                                    variables['P_discharge'][t].varValue)

            # Flexibility service throughput
            flexibility_throughput = (variables['A_fcr'][t].varValue +
                                      variables['A_afrr'][t].varValue +
                                      variables['A_mfrr'][t].varValue)

            total_energy_throughput += arbitrage_throughput + flexibility_throughput

        marginal_cost_per_mwh = (self.battery_params.degradation_cost_per_mwh /
                                 (2.0 * self.battery_params.cycle_life))
        degradation_cost = total_energy_throughput * marginal_cost_per_mwh

        # Calculate performance metrics
        battery_utilization = total_energy_throughput / (
                self.battery_params.max_power_mw * time_horizon * 2
        )  # Normalized by maximum possible throughput

        # Calculate SOH using existing degradation model
        # Import the degradation function from the existing system
        try:
            # Try to import from the main analysis file
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from legacy_degradation import degradation

            # Calculate equivalent cycles
            cycles_equivalent = total_energy_throughput / (2 * self.battery_params.capacity_mwh)
            final_soh = degradation(cycles_equivalent) / 100.0  # Convert percentage to fraction

        except ImportError:
            # Fallback to simplified linear degradation model if import fails
            cycles_equivalent = total_energy_throughput / (2 * self.battery_params.capacity_mwh)
            final_soh = max(0.8, 1.0 - cycles_equivalent * 0.0001)  # 0.01% per cycle

        # Calculate net profit
        net_profit = arbitrage_profit + flexibility_revenue - degradation_cost

        return OptimizationResult(
            algorithm="MILP",
            arbitrage_profit=arbitrage_profit,
            flexibility_revenue=flexibility_revenue,
            degradation_cost=degradation_cost,
            net_profit=net_profit,
            execution_time=execution_time,
            battery_utilization=battery_utilization,
            final_soh=final_soh,
            soc_trajectory=soc_trajectory,
            power_trajectory=power_trajectory,
            flexibility_reservations=flexibility_reservations,
            flexibility_activations=flexibility_activations,
            solver_status=pulp.LpStatus[self.problem.status],
            objective_value=pulp.value(self.problem.objective) if self.problem.objective else 0.0,
            flexibility_revenue_by_service=flex_rev_by_service,
        )

    def get_solver_info(self) -> Dict:
        """
        Get information about the solver and problem statistics

        Returns:
            Dictionary with solver information
        """
        if self.problem is None:
            return {"error": "No problem has been solved yet"}

        return {
            "solver": "CBC",
            "problem_status": pulp.LpStatus[self.problem.status],
            "objective_value": pulp.value(self.problem.objective) if self.problem.objective else None,
            "num_variables": len(self.problem.variables()),
            "num_constraints": len(self.problem.constraints),
            "problem_sense": "Maximize"
        }

    def set_rolling_horizon_config(self, config: RollingHorizonConfig):
        """
        Set configuration for rolling horizon optimization

        Args:
            config: Rolling horizon configuration parameters
        """
        self.rolling_horizon_config = config

    def optimize_rolling_horizon(self, total_time_horizon: int,
                                 energy_prices: List[float],
                                 flexibility_services: List[List[FlexibilityService]],
                                 forecast_errors: Optional[List[float]] = None) -> RollingHorizonResult:
        """
        Perform rolling horizon optimization with receding horizon control

        This method implements a rolling horizon approach where:
        1. Optimize over a fixed horizon length
        2. Execute only the first few steps (step_size)
        3. Update the initial state and move the horizon forward
        4. Handle forecast updates and reoptimization when needed

        Args:
            total_time_horizon: Total time horizon to optimize over
            energy_prices: Energy prices for the entire horizon
            flexibility_services: Flexibility services for the entire horizon
            forecast_errors: Optional forecast error updates at each time step

        Returns:
            RollingHorizonResult with complete optimization results
        """
        import time
        start_time = time.time()

        if self.rolling_horizon_config is None:
            # Use default configuration if none provided
            self.rolling_horizon_config = RollingHorizonConfig()

        config = self.rolling_horizon_config

        # Initialize tracking variables
        optimization_results = []
        forecast_updates = []
        reoptimizations = []
        current_time = 0
        iteration = 0
        previous_objective = None

        # Initialize current SOC
        self.current_soc = self.battery_params.initial_soc

        # Main rolling horizon loop
        while current_time < total_time_horizon and iteration < config.max_iterations:
            iteration += 1

            # Determine the current optimization window
            window_end = min(current_time + config.horizon_length, total_time_horizon)
            window_length = window_end - current_time

            if window_length <= 0:
                break

            # Extract data for current window
            window_energy_prices = energy_prices[current_time:window_end]
            window_flexibility_services = flexibility_services[current_time:window_end]

            # Apply forecast errors if provided
            if forecast_errors is not None:
                window_energy_prices = self._apply_forecast_errors(
                    window_energy_prices, forecast_errors[current_time:window_end]
                )
                forecast_updates.append(current_time)

            # Check if reoptimization is needed due to significant forecast changes
            if self._should_reoptimize(window_energy_prices, current_time, forecast_errors):
                reoptimizations.append(current_time)

            # Update initial SOC for this window
            original_initial_soc = self.battery_params.initial_soc
            self.battery_params.initial_soc = self.current_soc

            try:
                # Optimize current window
                window_result = self.optimize(
                    window_length,
                    window_energy_prices,
                    window_flexibility_services
                )

                # Store the result
                optimization_results.append(window_result)

                # Check convergence
                if previous_objective is not None:
                    objective_change = abs(window_result.objective_value - previous_objective)
                    relative_change = objective_change / max(abs(previous_objective), 1e-6)

                    if relative_change < config.convergence_tolerance:
                        # Converged
                        break

                previous_objective = window_result.objective_value

                # Update current SOC for next iteration
                # Use the SOC at step_size (or end of window if shorter)
                step_to_use = min(config.step_size, len(window_result.soc_trajectory) - 1)
                if step_to_use > 0:
                    self.current_soc = window_result.soc_trajectory[step_to_use]

                # Move to next window
                current_time += min(config.step_size, window_length)

            except Exception as e:
                # Handle optimization failures gracefully
                print(f"Rolling horizon optimization failed at iteration {iteration}: {e}")
                break

            finally:
                # Restore original initial SOC
                self.battery_params.initial_soc = original_initial_soc

        execution_time = time.time() - start_time

        # Determine if converged
        converged = (iteration < config.max_iterations and
                     current_time >= total_time_horizon)

        # Calculate final objective value (sum of all window objectives)
        final_objective_value = sum(result.objective_value for result in optimization_results)

        return RollingHorizonResult(
            total_iterations=iteration,
            converged=converged,
            final_objective_value=final_objective_value,
            execution_time=execution_time,
            optimization_results=optimization_results,
            forecast_updates=forecast_updates,
            reoptimizations=reoptimizations
        )

    def _apply_forecast_errors(self, prices: List[float], errors: List[float]) -> List[float]:
        """
        Apply forecast errors to energy prices

        Args:
            prices: Original energy prices
            errors: Forecast errors (as multiplicative factors, e.g., 1.1 = +10% error)

        Returns:
            Updated prices with forecast errors applied
        """
        if len(errors) != len(prices):
            # If errors list is shorter, pad with 1.0 (no error)
            errors = errors + [1.0] * (len(prices) - len(errors))

        return [price * error for price, error in zip(prices, errors)]

    def _should_reoptimize(self, current_prices: List[float], current_time: int,
                           forecast_errors: Optional[List[float]]) -> bool:
        """
        Determine if reoptimization should be triggered due to forecast changes

        Args:
            current_prices: Current energy prices for the window
            current_time: Current time step
            forecast_errors: Forecast errors (if any)

        Returns:
            True if reoptimization should be triggered
        """
        if forecast_errors is None or self.rolling_horizon_config is None:
            return False

        # Check if any forecast error exceeds the reoptimization threshold
        window_errors = forecast_errors[current_time:current_time + len(current_prices)]

        for error in window_errors:
            # Convert multiplicative error to percentage change
            percentage_change = abs(error - 1.0)
            if percentage_change > self.rolling_horizon_config.reoptimization_threshold:
                return True

        return False

    def get_rolling_horizon_summary(self, rh_result: RollingHorizonResult) -> Dict:
        """
        Generate a summary of rolling horizon optimization results

        Args:
            rh_result: Rolling horizon optimization result

        Returns:
            Dictionary with summary statistics
        """
        if not rh_result.optimization_results:
            return {"error": "No optimization results available"}

        # Aggregate results across all windows
        total_arbitrage_profit = sum(r.arbitrage_profit for r in rh_result.optimization_results)
        total_flexibility_revenue = sum(r.flexibility_revenue for r in rh_result.optimization_results)
        total_degradation_cost = sum(r.degradation_cost for r in rh_result.optimization_results)
        total_net_profit = sum(r.net_profit for r in rh_result.optimization_results)

        avg_battery_utilization = np.mean([r.battery_utilization for r in rh_result.optimization_results])
        avg_execution_time = np.mean([r.execution_time for r in rh_result.optimization_results])

        # Get final SOH from last result
        final_soh = rh_result.optimization_results[-1].final_soh if rh_result.optimization_results else 1.0

        return {
            "total_iterations": rh_result.total_iterations,
            "converged": rh_result.converged,
            "total_execution_time": rh_result.execution_time,
            "avg_window_execution_time": avg_execution_time,
            "total_arbitrage_profit": total_arbitrage_profit,
            "total_flexibility_revenue": total_flexibility_revenue,
            "total_degradation_cost": total_degradation_cost,
            "total_net_profit": total_net_profit,
            "avg_battery_utilization": avg_battery_utilization,
            "final_soh": final_soh,
            "num_forecast_updates": len(rh_result.forecast_updates),
            "num_reoptimizations": len(rh_result.reoptimizations),
            "final_objective_value": rh_result.final_objective_value
        }


# Utility functions for MILP optimization

def create_default_battery_params() -> BatteryParameters:
    """Create default battery parameters for Italian BESS"""
    # Use the same degradation cost as the existing PPO system for consistency
    # From legacy_degradation.py: DEGRADATION_COST_PER_MWH = 25000
    return BatteryParameters(
        capacity_mwh=4.0,
        max_power_mw=2.0,
        efficiency=0.95,
        soc_min=0.1,
        soc_max=0.9,
        initial_soc=0.5,
        degradation_cost_per_mwh=25000.0  # EUR/MWh - matches existing PPO system
    )


def validate_optimization_inputs(time_horizon: int, energy_prices: List[float],
                                 flexibility_services: List[List[FlexibilityService]]) -> bool:
    """
    Validate inputs for MILP optimization

    Args:
        time_horizon: Number of time steps
        energy_prices: Energy prices list
        flexibility_services: Flexibility services list

    Returns:
        True if inputs are valid, False otherwise
    """
    if time_horizon <= 0:
        return False

    if len(energy_prices) != time_horizon:
        return False

    if len(flexibility_services) != time_horizon:
        return False

    # Check that all prices are reasonable (not negative, not extremely high)
    if any(price < 0 or price > 1000 for price in energy_prices):
        return False

    return True