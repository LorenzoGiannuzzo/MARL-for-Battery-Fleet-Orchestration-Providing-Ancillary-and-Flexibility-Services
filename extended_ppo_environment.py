"""
Extended PPO Environment for Battery Trading with Flexibility Services
Extends the existing PPO environment to include Italian flexibility services (FCR, aFRR, mFRR)
Maintains backward compatibility with existing 26-feature observation space
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Optional
from dataclasses import dataclass
import copy

# Import existing components
from flexibility_market import FlexibilityMarket, BatteryState, ServiceType
from drl_flexibility_analysis import (
    Battery, ForecastErrorGenerator, FORECAST_HORIZON, BATTERY_POWER, 
    SOC_MIN, SOC_MAX, DEGRADATION_COST_PER_MWH, SOC_VIOLATION_PENALTY
)


@dataclass
class FlexibilityOpportunity:
    """Simplified flexibility opportunity for observation space"""
    service_type: str
    available: bool
    capacity_price_normalized: float  # Normalized to [0, 1]
    energy_price_normalized: float    # Normalized to [0, 1]


class ExtendedBatteryTradingEnv(gym.Env):
    """
    Extended PPO Environment with flexibility services
    
    Observation Space (35 features):
    - SOC (1)
    - Forecast prices for arbitrage (24) 
    - Current arbitrage price (1)
    - FCR opportunity (3: available, capacity_price_norm, energy_price_norm)
    - aFRR opportunity (3: available, capacity_price_norm, energy_price_norm)
    - mFRR opportunity (3: available, capacity_price_norm, energy_price_norm)
    
    Action Space (31 discrete actions):
    - Arbitrage: 21 actions from -2MW to +2MW
    - FCR: 4 actions (0%, 25%, 50%, 75% of available capacity)
    - aFRR: 3 actions (0%, 50%, 100% of available capacity)
    - mFRR: 3 actions (0%, 50%, 100% of available capacity)
    """
    
    def __init__(self, prices, error_generator, flexibility_enabled=True,
                 conflict_strategy='equal_priority', random_seed=None,
                 enable_reward_shaping=True, flex_price_error_pct=0.05):
        super(ExtendedBatteryTradingEnv, self).__init__()

        # Core components
        self.prices = prices
        self.error_gen = error_generator
        self.battery = Battery()
        self.flexibility_market = FlexibilityMarket(
            random_seed=random_seed,
            flex_price_error_pct=flex_price_error_pct,
        )
        self.flexibility_enabled = flexibility_enabled

        # Conflict resolution strategy configuration
        self.conflict_strategy = conflict_strategy  # 'revenue_priority', 'arbitrage_priority', 'equal_priority'
        # Fix 2: price-sensitivity reward shaping (learning aid; ON during training).
        # At evaluation/deployment time the realized economic profit (without the
        # bonus) is what matters; the bonus shapes only the learning gradient.
        self.enable_reward_shaping = enable_reward_shaping

        # Environment state
        self.current_step = 0
        self.max_steps = len(prices)
        self.forecast_horizon = FORECAST_HORIZON
        self.episode_length = 480

        # ====== ACTION SPACE: MultiDiscrete([21, 4, 3, 3]) ======
        # The action is a vector of FOUR independent components, one per
        # category, so the agent can combine arbitrage and all three
        # flexibility services in the same hour. This matches what the MILP
        # is allowed to do.
        #
        #   action[0] : arbitrage power level (0..20 mapped to [-2 MW, +2 MW])
        #   action[1] : FCR  reservation level (0..3 mapped to {0, 25, 50, 75}%)
        #   action[2] : aFRR reservation level (0..2 mapped to {0, 50, 100}%)
        #   action[3] : mFRR reservation level (0..2 mapped to {0, 50, 100}%)
        #
        # The combined power request is reconciled with the battery's nominal
        # power via the existing conflict-resolution machinery (see step()).
        self.n_arbitrage_actions = 21
        self.n_fcr_actions = 4   # 0%, 25%, 50%, 75%
        self.n_afrr_actions = 4  # Fix C: 0%, 50%, 75%, 90% (no 100% saturation)
        self.n_mfrr_actions = 4  # Fix C: 0%, 50%, 75%, 90% (no 100% saturation)

        self.action_space = spaces.MultiDiscrete([
            self.n_arbitrage_actions,
            self.n_fcr_actions,
            self.n_afrr_actions,
            self.n_mfrr_actions,
        ])

        # Total flat-equivalent size (kept for backward-compatible diagnostics)
        self.total_actions = (self.n_arbitrage_actions + self.n_fcr_actions +
                              self.n_afrr_actions + self.n_mfrr_actions)

        # Arbitrage values (signed power levels in MW)
        self.arbitrage_values = np.linspace(-BATTERY_POWER, BATTERY_POWER,
                                            self.n_arbitrage_actions)

        # Flexibility reservation fractions
        # Fix C: aFRR/mFRR upper bound reduced from 100% to 90% so that the
        # agent always leaves at least 10% of the battery's nominal power
        # available for arbitrage. This prevents the degenerate policy in
        # which 100% aFRR is reserved and arbitrage is forcibly zeroed by
        # conflict resolution.
        self.fcr_percentages = [0.0, 0.25, 0.50, 0.75]
        self.afrr_percentages = [0.0, 0.50, 0.75, 0.90]
        self.mfrr_percentages = [0.0, 0.50, 0.75, 0.90]

        # Observation space: 35 features (26 original + 9 flexibility)
        n_features = 1 + FORECAST_HORIZON + 1 + 9  # SOC + forecast + current + flexibility
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(n_features,),
            dtype=np.float32
        )

        # Price normalization constants for flexibility services
        self.max_capacity_price = 100.0  # EUR/MW/h
        self.max_energy_price = 200.0    # EUR/MWh

        # Current flexibility reservations
        self.reserved_fcr = 0.0
        self.reserved_afrr = 0.0
        self.reserved_mfrr = 0.0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.battery.reset()
        if self.error_gen is not None:
            self.error_gen.reset()
        self.current_step = 0

        # Reset flexibility reservations
        self.reserved_fcr = 0.0
        self.reserved_afrr = 0.0
        self.reserved_mfrr = 0.0

        # Random start position for longer datasets
        if self.max_steps > 1000:
            self.current_step = np.random.randint(0, min(1000, self.max_steps - 500))

        obs = self._get_observation()
        return obs, {}

    def _get_flexibility_opportunities(self) -> List[FlexibilityOpportunity]:
        """Get current flexibility opportunities"""
        if not self.flexibility_enabled:
            # Return dummy opportunities when flexibility is disabled
            return [
                FlexibilityOpportunity("FCR", False, 0.0, 0.0),
                FlexibilityOpportunity("aFRR", False, 0.0, 0.0),
                FlexibilityOpportunity("mFRR", False, 0.0, 0.0)
            ]

        # Get current hour and day of week
        hour = self.current_step % 24
        day_of_week = (self.current_step // 24) % 7

        # Get flexibility services from market
        services = self.flexibility_market.get_flexibility_opportunities(hour, day_of_week)

        # Create battery state for constraint checking
        battery_state = BatteryState(
            soc=self.battery.soc,
            available_power=self.battery.max_power,
            reserved_fcr=self.reserved_fcr,
            reserved_afrr=self.reserved_afrr,
            reserved_mfrr=self.reserved_mfrr,
            degradation_cycles=self.battery.equivalent_cycles
        )

        opportunities = []
        for service in services:
            # Check if service can be provided
            can_provide = self.flexibility_market.check_service_constraints(service, battery_state)

            # Normalize prices to [0, 1] range
            capacity_price_norm = min(service.capacity_price / self.max_capacity_price, 1.0)
            energy_price_norm = min(service.energy_price / self.max_energy_price, 1.0)

            opportunity = FlexibilityOpportunity(
                service_type=service.service_type.value,
                available=can_provide,
                capacity_price_normalized=capacity_price_norm,
                energy_price_normalized=energy_price_norm
            )
            opportunities.append(opportunity)

        return opportunities

    def _get_observation(self):
        """Construct extended observation vector with flexibility opportunities"""
        if self.current_step >= len(self.prices):
            return np.zeros(35, dtype=np.float32)

        # Original observation components (26 features)
        soc = self.battery.soc
        current_price = self.prices[self.current_step]

        # Forecast prices (24 features)
        forecast_prices_norm = []
        for h in range(self.forecast_horizon):
            if self.current_step + h < len(self.prices):
                true_future_price = self.prices[self.current_step + h]
                if self.error_gen is not None:
                    forecast_price, _ = self.error_gen.apply_error(true_future_price)
                else:
                    forecast_price = true_future_price
                forecast_prices_norm.append(forecast_price / 200.0)
            else:
                forecast_prices_norm.append(0.5)

        current_price_norm = current_price / 200.0

        # Flexibility opportunities (9 features: 3 services × 3 features each)
        opportunities = self._get_flexibility_opportunities()
        flexibility_features = []

        # Ensure we have exactly 3 opportunities (FCR, aFRR, mFRR)
        service_order = ["FCR", "aFRR", "mFRR"]
        opp_dict = {opp.service_type: opp for opp in opportunities}

        for service_type in service_order:
            if service_type in opp_dict:
                opp = opp_dict[service_type]
                flexibility_features.extend([
                    1.0 if opp.available else 0.0,
                    opp.capacity_price_normalized,
                    opp.energy_price_normalized
                ])
            else:
                # Default values if service not available
                flexibility_features.extend([0.0, 0.0, 0.0])

        # Combine all features (35 total)
        obs = np.array([
            soc,                          # 1 feature
            *forecast_prices_norm,        # 24 features
            current_price_norm,           # 1 feature
            *flexibility_features         # 9 features (3×3)
        ], dtype=np.float32)

        return obs

    def _decode_action(self, action):
        """Decode a MultiDiscrete action vector into per-category levels.

        Expected input: array-like of length 4 containing
            [arbitrage_idx, fcr_idx, afrr_idx, mfrr_idx]

        For backward compatibility with legacy single-action callers, a scalar
        input is also accepted and interpreted as the legacy flat encoding
        (21 arbitrage + 4 FCR + 3 aFRR + 3 mFRR = 31 actions).
        """
        # Backward-compatible scalar dispatch
        if np.ndim(action) == 0:
            action_idx = int(action)
            arbitrage_action = 0.0
            fcr_percentage = afrr_percentage = mfrr_percentage = 0.0
            if action_idx < self.n_arbitrage_actions:
                arbitrage_action = float(self.arbitrage_values[action_idx])
            elif action_idx < self.n_arbitrage_actions + self.n_fcr_actions:
                fcr_percentage = self.fcr_percentages[
                    action_idx - self.n_arbitrage_actions]
            elif action_idx < (self.n_arbitrage_actions + self.n_fcr_actions
                               + self.n_afrr_actions):
                afrr_percentage = self.afrr_percentages[
                    action_idx - self.n_arbitrage_actions - self.n_fcr_actions]
            else:
                mfrr_percentage = self.mfrr_percentages[
                    action_idx - self.n_arbitrage_actions
                    - self.n_fcr_actions - self.n_afrr_actions]
            return {
                'arbitrage': arbitrage_action,
                'fcr_percentage': fcr_percentage,
                'afrr_percentage': afrr_percentage,
                'mfrr_percentage': mfrr_percentage,
            }

        # Standard MultiDiscrete path
        action = np.asarray(action).flatten().astype(int)
        if action.size != 4:
            raise ValueError(
                f"MultiDiscrete action must have 4 components "
                f"(arbitrage, fcr, afrr, mfrr); got shape {action.shape}"
            )
        arb_idx, fcr_idx, afrr_idx, mfrr_idx = action.tolist()
        return {
            'arbitrage': float(self.arbitrage_values[arb_idx]),
            'fcr_percentage':  self.fcr_percentages[fcr_idx],
            'afrr_percentage': self.afrr_percentages[afrr_idx],
            'mfrr_percentage': self.mfrr_percentages[mfrr_idx],
        }

    def _calculate_flexibility_revenue(self, opportunities: List[FlexibilityOpportunity]) -> Dict[str, float]:
        """
        Calculate comprehensive revenue from flexibility services

        Returns:
            Dictionary with detailed revenue breakdown for reward function completeness
        """
        if not self.flexibility_enabled:
            return {
                'total_revenue': 0.0,
                'fcr_revenue': 0.0,
                'afrr_revenue': 0.0,
                'mfrr_revenue': 0.0,
                'capacity_revenue': 0.0,
                'energy_revenue': 0.0,
                'activation_throughput_mwh': 0.0,
            }

        # Get actual services for revenue calculation
        hour = self.current_step % 24
        day_of_week = (self.current_step // 24) % 7
        services = self.flexibility_market.get_flexibility_opportunities(hour, day_of_week)

        service_dict = {s.service_type.value: s for s in services}

        # Initialize revenue breakdown
        revenue_breakdown = {
            'total_revenue': 0.0,
            'fcr_revenue': 0.0,
            'afrr_revenue': 0.0,
            'mfrr_revenue': 0.0,
            'capacity_revenue': 0.0,
            'energy_revenue': 0.0,
            'activation_throughput_mwh': 0.0,  # MWh delivered for flex activation (Fix A)
        }

        # Calculate revenue for each reserved service
        reservations = [
            ('FCR', self.reserved_fcr),
            ('aFRR', self.reserved_afrr),
            ('mFRR', self.reserved_mfrr)
        ]

        for service_type, reserved_capacity in reservations:
            if reserved_capacity > 0 and service_type in service_dict:
                service = service_dict[service_type]

                # Capacity revenue (guaranteed)
                capacity_revenue = self.flexibility_market.calculate_capacity_revenue(
                    service, reserved_capacity
                )

                # Simulate activation and energy revenue
                activation_result = self.flexibility_market.simulate_service_activation(
                    service, reserved_capacity
                )

                service_total = capacity_revenue + activation_result['energy_revenue']

                # Update breakdown
                revenue_breakdown[f'{service_type.lower()}_revenue'] = service_total
                revenue_breakdown['capacity_revenue'] += capacity_revenue
                revenue_breakdown['energy_revenue'] += activation_result['energy_revenue']
                revenue_breakdown['total_revenue'] += service_total
                # BUG FIX (Fix A): track activation throughput so it can be charged
                # against degradation, consistent with the MILP formulation.
                # energy_provided is in MWh delivered by the battery this hour.
                revenue_breakdown['activation_throughput_mwh'] = (
                    revenue_breakdown.get('activation_throughput_mwh', 0.0)
                    + activation_result.get('energy_provided', 0.0)
                )

        return revenue_breakdown

    def _resolve_conflicts(self, actions: Dict, available_power: float) -> Dict:
        """
        Resolve conflicts between arbitrage and flexibility services
        Implements configurable prioritization strategy as per Requirement 1.6
        """
        # Make a deep copy to avoid modifying the original actions
        resolved_actions = copy.deepcopy(actions)

        # Calculate total power demand
        arbitrage_power = abs(resolved_actions['arbitrage'])

        # Calculate flexibility reservations
        fcr_power = resolved_actions['fcr_percentage'] * available_power
        afrr_power = resolved_actions['afrr_percentage'] * available_power
        mfrr_power = resolved_actions['mfrr_percentage'] * available_power

        total_flexibility = fcr_power + afrr_power + mfrr_power
        total_demand = arbitrage_power + total_flexibility

        # If total demand exceeds available power, apply configured prioritization strategy
        if total_demand > available_power:

            if self.conflict_strategy == 'revenue_priority':
                # Strategy 1: Revenue-based prioritization (flexibility services first)
                resolved_actions = self._apply_revenue_priority_strategy(resolved_actions, available_power)

            elif self.conflict_strategy == 'arbitrage_priority':
                # Strategy 2: Arbitrage-first prioritization
                resolved_actions = self._apply_arbitrage_priority_strategy(resolved_actions, available_power)

            elif self.conflict_strategy == 'equal_priority':
                # Strategy 3: Equal scaling of all actions
                resolved_actions = self._apply_equal_priority_strategy(resolved_actions, available_power, total_demand)

            else:
                # Default to revenue priority
                resolved_actions = self._apply_revenue_priority_strategy(resolved_actions, available_power)

        return resolved_actions

    def _apply_revenue_priority_strategy(self, actions: Dict, available_power: float) -> Dict:
        """Apply revenue-based prioritization strategy"""
        # Make a copy to avoid modifying the original
        result_actions = copy.deepcopy(actions)

        arbitrage_power = abs(result_actions['arbitrage'])
        fcr_power = result_actions['fcr_percentage'] * available_power
        afrr_power = result_actions['afrr_percentage'] * available_power
        mfrr_power = result_actions['mfrr_percentage'] * available_power

        # Get current flexibility opportunities for revenue estimation
        hour = self.current_step % 24
        day_of_week = (self.current_step // 24) % 7
        services = self.flexibility_market.get_flexibility_opportunities(hour, day_of_week)
        service_dict = {s.service_type.value: s for s in services}

        # Calculate revenue density (EUR/MW/h) for each service
        service_priorities = []

        if 'FCR' in service_dict and fcr_power > 0:
            fcr_revenue_density = (service_dict['FCR'].capacity_price +
                                 service_dict['FCR'].energy_price * service_dict['FCR'].activation_probability)
            service_priorities.append(('fcr_percentage', fcr_power, fcr_revenue_density))

        if 'aFRR' in service_dict and afrr_power > 0:
            afrr_revenue_density = (service_dict['aFRR'].capacity_price +
                                  service_dict['aFRR'].energy_price * service_dict['aFRR'].activation_probability)
            service_priorities.append(('afrr_percentage', afrr_power, afrr_revenue_density))

        if 'mFRR' in service_dict and mfrr_power > 0:
            mfrr_revenue_density = (service_dict['mFRR'].capacity_price +
                                  service_dict['mFRR'].energy_price * service_dict['mFRR'].activation_probability)
            service_priorities.append(('mfrr_percentage', mfrr_power, mfrr_revenue_density))

        # Sort by revenue density (highest first)
        service_priorities.sort(key=lambda x: x[2], reverse=True)

        # Allocate power based on priority
        remaining_power = available_power

        # First, allocate to flexibility services by priority
        for service_key, requested_power, _ in service_priorities:
            if remaining_power >= requested_power:
                # Full allocation possible
                remaining_power -= requested_power
            else:
                # Partial allocation
                if remaining_power > 0:
                    scale_factor = remaining_power / requested_power
                    result_actions[service_key] *= scale_factor
                    remaining_power = 0
                else:
                    # No power available
                    result_actions[service_key] = 0.0

        # Finally, allocate remaining power to arbitrage
        if remaining_power >= arbitrage_power:
            # Full arbitrage allocation possible
            pass  # Keep original arbitrage action
        elif remaining_power > 0:
            # Partial arbitrage allocation
            scale_factor = remaining_power / arbitrage_power if arbitrage_power > 0 else 0
            result_actions['arbitrage'] *= scale_factor
        else:
            # No power available for arbitrage
            result_actions['arbitrage'] = 0.0

        return result_actions

    def _apply_arbitrage_priority_strategy(self, actions: Dict, available_power: float) -> Dict:
        """Apply arbitrage-first prioritization strategy"""
        # Make a copy to avoid modifying the original
        result_actions = copy.deepcopy(actions)

        arbitrage_power = abs(result_actions['arbitrage'])

        # Allocate to arbitrage first
        remaining_power = available_power - arbitrage_power

        if remaining_power < 0:
            # Not enough power for full arbitrage, scale down
            scale_factor = available_power / arbitrage_power if arbitrage_power > 0 else 0
            result_actions['arbitrage'] *= scale_factor
            remaining_power = 0

        # Allocate remaining power to flexibility services proportionally
        fcr_power = result_actions['fcr_percentage'] * available_power
        afrr_power = result_actions['afrr_percentage'] * available_power
        mfrr_power = result_actions['mfrr_percentage'] * available_power
        total_flexibility_requested = fcr_power + afrr_power + mfrr_power

        if total_flexibility_requested > 0 and remaining_power > 0:
            if remaining_power >= total_flexibility_requested:
                # Full allocation possible for all flexibility services
                pass
            else:
                # Scale down flexibility services proportionally
                scale_factor = remaining_power / total_flexibility_requested
                result_actions['fcr_percentage'] *= scale_factor
                result_actions['afrr_percentage'] *= scale_factor
                result_actions['mfrr_percentage'] *= scale_factor
        elif total_flexibility_requested > 0 and remaining_power <= 0:
            # No power left for flexibility services
            result_actions['fcr_percentage'] = 0.0
            result_actions['afrr_percentage'] = 0.0
            result_actions['mfrr_percentage'] = 0.0

        return result_actions

    def _apply_equal_priority_strategy(self, actions: Dict, available_power: float, total_demand: float) -> Dict:
        """Apply equal scaling prioritization strategy"""
        # Make a copy to avoid modifying the original
        result_actions = copy.deepcopy(actions)

        # Calculate scale factor
        scale_factor = available_power / total_demand if total_demand > 0 else 1.0

        # Scale all actions equally
        result_actions['arbitrage'] *= scale_factor
        result_actions['fcr_percentage'] *= scale_factor
        result_actions['afrr_percentage'] *= scale_factor
        result_actions['mfrr_percentage'] *= scale_factor

        return result_actions

    def set_conflict_strategy(self, strategy: str):
        """
        Set the conflict resolution strategy

        Args:
            strategy: One of 'revenue_priority', 'arbitrage_priority', 'equal_priority'
        """
        valid_strategies = ['revenue_priority', 'arbitrage_priority', 'equal_priority']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")

        self.conflict_strategy = strategy

    def step(self, action):
        """Execute action and return reward.

        `action` is a MultiDiscrete vector of 4 components
        [arbitrage_idx, fcr_idx, afrr_idx, mfrr_idx]. Scalar integers are
        also accepted for backward compatibility with legacy callers (they
        get decoded via the flat 31-action encoding).
        """
        # Decode action
        actions = self._decode_action(action)

        # Get current price and flexibility opportunities
        true_price = self.prices[self.current_step]
        opportunities = self._get_flexibility_opportunities()

        # Resolve conflicts between arbitrage and flexibility
        available_power = self.battery.max_power
        actions = self._resolve_conflicts(actions, available_power)

        # Execute arbitrage action
        arbitrage_energy = self.battery.step(actions['arbitrage'])
        # BUG FIX: sign convention is enforced by Battery.step():
        # - arbitrage_energy > 0 -> charging (energy bought from grid), profit is a cost
        # - arbitrage_energy < 0 -> discharging (energy sold to grid),    profit is a revenue
        # Matches the convention used in BatteryTradingEnvClean (drl_flexibility_analysis.py)
        arbitrage_profit = -arbitrage_energy * true_price

        # Update flexibility reservations
        self.reserved_fcr = actions['fcr_percentage'] * available_power
        self.reserved_afrr = actions['afrr_percentage'] * available_power
        self.reserved_mfrr = actions['mfrr_percentage'] * available_power

        # Calculate comprehensive flexibility revenue
        flexibility_revenue_breakdown = self._calculate_flexibility_revenue(opportunities)
        flexibility_revenue = flexibility_revenue_breakdown['total_revenue']

        # Calculate degradation cost
        # BUG FIX (Fix A): degradation must amortize ALL battery throughput, not just
        # the arbitrage component. When a flexibility service is activated, the
        # battery delivers real energy and physically degrades. We now include the
        # activation throughput so that the PPO sees the same degradation formula
        # as the MILP and cannot 'free-ride' on flexibility services that ostensibly
        # do not move the battery.
        arbitrage_throughput = abs(arbitrage_energy)
        activation_throughput = flexibility_revenue_breakdown.get(
            'activation_throughput_mwh', 0.0)
        total_energy_used = arbitrage_throughput + activation_throughput
        degradation_cost = total_energy_used * DEGRADATION_COST_PER_MWH / (2 * 6000)

        # SOC violation penalty
        soc_penalty = 0.0
        if self.battery.soc > SOC_MAX:
            soc_penalty = -abs(self.battery.soc - SOC_MAX) * SOC_VIOLATION_PENALTY
        elif self.battery.soc < SOC_MIN:
            soc_penalty = -abs(self.battery.soc - SOC_MIN) * SOC_VIOLATION_PENALTY

        # Conflict resolution penalty (if actions were scaled down due to conflicts).
        # Use the pre-conflict-resolution arbitrage value, recovered by decoding
        # the raw action once more (cheap).
        conflict_penalty = 0.0
        pre_actions = self._decode_action(action)
        original_arbitrage = pre_actions['arbitrage']
        if abs(actions['arbitrage'] - original_arbitrage) > 0.01:  # Action was modified
            # Small penalty for not being able to execute desired action
            conflict_penalty = -0.1 * abs(actions['arbitrage'] - original_arbitrage)

        # ====================================================================
        # Fix 2: PRICE-SENSITIVITY INTRINSIC REWARD (learning aid only)
        # --------------------------------------------------------------------
        # The MILP can see the full 24-hour price horizon and arbitrage
        # optimally; the PPO must learn the price signal from the state. With
        # an arbitrage profit of ~5 EUR/MWh delta between peak/off-peak hours
        # and an aFRR capacity payment of ~25 EUR/MW/h dominating the reward,
        # the PPO has no incentive to learn price sensitivity and converges
        # to a price-blind policy.
        #
        # This term injects an explicit signal that rewards "discharge when
        # the price is HIGH relative to the daily mean" and "charge when LOW".
        # It does NOT change the realized economic profit (arbitrage_profit
        # remains the truth-of-record), it only shapes the learning gradient.
        # The coefficient is small enough that it does not distort the
        # asymptotic optimal policy.
        #
        # The shaping is disabled at evaluation time via the
        # `enable_reward_shaping` flag (default True during training).
        # ====================================================================
        price_shaping_bonus = 0.0
        if getattr(self, 'enable_reward_shaping', True) and self.flexibility_enabled:
            # Look at the daily mean price from the forecast window in the obs
            # We compute it on the fly from the raw forecast prices accessible
            # via the environment's internal state.
            forecast_window = self.prices[self.current_step:
                                          self.current_step + 24]
            if len(forecast_window) > 0:
                daily_mean = float(np.mean(forecast_window))
                price_deviation = (true_price - daily_mean) / max(daily_mean, 1.0)
                # arbitrage_energy > 0 = charging (we want this when price < mean)
                # arbitrage_energy < 0 = discharging (we want this when price > mean)
                # So the dot product (-energy) * deviation is positive when the
                # agent is doing the right thing.
                # Coefficient PRICE_SHAPING_K is small relative to typical
                # arbitrage_profit so as not to override the real economic signal.
                PRICE_SHAPING_K = 5.0
                price_shaping_bonus = (
                    PRICE_SHAPING_K * (-arbitrage_energy) * price_deviation
                )

        # Total reward: profit + flexibility_revenue - degradation + penalties + shaping
        reward = (arbitrage_profit + flexibility_revenue - degradation_cost +
                 soc_penalty + conflict_penalty + price_shaping_bonus)

        # Update step
        self.current_step += 1
        done = self.current_step >= min(self.max_steps, self.episode_length)
        truncated = False

        # Get next observation
        if done:
            obs = (self._get_observation() if self.current_step < self.max_steps
                  else np.zeros(35, dtype=np.float32))
        else:
            obs = self._get_observation()

        # Enhanced info dictionary with complete reward breakdown
        info = {
            'arbitrage_energy': arbitrage_energy,
            'arbitrage_profit': arbitrage_profit,
            'flexibility_revenue': flexibility_revenue,
            'flexibility_breakdown': flexibility_revenue_breakdown,
            'degradation_cost': degradation_cost,
            'arbitrage_throughput_mwh': arbitrage_throughput,
            'activation_throughput_mwh': activation_throughput,
            'total_throughput_mwh': total_energy_used,
            'soc_penalty': soc_penalty,
            'conflict_penalty': conflict_penalty,
            'total_reward': reward,
            'soc': self.battery.soc,
            'reserved_fcr': self.reserved_fcr,
            'reserved_afrr': self.reserved_afrr,
            'reserved_mfrr': self.reserved_mfrr,
            'price': true_price,
            'actions_executed': actions,
            'action_raw': np.asarray(action).flatten().astype(int).tolist()
                          if np.ndim(action) > 0 else [int(action)],
            'power_utilization': (abs(actions['arbitrage']) + self.reserved_fcr + 
                                self.reserved_afrr + self.reserved_mfrr) / available_power
        }
        
        return obs, reward, done, truncated, info
    
    def get_backward_compatible_observation(self):
        """Get 26-feature observation for backward compatibility testing"""
        if self.current_step >= len(self.prices):
            return np.zeros(26, dtype=np.float32)
        
        soc = self.battery.soc
        current_price = self.prices[self.current_step]
        
        forecast_prices_norm = []
        for h in range(self.forecast_horizon):
            if self.current_step + h < len(self.prices):
                true_future_price = self.prices[self.current_step + h]
                if self.error_gen is not None:
                    forecast_price, _ = self.error_gen.apply_error(true_future_price)
                else:
                    forecast_price = true_future_price
                forecast_prices_norm.append(forecast_price / 200.0)
            else:
                forecast_prices_norm.append(0.5)
        
        current_price_norm = current_price / 200.0
        
        obs = np.array([
            soc,
            *forecast_prices_norm,
            current_price_norm
        ], dtype=np.float32)
        
        return obs


class BackwardCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper to maintain backward compatibility with existing PPO models
    Converts 35-feature observations to 26-feature observations
    """
    
    def __init__(self, env):
        super(BackwardCompatibilityWrapper, self).__init__(env)
        
        # Override observation space to match original
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(26,),
            dtype=np.float32
        )
        
        # Override action space to match original (21 actions)
        self.action_space = spaces.Discrete(21)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Extract first 26 features (original observation space)
        compatible_obs = obs[:26]
        return compatible_obs, info
    
    def step(self, action):
        # Ensure action is within original action space
        action = min(action, 20)  # Clamp to original action space
        
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Extract first 26 features
        compatible_obs = obs[:26]
        
        # Remove flexibility-specific info for compatibility
        compatible_info = {
            'energy': info.get('arbitrage_energy', 0.0),
            'price': info.get('price', 0.0),
            'profit': info.get('arbitrage_profit', 0.0),
            'degradation_cost': info.get('degradation_cost', 0.0),
            'soc': info.get('soc', 0.5)
        }
        
        return compatible_obs, reward, done, truncated, compatible_info