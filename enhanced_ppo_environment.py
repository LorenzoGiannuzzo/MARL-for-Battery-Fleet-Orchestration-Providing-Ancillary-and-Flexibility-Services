"""
Enhanced PPO Environment for Better Performance
Migliora le performance del PPO per competere meglio con MILP
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


class EnhancedBatteryTradingEnv(gym.Env):
    """
    Enhanced PPO Environment with improved reward function and action space
    
    Key improvements:
    1. Better reward shaping to guide PPO toward optimal FCR strategy
    2. Simplified action space focused on high-value services
    3. Enhanced observation space with revenue density information
    4. Improved exploration incentives
    """
    
    def __init__(self, prices, error_generator, flexibility_enabled=True, random_seed=None):
        super(EnhancedBatteryTradingEnv, self).__init__()
        
        # Core components
        self.prices = prices
        self.error_gen = error_generator
        self.battery = Battery()
        self.flexibility_market = FlexibilityMarket(random_seed=random_seed)
        self.flexibility_enabled = flexibility_enabled
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(prices)
        self.forecast_horizon = FORECAST_HORIZON
        self.episode_length = 480
        
        # ENHANCED ACTION SPACE: Focus on FCR + some diversification
        # 15 actions total: 5 arbitrage + 10 FCR-focused flexibility
        self.action_space = spaces.Discrete(15)
        
        # Action mapping:
        # 0-4: Pure arbitrage (-2MW to +2MW, 5 levels)
        # 5-9: FCR-focused (50%, 75%, 90%, 95%, 100% FCR + minimal others)
        # 10-14: Mixed strategies (FCR + aFRR combinations)
        
        self.arbitrage_values = np.linspace(-BATTERY_POWER, BATTERY_POWER, 5)
        
        # ENHANCED OBSERVATION SPACE: Add revenue density information
        # 40 features: 26 original + 14 enhanced flexibility info
        n_features = 1 + FORECAST_HORIZON + 1 + 14  # SOC + forecast + current + enhanced flexibility
        self.observation_space = spaces.Box(
            low=0.0,
            high=3.0,  # Increased range for revenue density
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Price normalization constants
        self.max_capacity_price = 100.0
        self.max_energy_price = 200.0
        
        # Current flexibility reservations
        self.reserved_fcr = 0.0
        self.reserved_afrr = 0.0
        self.reserved_mfrr = 0.0
        
        # Performance tracking for reward shaping
        self.episode_revenue_history = []
        self.best_episode_revenue = 0.0
        
    def _decode_enhanced_action(self, action_idx):
        """Decode enhanced action space focused on high-value strategies"""
        
        if action_idx < 5:
            # Pure arbitrage actions (0-4)
            return {
                'arbitrage': self.arbitrage_values[action_idx],
                'fcr_percentage': 0.0,
                'afrr_percentage': 0.0,
                'mfrr_percentage': 0.0
            }
        elif action_idx < 10:
            # FCR-focused actions (5-9)
            fcr_levels = [0.5, 0.75, 0.9, 0.95, 1.0]  # 50% to 100% FCR
            fcr_level = fcr_levels[action_idx - 5]
            return {
                'arbitrage': 0.0,
                'fcr_percentage': fcr_level,
                'afrr_percentage': 0.0,
                'mfrr_percentage': 0.0
            }
        else:
            # Mixed strategies (10-14): FCR + aFRR combinations
            mixed_strategies = [
                {'fcr': 0.6, 'afrr': 0.4, 'mfrr': 0.0},  # 60% FCR + 40% aFRR
                {'fcr': 0.7, 'afrr': 0.3, 'mfrr': 0.0},  # 70% FCR + 30% aFRR
                {'fcr': 0.8, 'afrr': 0.2, 'mfrr': 0.0},  # 80% FCR + 20% aFRR
                {'fcr': 0.5, 'afrr': 0.3, 'mfrr': 0.2},  # Diversified
                {'fcr': 0.9, 'afrr': 0.1, 'mfrr': 0.0},  # 90% FCR + 10% aFRR
            ]
            strategy = mixed_strategies[action_idx - 10]
            return {
                'arbitrage': 0.0,
                'fcr_percentage': strategy['fcr'],
                'afrr_percentage': strategy['afrr'],
                'mfrr_percentage': strategy['mfrr']
            }
    
    def _get_enhanced_observation(self):
        """Enhanced observation with revenue density information"""
        if self.current_step >= len(self.prices):
            return np.zeros(40, dtype=np.float32)
        
        # Basic features (27 total)
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
                forecast_prices_norm.append(forecast_price / 100.0)
            else:
                forecast_prices_norm.append(0.5)
        
        current_price_norm = current_price / 100.0
        
        # ENHANCED FLEXIBILITY FEATURES (14 features)
        hour = self.current_step % 24
        day_of_week = (self.current_step // 24) % 7
        services = self.flexibility_market.get_flexibility_opportunities(hour, day_of_week)
        service_dict = {s.service_type.value: s for s in services}
        
        # Revenue density calculations (EUR/MW/h expected value)
        fcr_revenue_density = 0.0
        afrr_revenue_density = 0.0
        mfrr_revenue_density = 0.0
        
        if 'FCR' in service_dict:
            fcr_service = service_dict['FCR']
            fcr_revenue_density = (fcr_service.capacity_price + 
                                 fcr_service.energy_price * fcr_service.activation_probability)
        
        if 'aFRR' in service_dict:
            afrr_service = service_dict['aFRR']
            afrr_revenue_density = (afrr_service.capacity_price + 
                                  afrr_service.energy_price * afrr_service.activation_probability)
        
        if 'mFRR' in service_dict:
            mfrr_service = service_dict['mFRR']
            mfrr_revenue_density = (mfrr_service.capacity_price + 
                                  mfrr_service.energy_price * mfrr_service.activation_probability)
        
        # Normalize revenue densities (typical range 0-100 EUR/MW/h)
        fcr_revenue_norm = fcr_revenue_density / 100.0
        afrr_revenue_norm = afrr_revenue_density / 100.0
        mfrr_revenue_norm = mfrr_revenue_density / 100.0
        
        # Revenue ranking (helps PPO understand relative values)
        revenue_densities = [fcr_revenue_density, afrr_revenue_density, mfrr_revenue_density]
        revenue_ranks = np.argsort(np.argsort(revenue_densities)[::-1]) / 2.0  # Normalized ranks
        
        # Market conditions
        is_peak_hour = 1.0 if hour in [7, 8, 9, 18, 19, 20, 21] else 0.0
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        
        # Current reservations (normalized by max power)
        fcr_reservation_norm = self.reserved_fcr / BATTERY_POWER
        afrr_reservation_norm = self.reserved_afrr / BATTERY_POWER
        mfrr_reservation_norm = self.reserved_mfrr / BATTERY_POWER
        
        # Power availability
        total_reserved = self.reserved_fcr + self.reserved_afrr + self.reserved_mfrr
        available_power_norm = max(0, BATTERY_POWER - total_reserved) / BATTERY_POWER
        
        # Enhanced flexibility features (14 total)
        flexibility_features = [
            fcr_revenue_norm,           # FCR revenue density
            afrr_revenue_norm,          # aFRR revenue density  
            mfrr_revenue_norm,          # mFRR revenue density
            revenue_ranks[0],           # FCR rank (0=best, 1=worst)
            revenue_ranks[1],           # aFRR rank
            revenue_ranks[2],           # mFRR rank
            is_peak_hour,               # Peak hour indicator
            is_weekend,                 # Weekend indicator
            fcr_reservation_norm,       # Current FCR reservation
            afrr_reservation_norm,      # Current aFRR reservation
            mfrr_reservation_norm,      # Current mFRR reservation
            available_power_norm,       # Available power for new reservations
            fcr_revenue_density / max(afrr_revenue_density, 1.0),  # FCR vs aFRR ratio
            fcr_revenue_density / max(mfrr_revenue_density, 1.0),  # FCR vs mFRR ratio
        ]
        
        # Combine all features
        obs = np.array([soc] + forecast_prices_norm + [current_price_norm] + flexibility_features, 
                      dtype=np.float32)
        
        return obs
    
    def _calculate_enhanced_reward(self, actions, arbitrage_profit, flexibility_revenue_breakdown, degradation_cost):
        """Enhanced reward function with better shaping for FCR preference"""
        
        # Base reward components
        base_reward = arbitrage_profit + flexibility_revenue_breakdown['total_revenue'] - degradation_cost
        
        # REWARD SHAPING: Guide PPO toward optimal FCR strategy
        
        # 1. FCR Preference Bonus (encourage FCR over other services)
        fcr_bonus = 0.0
        if self.reserved_fcr > 0:
            # Bonus proportional to FCR capacity used
            fcr_utilization = self.reserved_fcr / BATTERY_POWER
            fcr_bonus = fcr_utilization * 5.0  # 5 EUR bonus for full FCR utilization
            
            # Extra bonus for high FCR utilization (>75%)
            if fcr_utilization > 0.75:
                fcr_bonus += (fcr_utilization - 0.75) * 10.0  # Additional bonus
        
        # 2. Efficiency Bonus (reward using full capacity efficiently)
        total_utilization = (abs(actions['arbitrage']) + self.reserved_fcr + 
                           self.reserved_afrr + self.reserved_mfrr) / BATTERY_POWER
        efficiency_bonus = 0.0
        if total_utilization > 0.9:  # High utilization bonus
            efficiency_bonus = (total_utilization - 0.9) * 20.0
        
        # 3. Strategy Consistency Bonus (reward sticking to profitable strategies)
        consistency_bonus = 0.0
        if len(self.episode_revenue_history) > 10:
            recent_avg = np.mean(self.episode_revenue_history[-10:])
            if flexibility_revenue_breakdown['total_revenue'] > recent_avg * 1.1:
                consistency_bonus = 2.0  # Bonus for improving performance
        
        # 4. Diversification Penalty (small penalty for suboptimal diversification)
        diversification_penalty = 0.0
        if self.reserved_afrr > 0 or self.reserved_mfrr > 0:
            # Small penalty if not using FCR when it's clearly better
            hour = self.current_step % 24
            day_of_week = (self.current_step // 24) % 7
            services = self.flexibility_market.get_flexibility_opportunities(hour, day_of_week)
            service_dict = {s.service_type.value: s for s in services}
            
            if 'FCR' in service_dict and 'aFRR' in service_dict:
                fcr_value = (service_dict['FCR'].capacity_price + 
                           service_dict['FCR'].energy_price * service_dict['FCR'].activation_probability)
                afrr_value = (service_dict['aFRR'].capacity_price + 
                            service_dict['aFRR'].energy_price * service_dict['aFRR'].activation_probability)
                
                if fcr_value > afrr_value * 1.1:  # FCR is clearly better
                    non_fcr_utilization = (self.reserved_afrr + self.reserved_mfrr) / BATTERY_POWER
                    diversification_penalty = -non_fcr_utilization * 2.0  # Small penalty
        
        # 5. SOC Management Bonus (reward keeping SOC in good range for flexibility)
        soc_bonus = 0.0
        if 0.4 <= self.battery.soc <= 0.6:  # Optimal SOC range for flexibility
            soc_bonus = 1.0
        
        # Total enhanced reward
        enhanced_reward = (base_reward + fcr_bonus + efficiency_bonus + 
                         consistency_bonus + diversification_penalty + soc_bonus)
        
        return enhanced_reward, {
            'base_reward': base_reward,
            'fcr_bonus': fcr_bonus,
            'efficiency_bonus': efficiency_bonus,
            'consistency_bonus': consistency_bonus,
            'diversification_penalty': diversification_penalty,
            'soc_bonus': soc_bonus,
            'total_enhanced_reward': enhanced_reward
        }
    
    def step(self, action_idx):
        """Enhanced step function with improved reward calculation"""
        
        # Decode enhanced action
        actions = self._decode_enhanced_action(action_idx)
        
        # Get current price and flexibility opportunities
        true_price = self.prices[self.current_step]
        
        # Execute arbitrage action
        arbitrage_energy = self.battery.step(actions['arbitrage'])
        # BUG FIX: sign convention is enforced by Battery.step():
        # - arbitrage_energy > 0 -> charging (energy bought from grid), profit is a cost
        # - arbitrage_energy < 0 -> discharging (energy sold to grid),    profit is a revenue
        # Matches the convention used in BatteryTradingEnvClean (drl_flexibility_analysis.py)
        arbitrage_profit = -arbitrage_energy * true_price
        
        # Update flexibility reservations (no conflicts in enhanced version - simpler)
        available_power = self.battery.max_power
        self.reserved_fcr = actions['fcr_percentage'] * available_power
        self.reserved_afrr = actions['afrr_percentage'] * available_power
        self.reserved_mfrr = actions['mfrr_percentage'] * available_power
        
        # Calculate flexibility revenue
        flexibility_revenue_breakdown = self._calculate_flexibility_revenue()
        
        # Calculate degradation cost
        total_energy_used = abs(arbitrage_energy)
        degradation_cost = total_energy_used * DEGRADATION_COST_PER_MWH / (2 * 6000)
        
        # SOC violation penalty
        soc_penalty = 0.0
        if self.battery.soc > SOC_MAX:
            soc_penalty = -abs(self.battery.soc - SOC_MAX) * SOC_VIOLATION_PENALTY
        elif self.battery.soc < SOC_MIN:
            soc_penalty = -abs(self.battery.soc - SOC_MIN) * SOC_VIOLATION_PENALTY
        
        # Enhanced reward calculation
        enhanced_reward, reward_breakdown = self._calculate_enhanced_reward(
            actions, arbitrage_profit, flexibility_revenue_breakdown, degradation_cost
        )
        
        # Add SOC penalty to final reward
        final_reward = enhanced_reward + soc_penalty
        
        # Track episode revenue for consistency bonus
        self.episode_revenue_history.append(flexibility_revenue_breakdown['total_revenue'])
        if len(self.episode_revenue_history) > 100:  # Keep last 100 steps
            self.episode_revenue_history.pop(0)
        
        # Update step
        self.current_step += 1
        done = self.current_step >= min(self.max_steps, self.episode_length)
        truncated = False
        
        # Get next observation
        if done:
            obs = (self._get_enhanced_observation() if self.current_step < self.max_steps 
                  else np.zeros(40, dtype=np.float32))
        else:
            obs = self._get_enhanced_observation()
        
        # Comprehensive info dictionary
        info = {
            'arbitrage_energy': arbitrage_energy,
            'arbitrage_profit': arbitrage_profit,
            'flexibility_revenue': flexibility_revenue_breakdown['total_revenue'],
            'flexibility_breakdown': flexibility_revenue_breakdown,
            'degradation_cost': degradation_cost,
            'soc_penalty': soc_penalty,
            'reward_breakdown': reward_breakdown,
            'total_reward': final_reward,
            'soc': self.battery.soc,
            'reserved_fcr': self.reserved_fcr,
            'reserved_afrr': self.reserved_afrr,
            'reserved_mfrr': self.reserved_mfrr,
            'price': true_price,
            'actions_executed': actions,
            'power_utilization': (abs(actions['arbitrage']) + self.reserved_fcr + 
                                self.reserved_afrr + self.reserved_mfrr) / available_power
        }
        
        return obs, final_reward, done, truncated, info
    
    def _calculate_flexibility_revenue(self):
        """Calculate flexibility revenue using the existing market model"""
        if not self.flexibility_enabled:
            return {
                'total_revenue': 0.0,
                'fcr_revenue': 0.0,
                'afrr_revenue': 0.0,
                'mfrr_revenue': 0.0,
                'capacity_revenue': 0.0,
                'energy_revenue': 0.0
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
            'energy_revenue': 0.0
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
        
        return revenue_breakdown
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.battery = Battery()
        self.reserved_fcr = 0.0
        self.reserved_afrr = 0.0
        self.reserved_mfrr = 0.0
        self.episode_revenue_history = []
        
        if self.error_gen is not None:
            self.error_gen.reset()
        
        obs = self._get_enhanced_observation()
        info = {}
        
        return obs, info


# Backward compatibility wrapper
class BackwardCompatibilityWrapper(gym.Wrapper):
    """Wrapper to maintain compatibility with existing 26-feature observation space"""
    
    def __init__(self, env):
        super().__init__(env)
        # Override observation space to match original
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(26,),
            dtype=np.float32
        )
        # Keep original action space (21 actions)
        self.action_space = spaces.Discrete(21)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Convert to 26-feature observation
        compatible_obs = self._convert_observation(obs)
        return compatible_obs, info
    
    def step(self, action):
        # Map 21 actions to 15 enhanced actions
        if action < 5:
            # Direct mapping for arbitrage actions
            enhanced_action = action
        elif action < 21:
            # Map remaining actions to FCR-focused strategies
            enhanced_action = 5 + ((action - 5) % 10)
        else:
            enhanced_action = 14  # Default to mixed strategy
        
        obs, reward, done, truncated, info = self.env.step(enhanced_action)
        compatible_obs = self._convert_observation(obs)
        return compatible_obs, reward, done, truncated, info
    
    def _convert_observation(self, enhanced_obs):
        """Convert 40-feature enhanced observation to 26-feature compatible observation"""
        # Take first 26 features (SOC + forecast + current price + basic flexibility info)
        if len(enhanced_obs) >= 26:
            return enhanced_obs[:26].astype(np.float32)
        else:
            # Pad if necessary
            compatible = np.zeros(26, dtype=np.float32)
            compatible[:len(enhanced_obs)] = enhanced_obs
            return compatible