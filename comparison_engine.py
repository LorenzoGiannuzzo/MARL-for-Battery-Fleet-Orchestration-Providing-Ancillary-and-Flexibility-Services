"""
Comparison Engine for PPO vs MILP Battery Trading Performance
Implements comprehensive comparison framework with fairness guarantees
and standardized performance metrics according to Italian market requirements
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
import copy
from pathlib import Path
import json

# Import existing components
from milp_optimizer import MILPOptimizer, BatteryParameters, OptimizationResult
from extended_ppo_environment import ExtendedBatteryTradingEnv, BackwardCompatibilityWrapper
from flexibility_market import FlexibilityMarket, FlexibilityService, ServiceType
from drl_flexibility_analysis import ForecastErrorGenerator, Battery
from italian_market_config import ITALIAN_CONFIG


@dataclass
class ComparisonConfig:
    """Configuration for PPO vs MILP comparison"""
    # Time horizon settings
    time_horizon: int = 168  # 1 week (168 hours)
    episode_length: int = 168
    
    # Error distribution settings
    error_distributions: List[str] = None  # Will default to all 4 distributions
    n_episodes_per_distribution: int = 10
    
    # Battery settings (should match both systems)
    battery_capacity_mwh: float = 4.0
    battery_max_power_mw: float = 2.0
    battery_efficiency: float = 0.95
    initial_soc: float = 0.5
    
    # Market settings
    flexibility_enabled: bool = True
    conflict_strategy: str = 'revenue_priority'
    
    # Performance measurement settings
    measure_execution_time: bool = True
    measure_scalability: bool = True
    detailed_logging: bool = True
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        if self.error_distributions is None:
            self.error_distributions = ['uniform', 'normal', 'ornstein-uhlenbeck', 'fixed-bias']


@dataclass
class EpisodeResult:
    """Results from a single comparison episode"""
    episode_id: int
    error_distribution: str
    algorithm: str  # 'PPO' or 'MILP'
    
    # Financial metrics
    arbitrage_profit: float
    flexibility_revenue: float
    degradation_cost: float
    net_profit: float
    
    # Performance metrics
    execution_time: float
    battery_utilization: float
    final_soh: float
    
    # Operational metrics
    soc_trajectory: List[float]
    power_trajectory: List[float]
    flexibility_reservations: Dict[str, List[float]]
    
    # Metadata
    solver_status: Optional[str] = None  # For MILP
    convergence_info: Optional[Dict] = None
    error_sequence: Optional[List[float]] = None


@dataclass
class ComparisonSummary:
    """Summary statistics for comparison results"""
    config: ComparisonConfig
    
    # Overall statistics
    total_episodes: int
    successful_episodes: int
    failed_episodes: int
    
    # Performance by algorithm
    ppo_results: Dict[str, Any]
    milp_results: Dict[str, Any]
    
    # Comparative metrics
    profit_advantage: Dict[str, float]  # By error distribution
    execution_time_ratio: Dict[str, float]  # MILP/PPO ratio by distribution
    robustness_metrics: Dict[str, Any]
    
    # Statistical significance
    statistical_tests: Dict[str, Any]
    
    # Execution metadata
    total_execution_time: float
    timestamp: str


class ComparisonEngine:
    """
    Main comparison engine for PPO vs MILP performance evaluation
    Ensures fairness by using identical datasets, errors, and conditions
    """
    
    def __init__(self, config: ComparisonConfig, ppo_model_path: Optional[str] = None):
        """
        Initialize comparison engine
        
        Args:
            config: Comparison configuration parameters
            ppo_model_path: Optional path to a trained stable-baselines3 PPO model.
                            If provided, the engine will use the trained policy for the
                            PPO episodes. If None, a greedy price-threshold heuristic
                            is used as a fallback baseline (NOT a real PPO policy).
        """
        self.config = config
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        # Initialize components with deterministic behavior
        self.flexibility_market = FlexibilityMarket(random_seed=config.random_seed)
        self.battery_params = self._create_battery_parameters()
        
        # Results storage
        self.episode_results: List[EpisodeResult] = []
        self.comparison_summary: Optional[ComparisonSummary] = None
        
        # Fairness tracking
        self.shared_datasets: Dict[str, Any] = {}
        self.shared_error_sequences: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.execution_times: Dict[str, List[float]] = {'PPO': [], 'MILP': []}
        
        # BUG FIX: optionally load a trained PPO model so that the engine actually
        # benchmarks a real PPO policy and not a hand-crafted greedy heuristic.
        self.ppo_model_path = ppo_model_path
        self.ppo_model = None
        if ppo_model_path is not None:
            try:
                from stable_baselines3 import PPO
                self.ppo_model = PPO.load(ppo_model_path)
                print(f"Loaded trained PPO model from {ppo_model_path}")
            except Exception as e:
                print(f"WARNING: failed to load PPO model from {ppo_model_path}: {e}")
                print("         Falling back to greedy heuristic baseline.")
                self.ppo_model = None
        
    def _create_battery_parameters(self) -> BatteryParameters:
        """Create standardized battery parameters for both algorithms"""
        return BatteryParameters(
            capacity_mwh=self.config.battery_capacity_mwh,
            max_power_mw=self.config.battery_max_power_mw,
            efficiency=self.config.battery_efficiency,
            soc_min=0.1,
            soc_max=0.9,
            initial_soc=self.config.initial_soc,
            degradation_cost_per_mwh=25000.0  # Match existing PPO system
        )
    
    def prepare_shared_datasets(self, prices: List[float]) -> Dict[str, Any]:
        """
        Prepare shared datasets for fair comparison
        Ensures both algorithms use identical market conditions
        
        Args:
            prices: Base energy prices for the comparison period
            
        Returns:
            Dictionary containing shared datasets for all error distributions
        """
        shared_data = {}
        
        for distribution in self.config.error_distributions:
            distribution_data = {
                'base_prices': prices[:self.config.time_horizon],
                'error_sequences': [],
                'flexibility_services': []
            }
            
            # Generate error sequences for all episodes
            error_gen = ForecastErrorGenerator(distribution)
            for episode in range(self.config.n_episodes_per_distribution):
                error_gen.reset()
                
                # Generate error sequence for this episode
                error_sequence = []
                for t in range(self.config.time_horizon):
                    if t < len(prices):
                        _, error = error_gen.apply_error(prices[t])
                        error_sequence.append(error)
                    else:
                        error_sequence.append(0.0)
                
                distribution_data['error_sequences'].append(error_sequence)
            
            # Generate flexibility services for each time step
            flexibility_services = []
            for t in range(self.config.time_horizon):
                hour = t % 24
                day_of_week = (t // 24) % 7
                services = self.flexibility_market.get_flexibility_opportunities(hour, day_of_week)
                flexibility_services.append(services)
            
            distribution_data['flexibility_services'] = flexibility_services
            shared_data[distribution] = distribution_data
        
        self.shared_datasets = shared_data
        return shared_data
    
    def run_ppo_episode(self, distribution: str, episode_id: int, 
                       base_prices: List[float], error_sequence: List[float],
                       flexibility_services: List[List[FlexibilityService]]) -> EpisodeResult:
        """
        Run a single PPO episode with given parameters
        
        Args:
            distribution: Error distribution name
            episode_id: Episode identifier
            base_prices: Base energy prices
            error_sequence: Predetermined error sequence
            flexibility_services: Flexibility services for each time step
            
        Returns:
            EpisodeResult with PPO performance metrics
        """
        start_time = time.time()
        
        # Create error generator with predetermined sequence
        error_gen = ForecastErrorGenerator(distribution)
        error_gen.reset()
        
        # Apply predetermined errors to get forecast prices
        forecast_prices = []
        for t, base_price in enumerate(base_prices):
            if t < len(error_sequence):
                forecast_price = base_price * (1 + error_sequence[t])
                forecast_prices.append(forecast_price)
            else:
                forecast_prices.append(base_price)
        
        # Create PPO environment
        env = ExtendedBatteryTradingEnv(
            prices=forecast_prices,
            error_generator=error_gen,
            flexibility_enabled=self.config.flexibility_enabled,
            conflict_strategy=self.config.conflict_strategy,
            random_seed=self.config.random_seed
        )
        
        # Run episode with simple greedy policy (placeholder for actual PPO agent)
        # In real implementation, this would use a trained PPO agent
        obs, _ = env.reset()
        
        total_arbitrage_profit = 0.0
        total_flexibility_revenue = 0.0
        total_degradation_cost = 0.0
        soc_trajectory = [env.battery.soc]
        power_trajectory = []
        flexibility_reservations = {'FCR': [], 'aFRR': [], 'mFRR': []}
        
        for step in range(min(self.config.episode_length, len(forecast_prices))):
            # BUG FIX: use the trained PPO model if available, otherwise fall back
            # to the greedy heuristic baseline. The previous version always used the
            # heuristic, so the "PPO" results were never actually produced by a PPO.
            if self.ppo_model is not None:
                action, _ = self.ppo_model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = self._select_greedy_action(obs, env)
            
            obs, reward, done, truncated, info = env.step(action)
            
            # Accumulate metrics
            total_arbitrage_profit += info.get('arbitrage_profit', 0.0)
            total_flexibility_revenue += info.get('flexibility_revenue', 0.0)
            total_degradation_cost += info.get('degradation_cost', 0.0)
            
            # Track trajectories
            soc_trajectory.append(info.get('soc', 0.5))
            power_trajectory.append(info.get('arbitrage_energy', 0.0))
            
            # Track flexibility reservations
            flexibility_reservations['FCR'].append(info.get('reserved_fcr', 0.0))
            flexibility_reservations['aFRR'].append(info.get('reserved_afrr', 0.0))
            flexibility_reservations['mFRR'].append(info.get('reserved_mfrr', 0.0))
            
            if done or truncated:
                break
        
        execution_time = time.time() - start_time
        
        # Calculate performance metrics
        net_profit = total_arbitrage_profit + total_flexibility_revenue - total_degradation_cost
        
        # Calculate battery utilization
        total_energy_throughput = sum(abs(p) for p in power_trajectory)
        max_possible_throughput = self.config.battery_max_power_mw * len(power_trajectory)
        battery_utilization = total_energy_throughput / max_possible_throughput if max_possible_throughput > 0 else 0.0
        
        # Calculate final SOH (simplified)
        cycles_equivalent = total_energy_throughput / (2 * self.config.battery_capacity_mwh)
        final_soh = max(0.8, 1.0 - cycles_equivalent * 0.0001)
        
        return EpisodeResult(
            episode_id=episode_id,
            error_distribution=distribution,
            algorithm='PPO',
            arbitrage_profit=total_arbitrage_profit,
            flexibility_revenue=total_flexibility_revenue,
            degradation_cost=total_degradation_cost,
            net_profit=net_profit,
            execution_time=execution_time,
            battery_utilization=battery_utilization,
            final_soh=final_soh,
            soc_trajectory=soc_trajectory,
            power_trajectory=power_trajectory,
            flexibility_reservations=flexibility_reservations,
            error_sequence=error_sequence
        )
    
    def run_milp_episode(self, distribution: str, episode_id: int,
                        base_prices: List[float], error_sequence: List[float],
                        flexibility_services: List[List[FlexibilityService]]) -> EpisodeResult:
        """
        Run a single MILP episode with given parameters
        
        Args:
            distribution: Error distribution name
            episode_id: Episode identifier
            base_prices: Base energy prices
            error_sequence: Predetermined error sequence
            flexibility_services: Flexibility services for each time step
            
        Returns:
            EpisodeResult with MILP performance metrics
        """
        start_time = time.time()
        
        # Apply predetermined errors to get forecast prices
        forecast_prices = []
        for t, base_price in enumerate(base_prices):
            if t < len(error_sequence):
                forecast_price = base_price * (1 + error_sequence[t])
                forecast_prices.append(forecast_price)
            else:
                forecast_prices.append(base_price)
        
        # Create MILP optimizer
        optimizer = MILPOptimizer(self.battery_params, self.flexibility_market)
        
        # Run optimization
        result = optimizer.optimize(
            time_horizon=len(forecast_prices),
            energy_prices=forecast_prices,
            flexibility_services=flexibility_services
        )
        
        execution_time = time.time() - start_time
        
        return EpisodeResult(
            episode_id=episode_id,
            error_distribution=distribution,
            algorithm='MILP',
            arbitrage_profit=result.arbitrage_profit,
            flexibility_revenue=result.flexibility_revenue,
            degradation_cost=result.degradation_cost,
            net_profit=result.net_profit,
            execution_time=execution_time,
            battery_utilization=result.battery_utilization,
            final_soh=result.final_soh,
            soc_trajectory=result.soc_trajectory,
            power_trajectory=result.power_trajectory,
            flexibility_reservations=result.flexibility_reservations,
            solver_status=result.solver_status,
            error_sequence=error_sequence
        )
    
    def _select_greedy_action(self, obs: np.ndarray, env: ExtendedBatteryTradingEnv) -> np.ndarray:
        """
        Greedy price-threshold heuristic used ONLY as a fallback baseline when no
        trained PPO model is supplied. This is NOT a real PPO policy.

        Returns a MultiDiscrete action vector [arb_idx, fcr_idx, afrr_idx, mfrr_idx]
        matching the environment's new action space. The heuristic charges at low
        prices and discharges at high prices, and reserves 50% aFRR by default.

        Args:
            obs: Current observation
            env: Environment instance

        Returns:
            Action vector of shape (4,) with dtype int.
        """
        # Extract current price from observation (index 25 is current_price_norm,
        # normalized by 200.0 EUR/MWh in ExtendedBatteryTradingEnv._get_observation)
        current_price_norm = obs[25] if len(obs) > 25 else 0.5
        current_price = current_price_norm * 200.0

        # Simple strategy: charge when price is low, discharge when high.
        # Arbitrage index 15 = +1.0 MW (charge), index 5 = -1.0 MW (discharge)
        price_threshold = 60.0
        arb_idx = 15 if current_price < price_threshold else 5

        # Default flexibility split: 50% aFRR, no FCR, no mFRR.
        # (FCR cannot be combined with strong arbitrage because of SOC symmetry.)
        return np.array([arb_idx, 0, 1, 0], dtype=int)
    
    def run_comparison(self, prices: List[float]) -> ComparisonSummary:
        """
        Run complete PPO vs MILP comparison
        
        Args:
            prices: Base energy prices for comparison period
            
        Returns:
            ComparisonSummary with complete results and analysis
        """
        comparison_start_time = time.time()
        
        print("=" * 70)
        print("STARTING PPO vs MILP COMPARISON")
        print("=" * 70)
        print(f"Time horizon: {self.config.time_horizon} hours")
        print(f"Error distributions: {self.config.error_distributions}")
        print(f"Episodes per distribution: {self.config.n_episodes_per_distribution}")
        print(f"Flexibility enabled: {self.config.flexibility_enabled}")
        print()
        
        # Prepare shared datasets for fairness
        print("Preparing shared datasets for fair comparison...")
        shared_data = self.prepare_shared_datasets(prices)
        print("✓ Shared datasets prepared")
        print()
        
        # Run comparison episodes
        self.episode_results = []
        
        for distribution in self.config.error_distributions:
            print(f"Running episodes for {distribution} distribution...")
            
            distribution_data = shared_data[distribution]
            base_prices = distribution_data['base_prices']
            flexibility_services = distribution_data['flexibility_services']
            
            for episode_id in range(self.config.n_episodes_per_distribution):
                error_sequence = distribution_data['error_sequences'][episode_id]
                
                # Run PPO episode
                ppo_result = self.run_ppo_episode(
                    distribution, episode_id, base_prices, error_sequence, flexibility_services
                )
                self.episode_results.append(ppo_result)
                
                # Run MILP episode with identical conditions
                milp_result = self.run_milp_episode(
                    distribution, episode_id, base_prices, error_sequence, flexibility_services
                )
                self.episode_results.append(milp_result)
                
                if self.config.detailed_logging:
                    print(f"  Episode {episode_id + 1}/{self.config.n_episodes_per_distribution}: "
                          f"PPO profit={ppo_result.net_profit:.2f}, "
                          f"MILP profit={milp_result.net_profit:.2f}")
            
            print(f"✓ Completed {distribution} distribution")
        
        print()
        print("Generating comparison summary...")
        
        # Generate comprehensive summary
        total_execution_time = time.time() - comparison_start_time
        self.comparison_summary = self._generate_summary(total_execution_time)
        
        print("✓ Comparison complete!")
        print(f"Total execution time: {total_execution_time:.2f} seconds")
        print("=" * 70)
        
        return self.comparison_summary
    
    def _generate_summary(self, total_execution_time: float) -> ComparisonSummary:
        """Generate comprehensive comparison summary with statistical analysis"""
        
        # Separate results by algorithm
        ppo_results = [r for r in self.episode_results if r.algorithm == 'PPO']
        milp_results = [r for r in self.episode_results if r.algorithm == 'MILP']
        
        # Calculate statistics by algorithm
        ppo_stats = self._calculate_algorithm_stats(ppo_results)
        milp_stats = self._calculate_algorithm_stats(milp_results)
        
        # Calculate comparative metrics by distribution
        profit_advantage = {}
        execution_time_ratio = {}
        
        for distribution in self.config.error_distributions:
            ppo_dist = [r for r in ppo_results if r.error_distribution == distribution]
            milp_dist = [r for r in milp_results if r.error_distribution == distribution]
            
            if ppo_dist and milp_dist:
                ppo_avg_profit = np.mean([r.net_profit for r in ppo_dist])
                milp_avg_profit = np.mean([r.net_profit for r in milp_dist])
                
                # Profit advantage: positive means MILP is better
                profit_advantage[distribution] = milp_avg_profit - ppo_avg_profit
                
                # Execution time ratio
                ppo_avg_time = np.mean([r.execution_time for r in ppo_dist])
                milp_avg_time = np.mean([r.execution_time for r in milp_dist])
                execution_time_ratio[distribution] = milp_avg_time / ppo_avg_time if ppo_avg_time > 0 else float('inf')
        
        # Calculate robustness metrics
        robustness_metrics = self._calculate_robustness_metrics(ppo_results, milp_results)
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(ppo_results, milp_results)
        
        # Count successful episodes
        successful_episodes = len([r for r in self.episode_results 
                                 if r.solver_status != 'Infeasible' and r.net_profit is not None])
        failed_episodes = len(self.episode_results) - successful_episodes
        
        return ComparisonSummary(
            config=self.config,
            total_episodes=len(self.episode_results),
            successful_episodes=successful_episodes,
            failed_episodes=failed_episodes,
            ppo_results=ppo_stats,
            milp_results=milp_stats,
            profit_advantage=profit_advantage,
            execution_time_ratio=execution_time_ratio,
            robustness_metrics=robustness_metrics,
            statistical_tests=statistical_tests,
            total_execution_time=total_execution_time,
            timestamp=pd.Timestamp.now().isoformat()
        )
    
    def _calculate_algorithm_stats(self, results: List[EpisodeResult]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for an algorithm's results"""
        if not results:
            return {}
        
        profits = [r.net_profit for r in results if r.net_profit is not None]
        execution_times = [r.execution_time for r in results]
        battery_utilizations = [r.battery_utilization for r in results]
        final_sohs = [r.final_soh for r in results]
        
        return {
            'count': len(results),
            'profit': {
                'mean': np.mean(profits) if profits else 0.0,
                'std': np.std(profits) if profits else 0.0,
                'min': np.min(profits) if profits else 0.0,
                'max': np.max(profits) if profits else 0.0,
                'median': np.median(profits) if profits else 0.0
            },
            'execution_time': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times)
            },
            'battery_utilization': {
                'mean': np.mean(battery_utilizations),
                'std': np.std(battery_utilizations)
            },
            'final_soh': {
                'mean': np.mean(final_sohs),
                'std': np.std(final_sohs)
            }
        }
    
    def _calculate_robustness_metrics(self, ppo_results: List[EpisodeResult], 
                                    milp_results: List[EpisodeResult]) -> Dict[str, Any]:
        """Calculate robustness metrics across error distributions"""
        robustness = {}
        
        for algorithm, results in [('PPO', ppo_results), ('MILP', milp_results)]:
            profits_by_dist = {}
            
            for distribution in self.config.error_distributions:
                dist_results = [r for r in results if r.error_distribution == distribution]
                profits = [r.net_profit for r in dist_results if r.net_profit is not None]
                profits_by_dist[distribution] = profits
            
            # Calculate coefficient of variation across distributions
            dist_means = [np.mean(profits) for profits in profits_by_dist.values() if profits]
            if dist_means:
                robustness[algorithm] = {
                    'mean_across_distributions': np.mean(dist_means),
                    'std_across_distributions': np.std(dist_means),
                    'coefficient_of_variation': np.std(dist_means) / np.mean(dist_means) if np.mean(dist_means) != 0 else float('inf')
                }
        
        return robustness
    
    def _perform_statistical_tests(self, ppo_results: List[EpisodeResult], 
                                  milp_results: List[EpisodeResult]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        from scipy import stats
        
        tests = {}
        
        # Overall profit comparison
        ppo_profits = [r.net_profit for r in ppo_results if r.net_profit is not None]
        milp_profits = [r.net_profit for r in milp_results if r.net_profit is not None]
        
        if len(ppo_profits) > 1 and len(milp_profits) > 1:
            # Welch's t-test (unequal variances)
            t_stat, p_value = stats.ttest_ind(milp_profits, ppo_profits, equal_var=False)
            tests['profit_ttest'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(milp_profits, ppo_profits, alternative='two-sided')
            tests['profit_mannwhitney'] = {
                'u_statistic': u_stat,
                'p_value': u_p_value,
                'significant': u_p_value < 0.05
            }
        
        return tests
    
    def save_results(self, filepath: str):
        """Save comparison results to file"""
        if self.comparison_summary is None:
            raise ValueError("No comparison results to save. Run comparison first.")
        
        # Convert to serializable format
        results_dict = {
            'summary': asdict(self.comparison_summary),
            'episode_results': [asdict(result) for result in self.episode_results]
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load comparison results from file"""
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        # Reconstruct objects (simplified - would need proper deserialization in production)
        print(f"Results loaded from {filepath}")
        return results_dict
    
    def get_episode_results(self) -> List[EpisodeResult]:
        """Get the episode results from the last comparison"""
        return self.episode_results


# Utility functions for comparison analysis

def calculate_performance_metrics(results: List[EpisodeResult]) -> Dict[str, float]:
    """
    Calculate standardized performance metrics for comparison
    
    Args:
        results: List of episode results
        
    Returns:
        Dictionary with performance metrics
    """
    if not results:
        return {}
    
    profits = [r.net_profit for r in results if r.net_profit is not None]
    execution_times = [r.execution_time for r in results]
    utilizations = [r.battery_utilization for r in results]
    
    return {
        'mean_profit': np.mean(profits) if profits else 0.0,
        'profit_volatility': np.std(profits) if profits else 0.0,
        'mean_execution_time': np.mean(execution_times),
        'mean_battery_utilization': np.mean(utilizations),
        'success_rate': len(profits) / len(results) if results else 0.0
    }


def analyze_forecast_error_robustness(results: List[EpisodeResult]) -> Dict[str, Any]:
    """
    Analyze robustness to forecast errors across different distributions
    
    Args:
        results: List of episode results
        
    Returns:
        Dictionary with robustness analysis
    """
    analysis = {}
    
    # Group by error distribution
    by_distribution = {}
    for result in results:
        dist = result.error_distribution
        if dist not in by_distribution:
            by_distribution[dist] = []
        by_distribution[dist].append(result)
    
    # Calculate metrics for each distribution
    for dist, dist_results in by_distribution.items():
        profits = [r.net_profit for r in dist_results if r.net_profit is not None]
        
        if profits:
            analysis[dist] = {
                'mean_profit': np.mean(profits),
                'profit_std': np.std(profits),
                'min_profit': np.min(profits),
                'max_profit': np.max(profits),
                'profit_range': np.max(profits) - np.min(profits),
                'coefficient_of_variation': np.std(profits) / np.mean(profits) if np.mean(profits) != 0 else float('inf')
            }
    
    return analysis