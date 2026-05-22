"""
Performance Metrics Calculation Module
Implements comprehensive performance metrics for PPO vs MILP comparison
according to Requirements 4.4, 4.5, 4.6
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
import time
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from comparison_engine import EpisodeResult, ComparisonSummary


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for algorithm comparison"""
    
    # Financial Performance
    mean_profit: float
    profit_std: float
    profit_median: float
    profit_min: float
    profit_max: float
    profit_percentile_25: float
    profit_percentile_75: float
    
    # Revenue Breakdown
    mean_arbitrage_profit: float
    mean_flexibility_revenue: float
    mean_degradation_cost: float
    
    # Execution Performance
    mean_execution_time: float
    execution_time_std: float
    median_execution_time: float
    max_execution_time: float
    
    # Battery Utilization
    mean_battery_utilization: float
    battery_utilization_std: float
    
    # Battery Health
    mean_final_soh: float
    final_soh_std: float
    
    # Robustness Metrics
    profit_coefficient_of_variation: float  # std/mean
    worst_case_profit: float  # 5th percentile
    best_case_profit: float   # 95th percentile
    
    # Success Rate
    success_rate: float  # Fraction of successful episodes
    
    # Scalability Metrics (execution time vs problem size)
    execution_time_per_hour: float  # ms per hour of optimization horizon
    
    # Risk Metrics
    value_at_risk_5: float    # 5% VaR for profit
    expected_shortfall_5: float  # Expected loss beyond VaR
    
    # Consistency Metrics
    profit_range: float       # max - min profit
    profit_interquartile_range: float  # 75th - 25th percentile


@dataclass
class RobustnessAnalysis:
    """Analysis of algorithm robustness to forecast errors"""
    
    # Performance by error distribution
    performance_by_distribution: Dict[str, PerformanceMetrics]
    
    # Cross-distribution statistics
    mean_profit_across_distributions: float
    profit_std_across_distributions: float
    
    # Robustness indicators
    coefficient_of_variation_across_distributions: float
    worst_distribution_performance: str
    best_distribution_performance: str
    
    # Degradation analysis
    max_performance_degradation: float  # Worst vs best distribution
    relative_performance_degradation: float  # As percentage of best
    
    # Consistency analysis
    distribution_consistency_score: float  # 1 - CV across distributions


@dataclass
class ScalabilityAnalysis:
    """Analysis of algorithm scalability with problem size"""
    
    # Time complexity analysis
    time_complexity_coefficient: float  # Linear regression coefficient
    time_complexity_r_squared: float    # Goodness of fit
    
    # Performance degradation with size
    profit_vs_size_correlation: float
    
    # Scalability score (0-1, higher is better)
    scalability_score: float
    
    # Memory usage (if available)
    memory_usage_coefficient: Optional[float] = None


class PerformanceCalculator:
    """
    Calculator for comprehensive performance metrics
    Implements Requirements 4.4, 4.5, 4.6
    """
    
    def __init__(self):
        """Initialize performance calculator"""
        self.results_cache = {}
    
    def calculate_performance_metrics(self, results: List[EpisodeResult]) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for a set of results
        
        Args:
            results: List of episode results for one algorithm
            
        Returns:
            PerformanceMetrics with all calculated metrics
        """
        if not results:
            return self._empty_metrics()
        
        # Filter successful results
        successful_results = [r for r in results if r.net_profit is not None and 
                            r.solver_status != 'Infeasible']
        
        if not successful_results:
            return self._empty_metrics()
        
        # Extract metrics arrays
        profits = np.array([r.net_profit for r in successful_results])
        arbitrage_profits = np.array([r.arbitrage_profit for r in successful_results])
        flexibility_revenues = np.array([r.flexibility_revenue for r in successful_results])
        degradation_costs = np.array([r.degradation_cost for r in successful_results])
        execution_times = np.array([r.execution_time for r in successful_results])
        battery_utilizations = np.array([r.battery_utilization for r in successful_results])
        final_sohs = np.array([r.final_soh for r in successful_results])
        
        # Calculate financial performance metrics
        profit_stats = self._calculate_distribution_stats(profits)
        
        # Calculate execution performance metrics
        execution_stats = self._calculate_distribution_stats(execution_times)
        
        # Calculate battery metrics
        utilization_stats = self._calculate_distribution_stats(battery_utilizations)
        soh_stats = self._calculate_distribution_stats(final_sohs)
        
        # Calculate robustness metrics
        profit_cv = np.std(profits) / np.mean(profits) if np.mean(profits) != 0 else float('inf')
        worst_case = np.percentile(profits, 5)
        best_case = np.percentile(profits, 95)
        
        # Calculate risk metrics
        var_5 = np.percentile(profits, 5)
        shortfall_mask = profits <= var_5
        expected_shortfall = np.mean(profits[shortfall_mask]) if np.any(shortfall_mask) else var_5
        
        # Calculate scalability metrics (approximate)
        time_horizon_estimate = 24  # Default assumption
        if successful_results and hasattr(successful_results[0], 'soc_trajectory'):
            time_horizon_estimate = len(successful_results[0].soc_trajectory) - 1
        
        execution_time_per_hour = (np.mean(execution_times) * 1000) / time_horizon_estimate  # ms per hour
        
        return PerformanceMetrics(
            # Financial Performance
            mean_profit=profit_stats['mean'],
            profit_std=profit_stats['std'],
            profit_median=profit_stats['median'],
            profit_min=profit_stats['min'],
            profit_max=profit_stats['max'],
            profit_percentile_25=profit_stats['p25'],
            profit_percentile_75=profit_stats['p75'],
            
            # Revenue Breakdown
            mean_arbitrage_profit=np.mean(arbitrage_profits),
            mean_flexibility_revenue=np.mean(flexibility_revenues),
            mean_degradation_cost=np.mean(degradation_costs),
            
            # Execution Performance
            mean_execution_time=execution_stats['mean'],
            execution_time_std=execution_stats['std'],
            median_execution_time=execution_stats['median'],
            max_execution_time=execution_stats['max'],
            
            # Battery Utilization
            mean_battery_utilization=utilization_stats['mean'],
            battery_utilization_std=utilization_stats['std'],
            
            # Battery Health
            mean_final_soh=soh_stats['mean'],
            final_soh_std=soh_stats['std'],
            
            # Robustness Metrics
            profit_coefficient_of_variation=profit_cv,
            worst_case_profit=worst_case,
            best_case_profit=best_case,
            
            # Success Rate
            success_rate=len(successful_results) / len(results),
            
            # Scalability Metrics
            execution_time_per_hour=execution_time_per_hour,
            
            # Risk Metrics
            value_at_risk_5=var_5,
            expected_shortfall_5=expected_shortfall,
            
            # Consistency Metrics
            profit_range=profit_stats['max'] - profit_stats['min'],
            profit_interquartile_range=profit_stats['p75'] - profit_stats['p25']
        )
    
    def analyze_robustness_to_forecast_errors(self, results: List[EpisodeResult]) -> RobustnessAnalysis:
        """
        Analyze algorithm robustness across different forecast error distributions
        
        Args:
            results: List of episode results across all error distributions
            
        Returns:
            RobustnessAnalysis with robustness metrics
        """
        # Group results by error distribution
        results_by_distribution = {}
        for result in results:
            dist = result.error_distribution
            if dist not in results_by_distribution:
                results_by_distribution[dist] = []
            results_by_distribution[dist].append(result)
        
        # Calculate performance metrics for each distribution
        performance_by_distribution = {}
        distribution_profits = {}
        
        for dist, dist_results in results_by_distribution.items():
            metrics = self.calculate_performance_metrics(dist_results)
            performance_by_distribution[dist] = metrics
            distribution_profits[dist] = metrics.mean_profit
        
        # Calculate cross-distribution statistics
        profit_values = list(distribution_profits.values())
        mean_profit_across = np.mean(profit_values)
        std_profit_across = np.std(profit_values)
        cv_across = std_profit_across / mean_profit_across if mean_profit_across != 0 else float('inf')
        
        # Find best and worst performing distributions
        best_dist = max(distribution_profits.keys(), key=lambda k: distribution_profits[k])
        worst_dist = min(distribution_profits.keys(), key=lambda k: distribution_profits[k])
        
        # Calculate performance degradation
        best_profit = distribution_profits[best_dist]
        worst_profit = distribution_profits[worst_dist]
        max_degradation = best_profit - worst_profit
        relative_degradation = (max_degradation / best_profit * 100) if best_profit != 0 else 0.0
        
        # Calculate consistency score (1 - CV, clamped to [0, 1])
        consistency_score = max(0.0, min(1.0, 1.0 - cv_across))
        
        return RobustnessAnalysis(
            performance_by_distribution=performance_by_distribution,
            mean_profit_across_distributions=mean_profit_across,
            profit_std_across_distributions=std_profit_across,
            coefficient_of_variation_across_distributions=cv_across,
            worst_distribution_performance=worst_dist,
            best_distribution_performance=best_dist,
            max_performance_degradation=max_degradation,
            relative_performance_degradation=relative_degradation,
            distribution_consistency_score=consistency_score
        )
    
    def analyze_scalability(self, results: List[EpisodeResult]) -> ScalabilityAnalysis:
        """
        Analyze algorithm scalability with respect to problem size
        
        Args:
            results: List of episode results with varying problem sizes
            
        Returns:
            ScalabilityAnalysis with scalability metrics
        """
        if not results:
            return ScalabilityAnalysis(
                time_complexity_coefficient=0.0,
                time_complexity_r_squared=0.0,
                profit_vs_size_correlation=0.0,
                scalability_score=0.0
            )
        
        # Extract problem sizes (estimated from trajectory lengths)
        problem_sizes = []
        execution_times = []
        profits = []
        
        for result in results:
            if result.soc_trajectory and result.execution_time is not None:
                size = len(result.soc_trajectory) - 1  # Time horizon
                problem_sizes.append(size)
                execution_times.append(result.execution_time)
                profits.append(result.net_profit if result.net_profit is not None else 0.0)
        
        if len(problem_sizes) < 2:
            return ScalabilityAnalysis(
                time_complexity_coefficient=0.0,
                time_complexity_r_squared=0.0,
                profit_vs_size_correlation=0.0,
                scalability_score=0.5  # Neutral score when insufficient data
            )
        
        # Analyze time complexity (linear regression)
        problem_sizes = np.array(problem_sizes)
        execution_times = np.array(execution_times)
        profits = np.array(profits)
        
        # Linear regression: execution_time = a * problem_size + b
        slope, intercept, r_value, p_value, std_err = stats.linregress(problem_sizes, execution_times)
        r_squared = r_value ** 2
        
        # Analyze profit vs size correlation
        profit_size_corr, _ = stats.pearsonr(problem_sizes, profits)
        
        # Calculate scalability score
        # Good scalability: low time complexity coefficient, high R², stable profits
        time_score = 1.0 / (1.0 + slope * 100)  # Lower slope is better
        fit_score = r_squared  # Higher R² means predictable scaling
        profit_score = max(0.0, 1.0 - abs(profit_size_corr))  # Profits should be size-independent
        
        scalability_score = (time_score + fit_score + profit_score) / 3.0
        
        return ScalabilityAnalysis(
            time_complexity_coefficient=slope,
            time_complexity_r_squared=r_squared,
            profit_vs_size_correlation=profit_size_corr,
            scalability_score=scalability_score
        )
    
    def compare_algorithms(self, ppo_results: List[EpisodeResult], 
                          milp_results: List[EpisodeResult]) -> Dict[str, Any]:
        """
        Comprehensive comparison between PPO and MILP algorithms
        
        Args:
            ppo_results: PPO episode results
            milp_results: MILP episode results
            
        Returns:
            Dictionary with comparative analysis
        """
        # Calculate individual metrics
        ppo_metrics = self.calculate_performance_metrics(ppo_results)
        milp_metrics = self.calculate_performance_metrics(milp_results)
        
        # Calculate robustness analysis
        ppo_robustness = self.analyze_robustness_to_forecast_errors(ppo_results)
        milp_robustness = self.analyze_robustness_to_forecast_errors(milp_results)
        
        # Calculate scalability analysis
        ppo_scalability = self.analyze_scalability(ppo_results)
        milp_scalability = self.analyze_scalability(milp_results)
        
        # Comparative metrics
        profit_advantage = milp_metrics.mean_profit - ppo_metrics.mean_profit
        profit_advantage_pct = (profit_advantage / ppo_metrics.mean_profit * 100) if ppo_metrics.mean_profit != 0 else 0.0
        
        execution_time_ratio = (milp_metrics.mean_execution_time / ppo_metrics.mean_execution_time 
                              if ppo_metrics.mean_execution_time > 0 else float('inf'))
        
        # Statistical significance tests
        ppo_profits = [r.net_profit for r in ppo_results if r.net_profit is not None]
        milp_profits = [r.net_profit for r in milp_results if r.net_profit is not None]
        
        statistical_tests = {}
        if len(ppo_profits) > 1 and len(milp_profits) > 1:
            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(milp_profits, ppo_profits, equal_var=False)
            statistical_tests['t_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(milp_profits, ppo_profits, alternative='two-sided')
            statistical_tests['mann_whitney'] = {
                'u_statistic': u_stat,
                'p_value': u_p_value,
                'significant': u_p_value < 0.05
            }
        
        # Risk-adjusted performance comparison
        ppo_sharpe = (ppo_metrics.mean_profit / ppo_metrics.profit_std 
                     if ppo_metrics.profit_std > 0 else 0.0)
        milp_sharpe = (milp_metrics.mean_profit / milp_metrics.profit_std 
                      if milp_metrics.profit_std > 0 else 0.0)
        
        return {
            'individual_metrics': {
                'PPO': asdict(ppo_metrics),
                'MILP': asdict(milp_metrics)
            },
            'robustness_analysis': {
                'PPO': asdict(ppo_robustness),
                'MILP': asdict(milp_robustness)
            },
            'scalability_analysis': {
                'PPO': asdict(ppo_scalability),
                'MILP': asdict(milp_scalability)
            },
            'comparative_metrics': {
                'profit_advantage': profit_advantage,
                'profit_advantage_percentage': profit_advantage_pct,
                'execution_time_ratio': execution_time_ratio,
                'ppo_sharpe_ratio': ppo_sharpe,
                'milp_sharpe_ratio': milp_sharpe,
                'robustness_advantage': (milp_robustness.distribution_consistency_score - 
                                       ppo_robustness.distribution_consistency_score),
                'scalability_advantage': (milp_scalability.scalability_score - 
                                        ppo_scalability.scalability_score)
            },
            'statistical_tests': statistical_tests,
            'summary': {
                'winner_profit': 'MILP' if profit_advantage > 0 else 'PPO',
                'winner_speed': 'PPO' if execution_time_ratio > 1 else 'MILP',
                'winner_robustness': ('MILP' if milp_robustness.distribution_consistency_score > 
                                    ppo_robustness.distribution_consistency_score else 'PPO'),
                'winner_scalability': ('MILP' if milp_scalability.scalability_score > 
                                     ppo_scalability.scalability_score else 'PPO')
            }
        }
    
    def _calculate_distribution_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive distribution statistics"""
        if len(data) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
                'p25': 0.0, 'p75': 0.0
            }
        
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data),
            'p25': np.percentile(data, 25),
            'p75': np.percentile(data, 75)
        }
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics for failed cases"""
        return PerformanceMetrics(
            mean_profit=0.0, profit_std=0.0, profit_median=0.0, profit_min=0.0, profit_max=0.0,
            profit_percentile_25=0.0, profit_percentile_75=0.0,
            mean_arbitrage_profit=0.0, mean_flexibility_revenue=0.0, mean_degradation_cost=0.0,
            mean_execution_time=0.0, execution_time_std=0.0, median_execution_time=0.0, max_execution_time=0.0,
            mean_battery_utilization=0.0, battery_utilization_std=0.0,
            mean_final_soh=1.0, final_soh_std=0.0,
            profit_coefficient_of_variation=0.0, worst_case_profit=0.0, best_case_profit=0.0,
            success_rate=0.0, execution_time_per_hour=0.0,
            value_at_risk_5=0.0, expected_shortfall_5=0.0,
            profit_range=0.0, profit_interquartile_range=0.0
        )


class PerformanceReporter:
    """
    Generate comprehensive performance reports and visualizations
    """
    
    def __init__(self):
        """Initialize performance reporter"""
        self.calculator = PerformanceCalculator()
    
    def generate_performance_report(self, comparison_results: Dict[str, Any], 
                                  output_dir: str = "results") -> str:
        """
        Generate comprehensive performance report
        
        Args:
            comparison_results: Results from PerformanceCalculator.compare_algorithms()
            output_dir: Directory to save report and visualizations
            
        Returns:
            Path to generated report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate report content
        report_content = self._generate_report_content(comparison_results)
        
        # Save report
        report_file = output_path / "performance_comparison_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Generate visualizations
        self._generate_visualizations(comparison_results, output_path)
        
        return str(report_file)
    
    def _generate_report_content(self, results: Dict[str, Any]) -> str:
        """Generate markdown report content"""
        
        ppo_metrics = results['individual_metrics']['PPO']
        milp_metrics = results['individual_metrics']['MILP']
        comparative = results['comparative_metrics']
        summary = results['summary']
        
        report = f"""# PPO vs MILP Performance Comparison Report

## Executive Summary

**Winner by Category:**
- **Profitability**: {summary['winner_profit']} (Advantage: {comparative['profit_advantage']:.2f} EUR)
- **Execution Speed**: {summary['winner_speed']} (Ratio: {comparative['execution_time_ratio']:.2f}x)
- **Robustness**: {summary['winner_robustness']}
- **Scalability**: {summary['winner_scalability']}

## Financial Performance

### PPO Algorithm
- **Mean Profit**: {ppo_metrics['mean_profit']:.2f} EUR
- **Profit Std Dev**: {ppo_metrics['profit_std']:.2f} EUR
- **Profit Range**: {ppo_metrics['profit_min']:.2f} to {ppo_metrics['profit_max']:.2f} EUR
- **Success Rate**: {ppo_metrics['success_rate']:.1%}

### MILP Algorithm  
- **Mean Profit**: {milp_metrics['mean_profit']:.2f} EUR
- **Profit Std Dev**: {milp_metrics['profit_std']:.2f} EUR
- **Profit Range**: {milp_metrics['profit_min']:.2f} to {milp_metrics['profit_max']:.2f} EUR
- **Success Rate**: {milp_metrics['success_rate']:.1%}

### Comparative Analysis
- **MILP Profit Advantage**: {comparative['profit_advantage']:.2f} EUR ({comparative['profit_advantage_percentage']:.1f}%)
- **PPO Sharpe Ratio**: {comparative['ppo_sharpe_ratio']:.3f}
- **MILP Sharpe Ratio**: {comparative['milp_sharpe_ratio']:.3f}

## Execution Performance

### PPO Algorithm
- **Mean Execution Time**: {ppo_metrics['mean_execution_time']:.3f} seconds
- **Execution Time per Hour**: {ppo_metrics['execution_time_per_hour']:.2f} ms/hour

### MILP Algorithm
- **Mean Execution Time**: {milp_metrics['mean_execution_time']:.3f} seconds  
- **Execution Time per Hour**: {milp_metrics['execution_time_per_hour']:.2f} ms/hour

### Speed Comparison
- **MILP/PPO Time Ratio**: {comparative['execution_time_ratio']:.2f}x
- **Speed Winner**: {summary['winner_speed']}

## Battery Utilization and Health

### PPO Algorithm
- **Mean Battery Utilization**: {ppo_metrics['mean_battery_utilization']:.1%}
- **Mean Final SOH**: {ppo_metrics['mean_final_soh']:.3f}

### MILP Algorithm
- **Mean Battery Utilization**: {milp_metrics['mean_battery_utilization']:.1%}
- **Mean Final SOH**: {milp_metrics['mean_final_soh']:.3f}

## Risk Analysis

### PPO Algorithm
- **Coefficient of Variation**: {ppo_metrics['profit_coefficient_of_variation']:.3f}
- **5% Value at Risk**: {ppo_metrics['value_at_risk_5']:.2f} EUR
- **Expected Shortfall**: {ppo_metrics['expected_shortfall_5']:.2f} EUR

### MILP Algorithm
- **Coefficient of Variation**: {milp_metrics['profit_coefficient_of_variation']:.3f}
- **5% Value at Risk**: {milp_metrics['value_at_risk_5']:.2f} EUR
- **Expected Shortfall**: {milp_metrics['expected_shortfall_5']:.2f} EUR

## Statistical Significance

"""
        
        # Add statistical test results if available
        if 'statistical_tests' in results and results['statistical_tests']:
            tests = results['statistical_tests']
            if 't_test' in tests:
                t_test = tests['t_test']
                report += f"""### T-Test Results
- **T-Statistic**: {t_test['t_statistic']:.3f}
- **P-Value**: {t_test['p_value']:.6f}
- **Statistically Significant**: {'Yes' if t_test['significant'] else 'No'}

"""
            
            if 'mann_whitney' in tests:
                mw_test = tests['mann_whitney']
                report += f"""### Mann-Whitney U Test Results
- **U-Statistic**: {mw_test['u_statistic']:.0f}
- **P-Value**: {mw_test['p_value']:.6f}
- **Statistically Significant**: {'Yes' if mw_test['significant'] else 'No'}

"""
        
        report += """## Conclusions and Recommendations

Based on the comprehensive performance analysis, the following conclusions can be drawn:

1. **Profitability**: The algorithm with higher mean profit demonstrates superior financial performance
2. **Risk-Adjusted Performance**: Consider both profit and volatility when making decisions
3. **Execution Speed**: Factor in computational requirements for real-time applications
4. **Robustness**: Evaluate performance consistency across different market conditions

## Methodology

This analysis was conducted using standardized performance metrics including:
- Financial performance indicators (profit, revenue breakdown)
- Execution performance metrics (time, scalability)
- Risk metrics (VaR, expected shortfall, coefficient of variation)
- Statistical significance testing (t-test, Mann-Whitney U test)
- Robustness analysis across forecast error distributions

All comparisons ensure fairness by using identical datasets, market conditions, and error distributions for both algorithms.
"""
        
        return report
    
    def _generate_visualizations(self, results: Dict[str, Any], output_dir: Path):
        """Generate performance visualization charts"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create profit comparison chart
        self._create_profit_comparison_chart(results, output_dir)
        
        # Create execution time comparison chart
        self._create_execution_time_chart(results, output_dir)
        
        # Create risk analysis chart
        self._create_risk_analysis_chart(results, output_dir)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _create_profit_comparison_chart(self, results: Dict[str, Any], output_dir: Path):
        """Create profit comparison visualization"""
        ppo_metrics = results['individual_metrics']['PPO']
        milp_metrics = results['individual_metrics']['MILP']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart of mean profits
        algorithms = ['PPO', 'MILP']
        mean_profits = [ppo_metrics['mean_profit'], milp_metrics['mean_profit']]
        profit_stds = [ppo_metrics['profit_std'], milp_metrics['profit_std']]
        
        bars = ax1.bar(algorithms, mean_profits, yerr=profit_stds, capsize=5, 
                      color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax1.set_ylabel('Mean Profit (EUR)')
        ax1.set_title('Mean Profit Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, profit in zip(bars, mean_profits):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + profit_stds[algorithms.index('PPO' if bar == bars[0] else 'MILP')],
                    f'{profit:.1f}', ha='center', va='bottom')
        
        # Box plot comparison
        profit_ranges = [
            [ppo_metrics['profit_min'], ppo_metrics['profit_percentile_25'], 
             ppo_metrics['profit_median'], ppo_metrics['profit_percentile_75'], ppo_metrics['profit_max']],
            [milp_metrics['profit_min'], milp_metrics['profit_percentile_25'], 
             milp_metrics['profit_median'], milp_metrics['profit_percentile_75'], milp_metrics['profit_max']]
        ]
        
        bp = ax2.boxplot(profit_ranges, labels=algorithms, patch_artist=True)
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][1].set_facecolor('#ff7f0e')
        
        ax2.set_ylabel('Profit Distribution (EUR)')
        ax2.set_title('Profit Distribution Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'profit_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_execution_time_chart(self, results: Dict[str, Any], output_dir: Path):
        """Create execution time comparison visualization"""
        ppo_metrics = results['individual_metrics']['PPO']
        milp_metrics = results['individual_metrics']['MILP']
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        algorithms = ['PPO', 'MILP']
        mean_times = [ppo_metrics['mean_execution_time'], milp_metrics['mean_execution_time']]
        time_stds = [ppo_metrics['execution_time_std'], milp_metrics['execution_time_std']]
        
        bars = ax.bar(algorithms, mean_times, yerr=time_stds, capsize=5,
                     color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        
        ax.set_ylabel('Mean Execution Time (seconds)')
        ax.set_title('Execution Time Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars, mean_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_risk_analysis_chart(self, results: Dict[str, Any], output_dir: Path):
        """Create risk analysis visualization"""
        ppo_metrics = results['individual_metrics']['PPO']
        milp_metrics = results['individual_metrics']['MILP']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coefficient of Variation comparison
        algorithms = ['PPO', 'MILP']
        cvs = [ppo_metrics['profit_coefficient_of_variation'], milp_metrics['profit_coefficient_of_variation']]
        
        bars1 = ax1.bar(algorithms, cvs, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax1.set_ylabel('Coefficient of Variation')
        ax1.set_title('Profit Volatility (Lower is Better)')
        ax1.grid(True, alpha=0.3)
        
        for bar, cv in zip(bars1, cvs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cv:.3f}', ha='center', va='bottom')
        
        # Value at Risk comparison
        vars_5 = [ppo_metrics['value_at_risk_5'], milp_metrics['value_at_risk_5']]
        
        bars2 = ax2.bar(algorithms, vars_5, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax2.set_ylabel('5% Value at Risk (EUR)')
        ax2.set_title('Worst-Case Performance (Higher is Better)')
        ax2.grid(True, alpha=0.3)
        
        for bar, var in zip(bars2, vars_5):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{var:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


# Utility functions for performance analysis

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for performance evaluation
    
    Args:
        returns: Array of returns/profits
        risk_free_rate: Risk-free rate (default 0)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_return = np.mean(returns) - risk_free_rate
    return excess_return / np.std(returns)


def calculate_maximum_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative returns
    
    Args:
        cumulative_returns: Array of cumulative returns
        
    Returns:
        Maximum drawdown (positive value)
    """
    if len(cumulative_returns) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    return np.max(drawdown)


def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate information ratio comparing to benchmark
    
    Args:
        returns: Algorithm returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    if len(returns) != len(benchmark_returns) or len(returns) == 0:
        return 0.0
    
    excess_returns = returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(excess_returns) / tracking_error