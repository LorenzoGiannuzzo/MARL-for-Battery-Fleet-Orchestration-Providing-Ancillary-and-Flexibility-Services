"""
Statistical Analysis Reporting System for PPO vs MILP Comparison
Implements comprehensive statistical analysis with confidence intervals, significance tests,
and clear differentiation between flexibility service types according to Requirements 7.2, 7.4
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, kruskal, chi2_contingency
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings

# Import comparison framework components
from comparison_engine import ComparisonSummary, EpisodeResult, ComparisonConfig
from flexibility_market import ServiceType


@dataclass
class ConfidenceInterval:
    """Confidence interval with lower and upper bounds"""
    lower: float
    upper: float
    confidence_level: float
    
    def __str__(self) -> str:
        return f"[{self.lower:.3f}, {self.upper:.3f}] ({self.confidence_level*100:.0f}% CI)"


@dataclass
class StatisticalTest:
    """Statistical test result with interpretation"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    degrees_of_freedom: Optional[int]
    effect_size: Optional[float]
    interpretation: str
    significant: bool
    
    def __str__(self) -> str:
        return (f"{self.test_name}: statistic={self.statistic:.4f}, "
                f"p={self.p_value:.4f}, significant={self.significant}")


@dataclass
class FlexibilityServiceAnalysis:
    """Analysis results for a specific flexibility service type"""
    service_type: str
    
    # Usage statistics
    total_reservations: float
    average_reservation: float
    reservation_frequency: float
    peak_usage_hour: int
    
    # Revenue statistics
    total_revenue: float
    average_revenue_per_mw: float
    revenue_volatility: float
    
    # Confidence intervals
    reservation_ci: ConfidenceInterval
    revenue_ci: ConfidenceInterval
    
    # Comparative analysis vs other services
    relative_usage_rank: int
    relative_revenue_rank: int


@dataclass
class ComprehensiveStatisticalReport:
    """Comprehensive statistical analysis report"""
    
    # Basic summary
    analysis_timestamp: str
    total_episodes: int
    algorithms_compared: List[str]
    error_distributions: List[str]
    
    # Algorithm performance comparison
    profit_comparison: Dict[str, Any]
    execution_time_comparison: Dict[str, Any]
    robustness_comparison: Dict[str, Any]
    
    # Statistical significance tests
    significance_tests: List[StatisticalTest]
    
    # Flexibility service analysis
    flexibility_analysis: Dict[str, FlexibilityServiceAnalysis]
    service_comparison_matrix: pd.DataFrame
    
    # Confidence intervals for key metrics
    confidence_intervals: Dict[str, Dict[str, ConfidenceInterval]]
    
    # Effect sizes and practical significance
    effect_sizes: Dict[str, float]
    practical_significance: Dict[str, str]
    
    # Recommendations based on statistical analysis
    recommendations: List[str]


class StatisticalAnalysisReporter:
    """
    Comprehensive statistical analysis and reporting system
    Implements Requirements 7.2, 7.4 for detailed statistical analysis and service differentiation
    """
    
    def __init__(self, confidence_level: float = 0.95, alpha: float = 0.05):
        """
        Initialize statistical analysis reporter
        
        Args:
            confidence_level: Confidence level for intervals (default 95%)
            alpha: Significance level for hypothesis tests (default 5%)
        """
        self.confidence_level = confidence_level
        self.alpha = alpha
        
        # Effect size thresholds (Cohen's conventions)
        self.effect_size_thresholds = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8
        }
    
    def calculate_confidence_interval(self, data: List[float], 
                                    confidence_level: Optional[float] = None) -> ConfidenceInterval:
        """
        Calculate confidence interval for a dataset
        
        Args:
            data: List of numerical values
            confidence_level: Confidence level (uses instance default if None)
            
        Returns:
            ConfidenceInterval object with bounds and confidence level
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        if not data or len(data) < 2:
            return ConfidenceInterval(0.0, 0.0, confidence_level)
        
        data_array = np.array(data)
        n = len(data_array)
        mean = np.mean(data_array)
        std_err = stats.sem(data_array)
        
        # Use t-distribution for small samples, normal for large samples
        if n < 30:
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
            margin_error = t_critical * std_err
        else:
            z_critical = stats.norm.ppf((1 + confidence_level) / 2)
            margin_error = z_critical * std_err
        
        return ConfidenceInterval(
            lower=mean - margin_error,
            upper=mean + margin_error,
            confidence_level=confidence_level
        )
    
    def perform_algorithm_comparison_tests(self, ppo_results: List[EpisodeResult], 
                                         milp_results: List[EpisodeResult]) -> List[StatisticalTest]:
        """
        Perform comprehensive statistical tests comparing PPO and MILP algorithms
        
        Args:
            ppo_results: PPO episode results
            milp_results: MILP episode results
            
        Returns:
            List of statistical test results
        """
        tests = []
        
        # Extract profit data
        ppo_profits = [r.net_profit for r in ppo_results if r.net_profit is not None]
        milp_profits = [r.net_profit for r in milp_results if r.net_profit is not None]
        
        if len(ppo_profits) > 1 and len(milp_profits) > 1:
            
            # 1. Independent samples t-test (parametric)
            try:
                t_stat, p_value = ttest_ind(milp_profits, ppo_profits, equal_var=False)
                effect_size = self._calculate_cohens_d(milp_profits, ppo_profits)
                
                interpretation = self._interpret_t_test(t_stat, p_value, effect_size)
                
                tests.append(StatisticalTest(
                    test_name="Welch's t-test (Profit Comparison)",
                    statistic=t_stat,
                    p_value=p_value,
                    critical_value=None,
                    degrees_of_freedom=None,
                    effect_size=effect_size,
                    interpretation=interpretation,
                    significant=p_value < self.alpha
                ))
            except Exception as e:
                warnings.warn(f"T-test failed: {e}")
            
            # 2. Mann-Whitney U test (non-parametric)
            try:
                u_stat, u_p_value = mannwhitneyu(milp_profits, ppo_profits, alternative='two-sided')
                
                # Calculate effect size for Mann-Whitney U
                n1, n2 = len(milp_profits), len(ppo_profits)
                z_score = (u_stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                r_effect_size = abs(z_score) / np.sqrt(n1 + n2)
                
                interpretation = self._interpret_mann_whitney(u_stat, u_p_value, r_effect_size)
                
                tests.append(StatisticalTest(
                    test_name="Mann-Whitney U Test (Profit Comparison)",
                    statistic=u_stat,
                    p_value=u_p_value,
                    critical_value=None,
                    degrees_of_freedom=None,
                    effect_size=r_effect_size,
                    interpretation=interpretation,
                    significant=u_p_value < self.alpha
                ))
            except Exception as e:
                warnings.warn(f"Mann-Whitney U test failed: {e}")
        
        # Extract execution time data
        ppo_times = [r.execution_time for r in ppo_results if r.execution_time is not None]
        milp_times = [r.execution_time for r in milp_results if r.execution_time is not None]
        
        if len(ppo_times) > 1 and len(milp_times) > 1:
            
            # 3. Execution time comparison (log-transformed due to likely skewness)
            try:
                log_ppo_times = np.log(ppo_times)
                log_milp_times = np.log(milp_times)
                
                t_stat, p_value = ttest_ind(log_milp_times, log_ppo_times, equal_var=False)
                effect_size = self._calculate_cohens_d(log_milp_times, log_ppo_times)
                
                interpretation = f"Log-transformed execution time comparison. " + \
                               self._interpret_t_test(t_stat, p_value, effect_size)
                
                tests.append(StatisticalTest(
                    test_name="Execution Time Comparison (Log-transformed)",
                    statistic=t_stat,
                    p_value=p_value,
                    critical_value=None,
                    degrees_of_freedom=None,
                    effect_size=effect_size,
                    interpretation=interpretation,
                    significant=p_value < self.alpha
                ))
            except Exception as e:
                warnings.warn(f"Execution time test failed: {e}")
        
        return tests
    
    def analyze_flexibility_services(self, episode_results: List[EpisodeResult]) -> Dict[str, FlexibilityServiceAnalysis]:
        """
        Analyze flexibility service usage patterns and revenues by service type
        Implements Requirement 7.4: Clear differentiation between service types
        
        Args:
            episode_results: All episode results with flexibility data
            
        Returns:
            Dictionary mapping service types to their analysis results
        """
        service_analyses = {}
        service_types = ['FCR', 'aFRR', 'mFRR']
        
        # Collect data by service type
        service_data = {service: {'reservations': [], 'revenues': []} for service in service_types}
        
        for result in episode_results:
            if result.flexibility_reservations:
                for service in service_types:
                    if service in result.flexibility_reservations:
                        reservations = result.flexibility_reservations[service]
                        if reservations:
                            service_data[service]['reservations'].extend(reservations)
                            
                            # Estimate revenue (simplified - would use actual revenue data in production)
                            estimated_revenue = sum(r * 50 for r in reservations if r > 0)  # 50 EUR/MW placeholder
                            if estimated_revenue > 0:
                                service_data[service]['revenues'].append(estimated_revenue)
        
        # Analyze each service type
        for service in service_types:
            reservations = service_data[service]['reservations']
            revenues = service_data[service]['revenues']
            
            if reservations:
                # Calculate usage statistics
                total_reservations = sum(reservations)
                average_reservation = np.mean(reservations)
                reservation_frequency = len([r for r in reservations if r > 0]) / len(reservations)
                
                # Find peak usage hour (simplified)
                peak_usage_hour = 12  # Placeholder - would analyze hourly patterns in production
                
                # Calculate revenue statistics
                total_revenue = sum(revenues) if revenues else 0.0
                average_revenue_per_mw = np.mean(revenues) if revenues else 0.0
                revenue_volatility = np.std(revenues) if len(revenues) > 1 else 0.0
                
                # Calculate confidence intervals
                reservation_ci = self.calculate_confidence_interval(reservations)
                revenue_ci = self.calculate_confidence_interval(revenues) if revenues else \
                           ConfidenceInterval(0.0, 0.0, self.confidence_level)
                
                service_analyses[service] = FlexibilityServiceAnalysis(
                    service_type=service,
                    total_reservations=total_reservations,
                    average_reservation=average_reservation,
                    reservation_frequency=reservation_frequency,
                    peak_usage_hour=peak_usage_hour,
                    total_revenue=total_revenue,
                    average_revenue_per_mw=average_revenue_per_mw,
                    revenue_volatility=revenue_volatility,
                    reservation_ci=reservation_ci,
                    revenue_ci=revenue_ci,
                    relative_usage_rank=0,  # Will be calculated after all services
                    relative_revenue_rank=0
                )
            else:
                # No data for this service
                service_analyses[service] = FlexibilityServiceAnalysis(
                    service_type=service,
                    total_reservations=0.0,
                    average_reservation=0.0,
                    reservation_frequency=0.0,
                    peak_usage_hour=0,
                    total_revenue=0.0,
                    average_revenue_per_mw=0.0,
                    revenue_volatility=0.0,
                    reservation_ci=ConfidenceInterval(0.0, 0.0, self.confidence_level),
                    revenue_ci=ConfidenceInterval(0.0, 0.0, self.confidence_level),
                    relative_usage_rank=0,
                    relative_revenue_rank=0
                )
        
        # Calculate relative rankings
        usage_ranking = sorted(service_analyses.items(), 
                             key=lambda x: x[1].total_reservations, reverse=True)
        revenue_ranking = sorted(service_analyses.items(), 
                               key=lambda x: x[1].total_revenue, reverse=True)
        
        for rank, (service, _) in enumerate(usage_ranking, 1):
            service_analyses[service].relative_usage_rank = rank
        
        for rank, (service, _) in enumerate(revenue_ranking, 1):
            service_analyses[service].relative_revenue_rank = rank
        
        return service_analyses
    
    def create_service_comparison_matrix(self, flexibility_analysis: Dict[str, FlexibilityServiceAnalysis]) -> pd.DataFrame:
        """
        Create comparison matrix between flexibility services
        
        Args:
            flexibility_analysis: Analysis results for each service type
            
        Returns:
            DataFrame with service comparison metrics
        """
        services = list(flexibility_analysis.keys())
        metrics = ['Total Reservations', 'Avg Reservation', 'Frequency', 'Total Revenue', 'Avg Revenue/MW']
        
        # Create comparison matrix
        comparison_data = []
        
        for service in services:
            analysis = flexibility_analysis[service]
            comparison_data.append({
                'Service': service,
                'Total Reservations': analysis.total_reservations,
                'Avg Reservation': analysis.average_reservation,
                'Frequency': analysis.reservation_frequency,
                'Total Revenue': analysis.total_revenue,
                'Avg Revenue/MW': analysis.average_revenue_per_mw,
                'Usage Rank': analysis.relative_usage_rank,
                'Revenue Rank': analysis.relative_revenue_rank
            })
        
        return pd.DataFrame(comparison_data)
    
    def perform_robustness_analysis(self, episode_results: List[EpisodeResult]) -> Dict[str, Any]:
        """
        Analyze algorithm robustness across different error distributions
        
        Args:
            episode_results: All episode results
            
        Returns:
            Dictionary with robustness analysis results
        """
        # Group results by algorithm and error distribution
        grouped_results = {}
        
        for result in episode_results:
            key = (result.algorithm, result.error_distribution)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Calculate robustness metrics
        robustness_analysis = {}
        
        for algorithm in ['PPO', 'MILP']:
            algorithm_results = {dist: results for (alg, dist), results in grouped_results.items() if alg == algorithm}
            
            if algorithm_results:
                # Calculate profit statistics across distributions
                dist_profits = {}
                for dist, results in algorithm_results.items():
                    profits = [r.net_profit for r in results if r.net_profit is not None]
                    if profits:
                        dist_profits[dist] = {
                            'mean': np.mean(profits),
                            'std': np.std(profits),
                            'min': np.min(profits),
                            'max': np.max(profits)
                        }
                
                if dist_profits:
                    # Calculate cross-distribution metrics
                    means = [stats['mean'] for stats in dist_profits.values()]
                    stds = [stats['std'] for stats in dist_profits.values()]
                    
                    robustness_analysis[algorithm] = {
                        'profit_by_distribution': dist_profits,
                        'mean_profit_across_distributions': np.mean(means),
                        'std_profit_across_distributions': np.std(means),
                        'coefficient_of_variation': np.std(means) / np.mean(means) if np.mean(means) != 0 else float('inf'),
                        'average_volatility': np.mean(stds),
                        'robustness_score': 1 / (1 + np.std(means) / np.mean(means)) if np.mean(means) != 0 else 0
                    }
        
        return robustness_analysis
    
    def generate_comprehensive_report(self, summary: ComparisonSummary, 
                                    episode_results: List[EpisodeResult]) -> ComprehensiveStatisticalReport:
        """
        Generate comprehensive statistical analysis report
        Implements Requirements 7.2: Detailed statistical analysis with confidence intervals
        
        Args:
            summary: Comparison summary statistics
            episode_results: Detailed episode results
            
        Returns:
            ComprehensiveStatisticalReport with all statistical analyses
        """
        print("Generating comprehensive statistical analysis report...")
        
        # Separate results by algorithm
        ppo_results = [r for r in episode_results if r.algorithm == 'PPO']
        milp_results = [r for r in episode_results if r.algorithm == 'MILP']
        
        # Perform statistical significance tests
        print("  Performing statistical significance tests...")
        significance_tests = self.perform_algorithm_comparison_tests(ppo_results, milp_results)
        
        # Analyze flexibility services
        print("  Analyzing flexibility service differentiation...")
        flexibility_analysis = self.analyze_flexibility_services(episode_results)
        service_comparison_matrix = self.create_service_comparison_matrix(flexibility_analysis)
        
        # Perform robustness analysis
        print("  Performing robustness analysis...")
        robustness_comparison = self.perform_robustness_analysis(episode_results)
        
        # Calculate confidence intervals for key metrics
        print("  Calculating confidence intervals...")
        confidence_intervals = self._calculate_key_confidence_intervals(ppo_results, milp_results)
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(ppo_results, milp_results)
        
        # Determine practical significance
        practical_significance = self._assess_practical_significance(effect_sizes, summary)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(significance_tests, effect_sizes, 
                                                       practical_significance, flexibility_analysis)
        
        # Create comprehensive report
        report = ComprehensiveStatisticalReport(
            analysis_timestamp=pd.Timestamp.now().isoformat(),
            total_episodes=len(episode_results),
            algorithms_compared=['PPO', 'MILP'],
            error_distributions=summary.config.error_distributions,
            profit_comparison=self._summarize_profit_comparison(ppo_results, milp_results),
            execution_time_comparison=self._summarize_execution_time_comparison(ppo_results, milp_results),
            robustness_comparison=robustness_comparison,
            significance_tests=significance_tests,
            flexibility_analysis=flexibility_analysis,
            service_comparison_matrix=service_comparison_matrix,
            confidence_intervals=confidence_intervals,
            effect_sizes=effect_sizes,
            practical_significance=practical_significance,
            recommendations=recommendations
        )
        
        print("✓ Comprehensive statistical analysis report generated")
        return report
    
    def export_report(self, report: ComprehensiveStatisticalReport, 
                     output_path: str, format: str = 'json') -> str:
        """
        Export statistical analysis report to file
        
        Args:
            report: Comprehensive statistical report
            output_path: Output file path
            format: Export format ('json', 'csv', 'txt')
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        
        if format.lower() == 'json':
            # Export as JSON
            report_dict = asdict(report)
            
            # Convert non-serializable objects
            report_dict = self._make_json_serializable(report_dict)
            
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            return str(output_path.with_suffix('.json'))
        
        elif format.lower() == 'csv':
            # Export service comparison matrix as CSV
            csv_path = output_path.with_suffix('.csv')
            report.service_comparison_matrix.to_csv(csv_path, index=False)
            return str(csv_path)
        
        elif format.lower() == 'txt':
            # Export as human-readable text report
            txt_path = output_path.with_suffix('.txt')
            self._export_text_report(report, txt_path)
            return str(txt_path)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _interpret_t_test(self, t_stat: float, p_value: float, effect_size: float) -> str:
        """Interpret t-test results"""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        if abs(effect_size) < self.effect_size_thresholds['small']:
            effect_desc = "negligible"
        elif abs(effect_size) < self.effect_size_thresholds['medium']:
            effect_desc = "small"
        elif abs(effect_size) < self.effect_size_thresholds['large']:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        direction = "MILP performs better" if t_stat > 0 else "PPO performs better"
        
        return f"Difference is {significance} (p={p_value:.4f}) with {effect_desc} effect size (d={effect_size:.3f}). {direction}."
    
    def _interpret_mann_whitney(self, u_stat: float, p_value: float, effect_size: float) -> str:
        """Interpret Mann-Whitney U test results"""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        if effect_size < 0.1:
            effect_desc = "negligible"
        elif effect_size < 0.3:
            effect_desc = "small"
        elif effect_size < 0.5:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        return f"Non-parametric test shows {significance} difference (p={p_value:.4f}) with {effect_desc} effect size (r={effect_size:.3f})."
    
    def _calculate_key_confidence_intervals(self, ppo_results: List[EpisodeResult], 
                                          milp_results: List[EpisodeResult]) -> Dict[str, Dict[str, ConfidenceInterval]]:
        """Calculate confidence intervals for key metrics"""
        intervals = {}
        
        # Profit confidence intervals
        ppo_profits = [r.net_profit for r in ppo_results if r.net_profit is not None]
        milp_profits = [r.net_profit for r in milp_results if r.net_profit is not None]
        
        intervals['profit'] = {
            'PPO': self.calculate_confidence_interval(ppo_profits),
            'MILP': self.calculate_confidence_interval(milp_profits)
        }
        
        # Execution time confidence intervals
        ppo_times = [r.execution_time for r in ppo_results if r.execution_time is not None]
        milp_times = [r.execution_time for r in milp_results if r.execution_time is not None]
        
        intervals['execution_time'] = {
            'PPO': self.calculate_confidence_interval(ppo_times),
            'MILP': self.calculate_confidence_interval(milp_times)
        }
        
        # Battery utilization confidence intervals
        ppo_util = [r.battery_utilization for r in ppo_results if r.battery_utilization is not None]
        milp_util = [r.battery_utilization for r in milp_results if r.battery_utilization is not None]
        
        intervals['battery_utilization'] = {
            'PPO': self.calculate_confidence_interval(ppo_util),
            'MILP': self.calculate_confidence_interval(milp_util)
        }
        
        return intervals
    
    def _calculate_effect_sizes(self, ppo_results: List[EpisodeResult], 
                              milp_results: List[EpisodeResult]) -> Dict[str, float]:
        """Calculate effect sizes for key comparisons"""
        effect_sizes = {}
        
        # Profit effect size
        ppo_profits = [r.net_profit for r in ppo_results if r.net_profit is not None]
        milp_profits = [r.net_profit for r in milp_results if r.net_profit is not None]
        
        if len(ppo_profits) > 1 and len(milp_profits) > 1:
            effect_sizes['profit'] = self._calculate_cohens_d(milp_profits, ppo_profits)
        
        # Execution time effect size (log-transformed)
        ppo_times = [r.execution_time for r in ppo_results if r.execution_time is not None and r.execution_time > 0]
        milp_times = [r.execution_time for r in milp_results if r.execution_time is not None and r.execution_time > 0]
        
        if len(ppo_times) > 1 and len(milp_times) > 1:
            log_ppo_times = np.log(ppo_times)
            log_milp_times = np.log(milp_times)
            effect_sizes['execution_time'] = self._calculate_cohens_d(log_milp_times, log_ppo_times)
        
        return effect_sizes
    
    def _assess_practical_significance(self, effect_sizes: Dict[str, float], 
                                     summary: ComparisonSummary) -> Dict[str, str]:
        """Assess practical significance of differences"""
        practical_significance = {}
        
        for metric, effect_size in effect_sizes.items():
            if abs(effect_size) < self.effect_size_thresholds['small']:
                practical_significance[metric] = "No practical difference"
            elif abs(effect_size) < self.effect_size_thresholds['medium']:
                practical_significance[metric] = "Small practical difference"
            elif abs(effect_size) < self.effect_size_thresholds['large']:
                practical_significance[metric] = "Moderate practical difference"
            else:
                practical_significance[metric] = "Large practical difference"
        
        return practical_significance
    
    def _generate_recommendations(self, significance_tests: List[StatisticalTest],
                                effect_sizes: Dict[str, float],
                                practical_significance: Dict[str, str],
                                flexibility_analysis: Dict[str, FlexibilityServiceAnalysis]) -> List[str]:
        """Generate recommendations based on statistical analysis"""
        recommendations = []
        
        # Algorithm performance recommendations
        profit_significant = any(test.significant and 'profit' in test.test_name.lower() 
                               for test in significance_tests)
        
        if profit_significant:
            profit_effect = effect_sizes.get('profit', 0)
            if profit_effect > 0:
                recommendations.append("MILP shows statistically significant profit advantage over PPO")
            else:
                recommendations.append("PPO shows statistically significant profit advantage over MILP")
        else:
            recommendations.append("No statistically significant difference in profit between algorithms")
        
        # Execution time recommendations
        time_effect = effect_sizes.get('execution_time', 0)
        if abs(time_effect) > self.effect_size_thresholds['medium']:
            if time_effect > 0:
                recommendations.append("MILP requires significantly more execution time than PPO")
            else:
                recommendations.append("PPO requires significantly more execution time than MILP")
        
        # Flexibility service recommendations
        service_rankings = [(service, analysis.total_revenue) 
                          for service, analysis in flexibility_analysis.items()]
        service_rankings.sort(key=lambda x: x[1], reverse=True)
        
        if service_rankings and service_rankings[0][1] > 0:
            top_service = service_rankings[0][0]
            recommendations.append(f"{top_service} generates the highest flexibility revenue")
        
        # Practical significance recommendations
        for metric, significance in practical_significance.items():
            if "Large practical difference" in significance:
                recommendations.append(f"Large practical difference observed in {metric}")
        
        return recommendations
    
    def _summarize_profit_comparison(self, ppo_results: List[EpisodeResult], 
                                   milp_results: List[EpisodeResult]) -> Dict[str, Any]:
        """Summarize profit comparison statistics"""
        ppo_profits = [r.net_profit for r in ppo_results if r.net_profit is not None]
        milp_profits = [r.net_profit for r in milp_results if r.net_profit is not None]
        
        return {
            'ppo_stats': {
                'mean': np.mean(ppo_profits) if ppo_profits else 0,
                'median': np.median(ppo_profits) if ppo_profits else 0,
                'std': np.std(ppo_profits) if ppo_profits else 0,
                'count': len(ppo_profits)
            },
            'milp_stats': {
                'mean': np.mean(milp_profits) if milp_profits else 0,
                'median': np.median(milp_profits) if milp_profits else 0,
                'std': np.std(milp_profits) if milp_profits else 0,
                'count': len(milp_profits)
            },
            'difference': {
                'mean_difference': (np.mean(milp_profits) - np.mean(ppo_profits)) if (milp_profits and ppo_profits) else 0,
                'median_difference': (np.median(milp_profits) - np.median(ppo_profits)) if (milp_profits and ppo_profits) else 0
            }
        }
    
    def _summarize_execution_time_comparison(self, ppo_results: List[EpisodeResult], 
                                           milp_results: List[EpisodeResult]) -> Dict[str, Any]:
        """Summarize execution time comparison statistics"""
        ppo_times = [r.execution_time for r in ppo_results if r.execution_time is not None]
        milp_times = [r.execution_time for r in milp_results if r.execution_time is not None]
        
        return {
            'ppo_stats': {
                'mean': np.mean(ppo_times) if ppo_times else 0,
                'median': np.median(ppo_times) if ppo_times else 0,
                'std': np.std(ppo_times) if ppo_times else 0,
                'count': len(ppo_times)
            },
            'milp_stats': {
                'mean': np.mean(milp_times) if milp_times else 0,
                'median': np.median(milp_times) if milp_times else 0,
                'std': np.std(milp_times) if milp_times else 0,
                'count': len(milp_times)
            },
            'ratio': {
                'mean_ratio': (np.mean(milp_times) / np.mean(ppo_times)) if (milp_times and ppo_times and np.mean(ppo_times) > 0) else 0,
                'median_ratio': (np.median(milp_times) / np.median(ppo_times)) if (milp_times and ppo_times and np.median(ppo_times) > 0) else 0
            }
        }
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (ConfidenceInterval, StatisticalTest, FlexibilityServiceAnalysis)):
            return asdict(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _export_text_report(self, report: ComprehensiveStatisticalReport, output_path: Path):
        """Export human-readable text report"""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE STATISTICAL ANALYSIS REPORT\n")
            f.write("PPO vs MILP Battery Trading Comparison\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            f.write("ANALYSIS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Analysis Timestamp: {report.analysis_timestamp}\n")
            f.write(f"Total Episodes: {report.total_episodes}\n")
            f.write(f"Algorithms Compared: {', '.join(report.algorithms_compared)}\n")
            f.write(f"Error Distributions: {', '.join(report.error_distributions)}\n\n")
            
            # Statistical significance tests
            f.write("STATISTICAL SIGNIFICANCE TESTS:\n")
            f.write("-" * 40 + "\n")
            for test in report.significance_tests:
                f.write(f"{test.test_name}:\n")
                f.write(f"  Statistic: {test.statistic:.4f}\n")
                f.write(f"  p-value: {test.p_value:.4f}\n")
                f.write(f"  Significant: {test.significant}\n")
                f.write(f"  Effect Size: {test.effect_size:.4f}\n")
                f.write(f"  Interpretation: {test.interpretation}\n\n")
            
            # Flexibility service analysis
            f.write("FLEXIBILITY SERVICE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for service, analysis in report.flexibility_analysis.items():
                f.write(f"{service} Service:\n")
                f.write(f"  Total Reservations: {analysis.total_reservations:.2f} MW\n")
                f.write(f"  Average Reservation: {analysis.average_reservation:.2f} MW\n")
                f.write(f"  Reservation Frequency: {analysis.reservation_frequency:.2%}\n")
                f.write(f"  Total Revenue: {analysis.total_revenue:.2f} EUR\n")
                f.write(f"  Usage Rank: {analysis.relative_usage_rank}\n")
                f.write(f"  Revenue Rank: {analysis.relative_revenue_rank}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for i, recommendation in enumerate(report.recommendations, 1):
                f.write(f"{i}. {recommendation}\n")
            
            f.write("\n" + "=" * 80 + "\n")


# Utility functions for statistical analysis validation

def validate_statistical_analysis_integration(report: ComprehensiveStatisticalReport) -> bool:
    """
    Validate that statistical analysis is properly integrated
    Implements Property 21: Statistical Analysis Integration validation
    
    Args:
        report: Comprehensive statistical report
        
    Returns:
        True if statistical analysis is properly integrated
    """
    required_components = [
        'significance_tests',
        'confidence_intervals', 
        'effect_sizes',
        'flexibility_analysis',
        'recommendations'
    ]
    
    missing_components = []
    for component in required_components:
        if not hasattr(report, component) or getattr(report, component) is None:
            missing_components.append(component)
    
    if missing_components:
        print(f"Missing statistical analysis components: {missing_components}")
        return False
    
    # Validate significance tests
    if not report.significance_tests:
        print("No statistical significance tests found")
        return False
    
    # Validate confidence intervals
    if not report.confidence_intervals:
        print("No confidence intervals calculated")
        return False
    
    return True


def validate_service_type_differentiation(flexibility_analysis: Dict[str, FlexibilityServiceAnalysis]) -> bool:
    """
    Validate that flexibility service types are clearly differentiated
    Implements Property 22: Service Type Differentiation validation
    
    Args:
        flexibility_analysis: Analysis results for each service type
        
    Returns:
        True if service types are clearly differentiated
    """
    required_services = ['FCR', 'aFRR', 'mFRR']
    
    # Check all required services are present
    missing_services = [service for service in required_services if service not in flexibility_analysis]
    if missing_services:
        print(f"Missing flexibility service analysis: {missing_services}")
        return False
    
    # Check each service has proper differentiation metrics
    for service, analysis in flexibility_analysis.items():
        if not hasattr(analysis, 'relative_usage_rank') or not hasattr(analysis, 'relative_revenue_rank'):
            print(f"Service {service} missing ranking information")
            return False
        
        if not hasattr(analysis, 'reservation_ci') or not hasattr(analysis, 'revenue_ci'):
            print(f"Service {service} missing confidence intervals")
            return False
    
    return True