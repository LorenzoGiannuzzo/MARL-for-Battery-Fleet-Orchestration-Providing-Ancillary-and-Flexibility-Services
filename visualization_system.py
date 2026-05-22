"""
Comprehensive Visualization System for PPO vs MILP Battery Trading Comparison
Extends existing plotting system with comparative visualizations, heatmaps, and statistical analysis
Supports both interactive dashboards and high-resolution PDF exports
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
from dataclasses import asdict
import json

# Import comparison framework components
from comparison_engine import ComparisonSummary, EpisodeResult, ComparisonConfig
from flexibility_market import ServiceType


class VisualizationSystem:
    """
    Comprehensive visualization system for PPO vs MILP comparison results
    Implements Requirements 7.1, 7.3, 7.5, 7.6 for comparative visualizations
    """
    
    def __init__(self, output_dir: str = "results", style: str = "seaborn-v0_8"):
        """
        Initialize visualization system
        
        Args:
            output_dir: Directory for saving visualizations
            style: Matplotlib style for consistent appearance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set consistent style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Color scheme for algorithms
        self.algorithm_colors = {
            'PPO': '#FF6B6B',  # Coral red
            'MILP': '#4ECDC4'  # Teal
        }
        
        # Color scheme for flexibility services
        self.service_colors = {
            'FCR': '#FFD93D',    # Yellow
            'aFRR': '#6BCF7F',   # Green  
            'mFRR': '#4D96FF'    # Blue
        }
        
        # Error distribution colors
        self.distribution_colors = {
            'uniform': '#E74C3C',
            'normal': '#3498DB', 
            'ornstein-uhlenbeck': '#9B59B6',
            'fixed-bias': '#F39C12'
        }
        
        # Figure settings for high-quality output
        self.fig_params = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
    
    def create_profit_comparison_plots(self, summary: ComparisonSummary, 
                                     episode_results: List[EpisodeResult]) -> Dict[str, str]:
        """
        Create comprehensive profit comparison visualizations
        
        Args:
            summary: Comparison summary statistics
            episode_results: Detailed episode results
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        saved_files = {}
        
        # 1. Overall profit comparison boxplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PPO vs MILP Profit Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        ppo_results = [r for r in episode_results if r.algorithm == 'PPO']
        milp_results = [r for r in episode_results if r.algorithm == 'MILP']
        
        # Overall profit distribution
        ax = axes[0, 0]
        profit_data = []
        algorithms = []
        
        for result in ppo_results:
            if result.net_profit is not None:
                profit_data.append(result.net_profit)
                algorithms.append('PPO')
        
        for result in milp_results:
            if result.net_profit is not None:
                profit_data.append(result.net_profit)
                algorithms.append('MILP')
        
        df_profits = pd.DataFrame({'Profit': profit_data, 'Algorithm': algorithms})
        sns.boxplot(data=df_profits, x='Algorithm', y='Profit', hue='Algorithm', ax=ax, 
                   palette=[self.algorithm_colors['PPO'], self.algorithm_colors['MILP']], legend=False)
        ax.set_title('Overall Profit Distribution', fontweight='bold')
        ax.set_ylabel('Net Profit (EUR)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Profit by error distribution
        ax = axes[0, 1]
        profit_by_dist = []
        
        for dist in summary.config.error_distributions:
            for result in episode_results:
                if result.error_distribution == dist and result.net_profit is not None:
                    profit_by_dist.append({
                        'Distribution': dist,
                        'Algorithm': result.algorithm,
                        'Profit': result.net_profit
                    })
        
        df_dist = pd.DataFrame(profit_by_dist)
        if not df_dist.empty:
            sns.boxplot(data=df_dist, x='Distribution', y='Profit', hue='Algorithm', ax=ax,
                       palette=[self.algorithm_colors['PPO'], self.algorithm_colors['MILP']])
            ax.set_title('Profit by Error Distribution', fontweight='bold')
            ax.set_ylabel('Net Profit (EUR)', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Cumulative profit advantage
        ax = axes[1, 0]
        distributions = list(summary.profit_advantage.keys())
        advantages = list(summary.profit_advantage.values())
        
        colors = [self.algorithm_colors['MILP'] if adv > 0 else self.algorithm_colors['PPO'] 
                 for adv in advantages]
        
        bars = ax.bar(distributions, advantages, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title('MILP Profit Advantage by Distribution', fontweight='bold')
        ax.set_ylabel('Profit Advantage (EUR)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, advantages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(abs(min(advantages)), max(advantages))),
                   f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Execution time comparison
        ax = axes[1, 1]
        time_ratios = list(summary.execution_time_ratio.values())
        
        bars = ax.bar(distributions, time_ratios, color=self.algorithm_colors['MILP'], alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Equal Time')
        ax.set_title('MILP/PPO Execution Time Ratio', fontweight='bold')
        ax.set_ylabel('Time Ratio (MILP/PPO)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add value labels
        for bar, value in zip(bars, time_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        profit_path = self.output_dir / "profit_comparison_analysis.png"
        plt.savefig(profit_path, **self.fig_params)
        plt.savefig(profit_path.with_suffix('.pdf'), **self.fig_params)
        plt.close()
        
        saved_files['profit_comparison'] = str(profit_path)
        
        return saved_files
    
    def create_soc_utilization_plots(self, episode_results: List[EpisodeResult]) -> Dict[str, str]:
        """
        Create SOC and battery utilization comparison plots
        
        Args:
            episode_results: Detailed episode results
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        saved_files = {}
        
        # Create comprehensive SOC analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Battery SOC and Utilization Analysis', fontsize=16, fontweight='bold')
        
        # Sample a few representative episodes for trajectory plotting
        sample_episodes = {}
        for algorithm in ['PPO', 'MILP']:
            alg_results = [r for r in episode_results if r.algorithm == algorithm]
            if alg_results:
                # Take first episode from each distribution
                for dist in ['uniform', 'normal']:
                    dist_results = [r for r in alg_results if r.error_distribution == dist]
                    if dist_results:
                        sample_episodes[f"{algorithm}_{dist}"] = dist_results[0]
        
        # SOC trajectories comparison
        ax = axes[0, 0]
        for key, result in sample_episodes.items():
            algorithm, dist = key.split('_')
            if result.soc_trajectory:
                hours = range(len(result.soc_trajectory))
                soc_percent = [s * 100 for s in result.soc_trajectory]
                
                color = self.algorithm_colors[algorithm]
                linestyle = '-' if dist == 'uniform' else '--'
                
                ax.plot(hours, soc_percent, color=color, linestyle=linestyle, 
                       linewidth=2, alpha=0.8, label=f'{algorithm} ({dist})')
        
        ax.set_title('Sample SOC Trajectories', fontweight='bold')
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel('SOC (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Battery utilization distribution
        ax = axes[0, 1]
        utilization_data = []
        algorithms = []
        
        for result in episode_results:
            if result.battery_utilization is not None:
                utilization_data.append(result.battery_utilization * 100)
                algorithms.append(result.algorithm)
        
        df_util = pd.DataFrame({'Utilization': utilization_data, 'Algorithm': algorithms})
        sns.boxplot(data=df_util, x='Algorithm', y='Utilization', hue='Algorithm', ax=ax,
                   palette=[self.algorithm_colors['PPO'], self.algorithm_colors['MILP']], legend=False)
        ax.set_title('Battery Utilization Distribution', fontweight='bold')
        ax.set_ylabel('Utilization (%)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Power trajectory comparison
        ax = axes[1, 0]
        for key, result in sample_episodes.items():
            algorithm, dist = key.split('_')
            if result.power_trajectory:
                hours = range(len(result.power_trajectory))
                
                color = self.algorithm_colors[algorithm]
                linestyle = '-' if dist == 'uniform' else '--'
                
                ax.plot(hours, result.power_trajectory, color=color, linestyle=linestyle,
                       linewidth=1.5, alpha=0.8, label=f'{algorithm} ({dist})')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title('Sample Power Trajectories', fontweight='bold')
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel('Power (MW)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Final SOH comparison
        ax = axes[1, 1]
        soh_data = []
        algorithms = []
        
        for result in episode_results:
            if result.final_soh is not None:
                soh_data.append(result.final_soh * 100)
                algorithms.append(result.algorithm)
        
        df_soh = pd.DataFrame({'SOH': soh_data, 'Algorithm': algorithms})
        sns.boxplot(data=df_soh, x='Algorithm', y='SOH', hue='Algorithm', ax=ax,
                   palette=[self.algorithm_colors['PPO'], self.algorithm_colors['MILP']], legend=False)
        ax.set_title('Final State of Health Distribution', fontweight='bold')
        ax.set_ylabel('Final SOH (%)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        soc_path = self.output_dir / "soc_utilization_analysis.png"
        plt.savefig(soc_path, **self.fig_params)
        plt.savefig(soc_path.with_suffix('.pdf'), **self.fig_params)
        plt.close()
        
        saved_files['soc_utilization'] = str(soc_path)
        
        return saved_files
    
    def create_flexibility_heatmaps(self, episode_results: List[EpisodeResult]) -> Dict[str, str]:
        """
        Create heatmaps for flexibility service usage patterns
        
        Args:
            episode_results: Detailed episode results
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        saved_files = {}
        
        # Create flexibility service heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Flexibility Service Usage Patterns (BESS Heatmaps)', fontsize=16, fontweight='bold')
        
        services = ['FCR', 'aFRR', 'mFRR']
        algorithms = ['PPO', 'MILP']
        
        for alg_idx, algorithm in enumerate(algorithms):
            alg_results = [r for r in episode_results if r.algorithm == algorithm]
            
            for svc_idx, service in enumerate(services):
                ax = axes[alg_idx, svc_idx]
                
                # Aggregate flexibility reservations across episodes
                hourly_reservations = {}
                day_reservations = {}
                
                for result in alg_results:
                    if (result.flexibility_reservations and 
                        service in result.flexibility_reservations):
                        
                        reservations = result.flexibility_reservations[service]
                        
                        for hour_idx, reservation in enumerate(reservations):
                            hour_of_day = hour_idx % 24
                            day_of_week = (hour_idx // 24) % 7
                            
                            key = (day_of_week, hour_of_day)
                            if key not in hourly_reservations:
                                hourly_reservations[key] = []
                            hourly_reservations[key].append(reservation)
                
                # Create heatmap data
                if hourly_reservations:
                    heatmap_data = np.zeros((7, 24))  # 7 days x 24 hours
                    
                    for (day, hour), reservations in hourly_reservations.items():
                        if reservations:
                            heatmap_data[day, hour] = np.mean(reservations)
                    
                    # Create heatmap
                    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', 
                                  interpolation='nearest')
                    
                    # Customize axes
                    ax.set_xticks(range(0, 24, 4))
                    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 4)])
                    ax.set_yticks(range(7))
                    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                    
                    ax.set_title(f'{algorithm} - {service} Reservations', fontweight='bold')
                    if svc_idx == 0:
                        ax.set_ylabel('Day of Week', fontweight='bold')
                    if alg_idx == 1:
                        ax.set_xlabel('Hour of Day', fontweight='bold')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Avg Reservation (MW)', fontweight='bold')
                else:
                    # No data available
                    ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes,
                           ha='center', va='center', fontsize=12, fontweight='bold')
                    ax.set_title(f'{algorithm} - {service} Reservations', fontweight='bold')
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = self.output_dir / "flexibility_usage_heatmaps.png"
        plt.savefig(heatmap_path, **self.fig_params)
        plt.savefig(heatmap_path.with_suffix('.pdf'), **self.fig_params)
        plt.close()
        
        saved_files['flexibility_heatmaps'] = str(heatmap_path)
        
        return saved_files
    
    def create_statistical_summary_plots(self, summary: ComparisonSummary) -> Dict[str, str]:
        """
        Create statistical analysis summary visualizations
        
        Args:
            summary: Comparison summary with statistical tests
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        saved_files = {}
        
        # Create statistical summary dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
        
        # Algorithm performance comparison
        ax = axes[0, 0]
        
        algorithms = ['PPO', 'MILP']
        metrics = ['mean', 'std', 'min', 'max']
        
        ppo_profits = [
            summary.ppo_results.get('profit', {}).get(metric, 0) 
            for metric in metrics
        ]
        milp_profits = [
            summary.milp_results.get('profit', {}).get(metric, 0) 
            for metric in metrics
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ppo_profits, width, label='PPO', 
                      color=self.algorithm_colors['PPO'], alpha=0.7)
        bars2 = ax.bar(x + width/2, milp_profits, width, label='MILP',
                      color=self.algorithm_colors['MILP'], alpha=0.7)
        
        ax.set_title('Profit Statistics Comparison', fontweight='bold')
        ax.set_ylabel('Profit (EUR)', fontweight='bold')
        ax.set_xlabel('Statistic', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Robustness comparison
        ax = axes[0, 1]
        
        if summary.robustness_metrics:
            robustness_data = []
            
            for algorithm in ['PPO', 'MILP']:
                if algorithm in summary.robustness_metrics:
                    metrics = summary.robustness_metrics[algorithm]
                    robustness_data.append({
                        'Algorithm': algorithm,
                        'Mean': metrics.get('mean_across_distributions', 0),
                        'Std': metrics.get('std_across_distributions', 0),
                        'CV': metrics.get('coefficient_of_variation', 0)
                    })
            
            if robustness_data:
                df_robust = pd.DataFrame(robustness_data)
                
                # Plot coefficient of variation (lower is more robust)
                bars = ax.bar(df_robust['Algorithm'], df_robust['CV'], 
                             color=[self.algorithm_colors[alg] for alg in df_robust['Algorithm']],
                             alpha=0.7)
                
                ax.set_title('Robustness (Coefficient of Variation)', fontweight='bold')
                ax.set_ylabel('Coefficient of Variation', fontweight='bold')
                ax.set_xlabel('Algorithm', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, df_robust['CV']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height * 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Statistical significance tests
        ax = axes[1, 0]
        
        if summary.statistical_tests:
            test_results = []
            test_names = []
            p_values = []
            
            for test_name, test_data in summary.statistical_tests.items():
                if isinstance(test_data, dict) and 'p_value' in test_data:
                    test_names.append(test_name.replace('_', ' ').title())
                    p_values.append(test_data['p_value'])
                    test_results.append('Significant' if test_data.get('significant', False) else 'Not Significant')
            
            if test_names:
                colors = ['green' if result == 'Significant' else 'red' for result in test_results]
                
                bars = ax.barh(test_names, p_values, color=colors, alpha=0.7)
                ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
                
                ax.set_title('Statistical Significance Tests', fontweight='bold')
                ax.set_xlabel('p-value', fontweight='bold')
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add value labels
                for bar, value in zip(bars, p_values):
                    width = bar.get_width()
                    ax.text(width * 1.1, bar.get_y() + bar.get_height()/2.,
                           f'{value:.4f}', ha='left', va='center', fontweight='bold')
        
        # Execution time analysis
        ax = axes[1, 1]
        
        ppo_times = summary.ppo_results.get('execution_time', {})
        milp_times = summary.milp_results.get('execution_time', {})
        
        time_metrics = ['mean', 'std', 'min', 'max']
        ppo_time_values = [ppo_times.get(metric, 0) for metric in time_metrics]
        milp_time_values = [milp_times.get(metric, 0) for metric in time_metrics]
        
        x = np.arange(len(time_metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ppo_time_values, width, label='PPO',
                      color=self.algorithm_colors['PPO'], alpha=0.7)
        bars2 = ax.bar(x + width/2, milp_time_values, width, label='MILP',
                      color=self.algorithm_colors['MILP'], alpha=0.7)
        
        ax.set_title('Execution Time Statistics', fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontweight='bold')
        ax.set_xlabel('Statistic', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in time_metrics])
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save statistical summary
        stats_path = self.output_dir / "statistical_analysis_summary.png"
        plt.savefig(stats_path, **self.fig_params)
        plt.savefig(stats_path.with_suffix('.pdf'), **self.fig_params)
        plt.close()
        
        saved_files['statistical_summary'] = str(stats_path)
        
        return saved_files
    
    def create_interactive_dashboard(self, summary: ComparisonSummary, 
                                   episode_results: List[EpisodeResult]) -> str:
        """
        Create interactive Plotly dashboard for detailed analysis
        
        Args:
            summary: Comparison summary statistics
            episode_results: Detailed episode results
            
        Returns:
            Path to saved HTML dashboard
        """
        # Create subplots for interactive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Profit Distribution by Algorithm',
                'Execution Time Comparison',
                'Battery Utilization Analysis', 
                'Flexibility Service Usage',
                'SOC Trajectories (Sample)',
                'Statistical Test Results'
            ],
            specs=[
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "violin"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Prepare data
        ppo_results = [r for r in episode_results if r.algorithm == 'PPO']
        milp_results = [r for r in episode_results if r.algorithm == 'MILP']
        
        # 1. Profit distribution boxplot
        ppo_profits = [r.net_profit for r in ppo_results if r.net_profit is not None]
        milp_profits = [r.net_profit for r in milp_results if r.net_profit is not None]
        
        fig.add_trace(
            go.Box(y=ppo_profits, name='PPO', marker_color=self.algorithm_colors['PPO']),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=milp_profits, name='MILP', marker_color=self.algorithm_colors['MILP']),
            row=1, col=1
        )
        
        # 2. Execution time comparison
        distributions = list(summary.execution_time_ratio.keys())
        time_ratios = list(summary.execution_time_ratio.values())
        
        fig.add_trace(
            go.Bar(x=distributions, y=time_ratios, name='MILP/PPO Time Ratio',
                  marker_color=self.algorithm_colors['MILP']),
            row=1, col=2
        )
        
        # 3. Battery utilization violin plot
        ppo_util = [r.battery_utilization * 100 for r in ppo_results if r.battery_utilization is not None]
        milp_util = [r.battery_utilization * 100 for r in milp_results if r.battery_utilization is not None]
        
        fig.add_trace(
            go.Violin(y=ppo_util, name='PPO Utilization', 
                     line_color=self.algorithm_colors['PPO']),
            row=2, col=1
        )
        fig.add_trace(
            go.Violin(y=milp_util, name='MILP Utilization',
                     line_color=self.algorithm_colors['MILP']),
            row=2, col=1
        )
        
        # 4. Flexibility service usage
        service_usage = {'FCR': 0, 'aFRR': 0, 'mFRR': 0}
        
        for result in episode_results:
            if result.flexibility_reservations:
                for service, reservations in result.flexibility_reservations.items():
                    if reservations and service in service_usage:
                        service_usage[service] += sum(r for r in reservations if r > 0)
        
        fig.add_trace(
            go.Bar(x=list(service_usage.keys()), y=list(service_usage.values()),
                  name='Total Service Usage',
                  marker_color=[self.service_colors[s] for s in service_usage.keys()]),
            row=2, col=2
        )
        
        # 5. Sample SOC trajectories
        sample_results = episode_results[:4]  # First 4 episodes
        
        for i, result in enumerate(sample_results):
            if result.soc_trajectory:
                hours = list(range(len(result.soc_trajectory)))
                soc_percent = [s * 100 for s in result.soc_trajectory]
                
                fig.add_trace(
                    go.Scatter(x=hours, y=soc_percent, 
                             name=f'{result.algorithm} ({result.error_distribution})',
                             line=dict(color=self.algorithm_colors[result.algorithm])),
                    row=3, col=1
                )
        
        # 6. Statistical test results
        if summary.statistical_tests:
            test_names = []
            p_values = []
            
            for test_name, test_data in summary.statistical_tests.items():
                if isinstance(test_data, dict) and 'p_value' in test_data:
                    test_names.append(test_name.replace('_', ' ').title())
                    p_values.append(test_data['p_value'])
            
            if test_names:
                colors = ['green' if p < 0.05 else 'red' for p in p_values]
                
                fig.add_trace(
                    go.Bar(x=test_names, y=p_values, name='p-values',
                          marker_color=colors),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="PPO vs MILP Interactive Comparison Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Net Profit (EUR)", row=1, col=1)
        fig.update_yaxes(title_text="Time Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Utilization (%)", row=2, col=1)
        fig.update_yaxes(title_text="Total Usage (MW)", row=2, col=2)
        fig.update_yaxes(title_text="SOC (%)", row=3, col=1)
        fig.update_yaxes(title_text="p-value", row=3, col=2)
        
        fig.update_xaxes(title_text="Error Distribution", row=1, col=2)
        fig.update_xaxes(title_text="Service Type", row=2, col=2)
        fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
        fig.update_xaxes(title_text="Statistical Test", row=3, col=2)
        
        # Save interactive dashboard
        dashboard_path = self.output_dir / "interactive_dashboard.html"
        pyo.plot(fig, filename=str(dashboard_path), auto_open=False)
        
        return str(dashboard_path)
    
    def generate_comprehensive_report(self, summary: ComparisonSummary, 
                                    episode_results: List[EpisodeResult]) -> Dict[str, str]:
        """
        Generate comprehensive visualization report with all plots
        
        Args:
            summary: Comparison summary statistics
            episode_results: Detailed episode results
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        print("Generating comprehensive visualization report...")
        
        all_files = {}
        
        # Generate all visualization types
        print("  Creating profit comparison plots...")
        profit_files = self.create_profit_comparison_plots(summary, episode_results)
        all_files.update(profit_files)
        
        print("  Creating SOC and utilization plots...")
        soc_files = self.create_soc_utilization_plots(episode_results)
        all_files.update(soc_files)
        
        print("  Creating flexibility service heatmaps...")
        heatmap_files = self.create_flexibility_heatmaps(episode_results)
        all_files.update(heatmap_files)
        
        print("  Creating statistical summary plots...")
        stats_files = self.create_statistical_summary_plots(summary)
        all_files.update(stats_files)
        
        print("  Creating interactive dashboard...")
        dashboard_path = self.create_interactive_dashboard(summary, episode_results)
        all_files['interactive_dashboard'] = dashboard_path
        
        # Create summary report
        print("  Generating summary report...")
        report_path = self._create_summary_report(summary, all_files)
        all_files['summary_report'] = report_path
        
        print(f"✓ Comprehensive report generated with {len(all_files)} visualizations")
        print(f"  Output directory: {self.output_dir}")
        
        return all_files
    
    def _create_summary_report(self, summary: ComparisonSummary, 
                              visualization_files: Dict[str, str]) -> str:
        """Create a text summary report of the comparison results"""
        
        report_path = self.output_dir / "comparison_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PPO vs MILP BATTERY TRADING COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Configuration summary
            f.write("CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Time Horizon: {summary.config.time_horizon} hours\n")
            f.write(f"Error Distributions: {', '.join(summary.config.error_distributions)}\n")
            f.write(f"Episodes per Distribution: {summary.config.n_episodes_per_distribution}\n")
            f.write(f"Battery Capacity: {summary.config.battery_capacity_mwh} MWh\n")
            f.write(f"Battery Max Power: {summary.config.battery_max_power_mw} MW\n")
            f.write(f"Flexibility Enabled: {summary.config.flexibility_enabled}\n\n")
            
            # Results summary
            f.write("RESULTS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Episodes: {summary.total_episodes}\n")
            f.write(f"Successful Episodes: {summary.successful_episodes}\n")
            f.write(f"Failed Episodes: {summary.failed_episodes}\n")
            f.write(f"Success Rate: {summary.successful_episodes/summary.total_episodes*100:.1f}%\n\n")
            
            # Algorithm performance
            f.write("ALGORITHM PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            
            for alg_name, alg_results in [('PPO', summary.ppo_results), ('MILP', summary.milp_results)]:
                if alg_results and 'profit' in alg_results:
                    profit_stats = alg_results['profit']
                    f.write(f"{alg_name}:\n")
                    f.write(f"  Mean Profit: {profit_stats.get('mean', 0):.2f} EUR\n")
                    f.write(f"  Profit Std: {profit_stats.get('std', 0):.2f} EUR\n")
                    f.write(f"  Min Profit: {profit_stats.get('min', 0):.2f} EUR\n")
                    f.write(f"  Max Profit: {profit_stats.get('max', 0):.2f} EUR\n")
                    
                    if 'execution_time' in alg_results:
                        time_stats = alg_results['execution_time']
                        f.write(f"  Mean Execution Time: {time_stats.get('mean', 0):.4f} seconds\n")
                    f.write("\n")
            
            # Profit advantage by distribution
            f.write("PROFIT ADVANTAGE BY DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            for dist, advantage in summary.profit_advantage.items():
                winner = "MILP" if advantage > 0 else "PPO"
                f.write(f"{dist}: {advantage:.2f} EUR (advantage to {winner})\n")
            f.write("\n")
            
            # Execution time ratios
            f.write("EXECUTION TIME RATIOS (MILP/PPO):\n")
            f.write("-" * 40 + "\n")
            for dist, ratio in summary.execution_time_ratio.items():
                f.write(f"{dist}: {ratio:.2f}x\n")
            f.write("\n")
            
            # Statistical tests
            if summary.statistical_tests:
                f.write("STATISTICAL SIGNIFICANCE TESTS:\n")
                f.write("-" * 40 + "\n")
                for test_name, test_data in summary.statistical_tests.items():
                    if isinstance(test_data, dict):
                        significance = "Significant" if test_data.get('significant', False) else "Not Significant"
                        p_value = test_data.get('p_value', 0)
                        f.write(f"{test_name}: p={p_value:.4f} ({significance})\n")
                f.write("\n")
            
            # Generated files
            f.write("GENERATED VISUALIZATIONS:\n")
            f.write("-" * 40 + "\n")
            for viz_type, filepath in visualization_files.items():
                f.write(f"{viz_type}: {filepath}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Report generated at: {summary.timestamp}\n")
            f.write("=" * 80 + "\n")
        
        return str(report_path)


# Utility functions for visualization support

def validate_visualization_completeness(visualization_files: Dict[str, str]) -> bool:
    """
    Validate that all required visualizations have been generated
    Implements Property 20: Visualization Completeness validation
    
    Args:
        visualization_files: Dictionary of generated visualization files
        
    Returns:
        True if all required visualizations are present
    """
    required_visualizations = [
        'profit_comparison',
        'soc_utilization', 
        'flexibility_heatmaps',
        'statistical_summary',
        'interactive_dashboard'
    ]
    
    missing_visualizations = []
    for required in required_visualizations:
        if required not in visualization_files:
            missing_visualizations.append(required)
        else:
            # Check if file actually exists
            filepath = Path(visualization_files[required])
            if not filepath.exists():
                missing_visualizations.append(f"{required} (file not found)")
    
    if missing_visualizations:
        print(f"Missing visualizations: {missing_visualizations}")
        return False
    
    return True


def export_visualization_metadata(visualization_files: Dict[str, str], 
                                output_path: str) -> None:
    """
    Export metadata about generated visualizations for tracking
    
    Args:
        visualization_files: Dictionary of generated visualization files
        output_path: Path to save metadata JSON
    """
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_visualizations': len(visualization_files),
        'visualization_types': list(visualization_files.keys()),
        'file_paths': visualization_files,
        'file_sizes': {}
    }
    
    # Add file size information
    for viz_type, filepath in visualization_files.items():
        path = Path(filepath)
        if path.exists():
            metadata['file_sizes'][viz_type] = path.stat().st_size
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)