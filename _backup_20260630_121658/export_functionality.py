"""
Export Functionality for PPO vs MILP Visualization and Reporting System
Maintains high-resolution PDF export and adds interactive dashboard capabilities
Implements Requirements 7.5, 7.6 for consistent export formats
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.io as pio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import json
import zipfile
import shutil
from datetime import datetime
from dataclasses import asdict
import base64
import io

# Import system components
from visualization_system import VisualizationSystem
from statistical_analysis_reporting import StatisticalAnalysisReporter, ComprehensiveStatisticalReport
from comparison_engine import ComparisonSummary, EpisodeResult


class ExportManager:
    """
    Comprehensive export functionality manager
    Implements Requirements 7.5, 7.6 for high-resolution PDF and interactive formats
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize export manager
        
        Args:
            output_dir: Base directory for all exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different export types
        self.pdf_dir = self.output_dir / "pdf_exports"
        self.interactive_dir = self.output_dir / "interactive_exports"
        self.data_dir = self.output_dir / "data_exports"
        self.archive_dir = self.output_dir / "archives"
        
        for directory in [self.pdf_dir, self.interactive_dir, self.data_dir, self.archive_dir]:
            directory.mkdir(exist_ok=True)
        
        # Configure high-quality PDF settings
        self.pdf_settings = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'format': 'pdf',
            'metadata': {
                'Title': 'PPO vs MILP Battery Trading Analysis',
                'Author': 'Battery Flexibility Analysis System',
                'Subject': 'Italian Market Flexibility Services Comparison',
                'Creator': 'Python Matplotlib/Plotly Export System'
            }
        }
        
        # Configure interactive dashboard settings
        self.interactive_settings = {
            'include_plotlyjs': True,
            'div_id': 'ppo-milp-dashboard',
            'config': {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'ppo_milp_analysis',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
        }
    
    def export_high_resolution_pdfs(self, visualization_files: Dict[str, str], 
                                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Export all visualizations as high-resolution PDFs
        Implements Requirement 7.5: High-resolution PDF export
        
        Args:
            visualization_files: Dictionary mapping visualization types to file paths
            metadata: Optional metadata to include in PDF properties
            
        Returns:
            Dictionary mapping visualization types to PDF file paths
        """
        pdf_exports = {}
        
        print("Exporting high-resolution PDFs...")
        
        # Update metadata if provided
        if metadata:
            self.pdf_settings['metadata'].update(metadata)
        
        for viz_type, source_path in visualization_files.items():
            if viz_type == 'interactive_dashboard':
                # Skip interactive dashboard for PDF export
                continue
            
            source_file = Path(source_path)
            
            if source_file.exists() and source_file.suffix in ['.png', '.jpg', '.jpeg']:
                # Convert existing image to high-resolution PDF
                pdf_path = self.pdf_dir / f"{viz_type}_high_res.pdf"
                
                try:
                    # Create PDF with matplotlib for consistent high quality
                    fig, ax = plt.subplots(figsize=(16, 12))
                    
                    # Load and display image
                    img = plt.imread(source_file)
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Set title based on visualization type
                    title = self._get_visualization_title(viz_type)
                    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
                    
                    # Save as high-resolution PDF
                    plt.savefig(pdf_path, **self.pdf_settings)
                    plt.close()
                    
                    pdf_exports[viz_type] = str(pdf_path)
                    print(f"  ✓ Exported {viz_type} to {pdf_path}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to export {viz_type}: {e}")
        
        # Create combined PDF report
        combined_pdf_path = self._create_combined_pdf_report(pdf_exports, metadata)
        if combined_pdf_path:
            pdf_exports['combined_report'] = combined_pdf_path
        
        print(f"✓ PDF export complete. {len(pdf_exports)} files exported.")
        return pdf_exports
    
    def create_interactive_dashboard(self, summary: ComparisonSummary, 
                                   episode_results: List[EpisodeResult],
                                   statistical_report: Optional[ComprehensiveStatisticalReport] = None) -> str:
        """
        Create comprehensive interactive dashboard
        Implements Requirement 7.6: Interactive dashboard for analysis
        
        Args:
            summary: Comparison summary statistics
            episode_results: Detailed episode results
            statistical_report: Optional statistical analysis report
            
        Returns:
            Path to interactive dashboard HTML file
        """
        print("Creating interactive dashboard...")
        
        # Create comprehensive dashboard with multiple tabs/sections
        dashboard_html = self._build_dashboard_html(summary, episode_results, statistical_report)
        
        # Save dashboard
        dashboard_path = self.interactive_dir / "comprehensive_dashboard.html"
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        # Create standalone dashboard assets
        self._create_dashboard_assets(dashboard_path.parent)
        
        print(f"✓ Interactive dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def export_data_formats(self, summary: ComparisonSummary, 
                           episode_results: List[EpisodeResult],
                           statistical_report: Optional[ComprehensiveStatisticalReport] = None) -> Dict[str, str]:
        """
        Export data in multiple formats (CSV, JSON, Excel)
        
        Args:
            summary: Comparison summary statistics
            episode_results: Detailed episode results
            statistical_report: Optional statistical analysis report
            
        Returns:
            Dictionary mapping format names to file paths
        """
        data_exports = {}
        
        print("Exporting data in multiple formats...")
        
        # 1. Export episode results as CSV
        csv_path = self._export_episode_results_csv(episode_results)
        if csv_path:
            data_exports['episode_results_csv'] = csv_path
        
        # 2. Export summary as JSON
        json_path = self._export_summary_json(summary)
        if json_path:
            data_exports['summary_json'] = json_path
        
        # 3. Export statistical report as JSON
        if statistical_report:
            stats_json_path = self._export_statistical_report_json(statistical_report)
            if stats_json_path:
                data_exports['statistical_report_json'] = stats_json_path
        
        # 4. Create Excel workbook with multiple sheets
        excel_path = self._create_excel_workbook(summary, episode_results, statistical_report)
        if excel_path:
            data_exports['comprehensive_excel'] = excel_path
        
        print(f"✓ Data export complete. {len(data_exports)} files exported.")
        return data_exports
    
    def create_export_archive(self, visualization_files: Dict[str, str],
                            pdf_exports: Dict[str, str],
                            data_exports: Dict[str, str],
                            dashboard_path: str) -> str:
        """
        Create comprehensive export archive with all outputs
        
        Args:
            visualization_files: Original visualization files
            pdf_exports: High-resolution PDF exports
            data_exports: Data format exports
            dashboard_path: Interactive dashboard path
            
        Returns:
            Path to created archive file
        """
        print("Creating comprehensive export archive...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / f"ppo_milp_analysis_{timestamp}.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Add visualizations
            for viz_type, file_path in visualization_files.items():
                if Path(file_path).exists():
                    zipf.write(file_path, f"visualizations/{Path(file_path).name}")
            
            # Add PDF exports
            for viz_type, file_path in pdf_exports.items():
                if Path(file_path).exists():
                    zipf.write(file_path, f"pdf_exports/{Path(file_path).name}")
            
            # Add data exports
            for format_type, file_path in data_exports.items():
                if Path(file_path).exists():
                    zipf.write(file_path, f"data_exports/{Path(file_path).name}")
            
            # Add interactive dashboard
            if Path(dashboard_path).exists():
                zipf.write(dashboard_path, f"interactive/{Path(dashboard_path).name}")
                
                # Add dashboard assets if they exist
                dashboard_dir = Path(dashboard_path).parent
                for asset_file in dashboard_dir.glob("*.css"):
                    zipf.write(asset_file, f"interactive/{asset_file.name}")
                for asset_file in dashboard_dir.glob("*.js"):
                    zipf.write(asset_file, f"interactive/{asset_file.name}")
            
            # Add metadata file
            metadata = self._create_export_metadata(visualization_files, pdf_exports, 
                                                  data_exports, dashboard_path)
            zipf.writestr("export_metadata.json", json.dumps(metadata, indent=2))
            
            # Add README
            readme_content = self._create_export_readme(metadata)
            zipf.writestr("README.txt", readme_content)
        
        print(f"✓ Export archive created: {archive_path}")
        return str(archive_path)
    
    def _get_visualization_title(self, viz_type: str) -> str:
        """Get appropriate title for visualization type"""
        titles = {
            'profit_comparison': 'PPO vs MILP Profit Comparison Analysis',
            'soc_utilization': 'Battery SOC and Utilization Analysis',
            'flexibility_heatmaps': 'Flexibility Service Usage Heatmaps',
            'statistical_summary': 'Statistical Analysis Summary',
            'robustness_analysis': 'Algorithm Robustness Analysis',
            'performance_metrics': 'Performance Metrics Comparison'
        }
        return titles.get(viz_type, f'PPO vs MILP Analysis: {viz_type.replace("_", " ").title()}')
    
    def _create_combined_pdf_report(self, pdf_exports: Dict[str, str], 
                                  metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create combined PDF report with all visualizations"""
        
        if not pdf_exports:
            return None
        
        combined_path = self.pdf_dir / "combined_analysis_report.pdf"
        
        try:
            with pdf_backend.PdfPages(combined_path, metadata=self.pdf_settings.get('metadata', {})) as pdf:
                
                # Create title page
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.text(0.5, 0.7, 'PPO vs MILP Battery Trading Analysis', 
                       ha='center', va='center', fontsize=24, fontweight='bold',
                       transform=ax.transAxes)
                
                ax.text(0.5, 0.6, 'Comprehensive Comparison Report', 
                       ha='center', va='center', fontsize=16,
                       transform=ax.transAxes)
                
                ax.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                
                if metadata:
                    info_text = []
                    for key, value in metadata.items():
                        if key not in ['Title', 'Author', 'Subject', 'Creator']:
                            info_text.append(f"{key}: {value}")
                    
                    if info_text:
                        ax.text(0.5, 0.2, '\n'.join(info_text), 
                               ha='center', va='center', fontsize=10,
                               transform=ax.transAxes)
                
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Add each visualization
                for viz_type, pdf_path in pdf_exports.items():
                    if viz_type == 'combined_report':
                        continue
                    
                    if Path(pdf_path).exists():
                        # Read the PDF and add to combined report
                        try:
                            # For simplicity, we'll recreate the figure
                            # In a full implementation, you'd use PyPDF2 or similar
                            fig, ax = plt.subplots(figsize=(11, 8.5))
                            
                            # Add a page with the visualization title
                            title = self._get_visualization_title(viz_type)
                            ax.text(0.5, 0.5, f'See separate file:\n{Path(pdf_path).name}', 
                                   ha='center', va='center', fontsize=14,
                                   transform=ax.transAxes)
                            ax.text(0.5, 0.7, title, 
                                   ha='center', va='center', fontsize=16, fontweight='bold',
                                   transform=ax.transAxes)
                            ax.axis('off')
                            
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close()
                            
                        except Exception as e:
                            print(f"Warning: Could not add {viz_type} to combined PDF: {e}")
            
            return str(combined_path)
            
        except Exception as e:
            print(f"Error creating combined PDF: {e}")
            return None
    
    def _build_dashboard_html(self, summary: ComparisonSummary, 
                            episode_results: List[EpisodeResult],
                            statistical_report: Optional[ComprehensiveStatisticalReport] = None) -> str:
        """Build comprehensive HTML dashboard"""
        
        # Create Plotly figures for interactive dashboard
        figures = self._create_interactive_figures(summary, episode_results, statistical_report)
        
        # Build HTML structure
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PPO vs MILP Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="dashboard_styles.css">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .dashboard-section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .tabs {{
            display: flex;
            background: #e9ecef;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .tab {{
            flex: 1;
            padding: 10px 20px;
            text-align: center;
            cursor: pointer;
            border: none;
            background: transparent;
            transition: background-color 0.3s;
        }}
        .tab.active {{
            background: #667eea;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PPO vs MILP Battery Trading Analysis</h1>
        <p>Interactive Dashboard for Italian Flexibility Market Comparison</p>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="dashboard-grid">
        <div class="dashboard-section">
            <div class="section-title">Key Metrics</div>
            {key_metrics_html}
        </div>
        
        <div class="dashboard-section">
            <div class="section-title">Algorithm Performance</div>
            {performance_summary_html}
        </div>
    </div>
    
    <div class="dashboard-section full-width">
        <div class="section-title">Interactive Analysis</div>
        <div class="tabs">
            <button class="tab active" onclick="showTab('profit-analysis')">Profit Analysis</button>
            <button class="tab" onclick="showTab('performance-metrics')">Performance Metrics</button>
            <button class="tab" onclick="showTab('flexibility-services')">Flexibility Services</button>
            <button class="tab" onclick="showTab('statistical-tests')">Statistical Tests</button>
        </div>
        
        <div id="profit-analysis" class="tab-content active">
            {profit_analysis_html}
        </div>
        
        <div id="performance-metrics" class="tab-content">
            {performance_metrics_html}
        </div>
        
        <div id="flexibility-services" class="tab-content">
            {flexibility_services_html}
        </div>
        
        <div id="statistical-tests" class="tab-content">
            {statistical_tests_html}
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
        
        // Initialize dashboard
        window.addEventListener('load', function() {{
            console.log('PPO vs MILP Dashboard loaded successfully');
        }});
    </script>
</body>
</html>
        """
        
        # Generate content sections
        key_metrics_html = self._generate_key_metrics_html(summary)
        performance_summary_html = self._generate_performance_summary_html(summary)
        
        # Generate interactive plot HTML
        profit_analysis_html = pio.to_html(figures['profit_analysis'], include_plotlyjs=False, div_id="profit-plot")
        performance_metrics_html = pio.to_html(figures['performance_metrics'], include_plotlyjs=False, div_id="performance-plot")
        flexibility_services_html = pio.to_html(figures['flexibility_services'], include_plotlyjs=False, div_id="flexibility-plot")
        
        if statistical_report:
            statistical_tests_html = self._generate_statistical_tests_html(statistical_report)
        else:
            statistical_tests_html = "<p>Statistical analysis not available.</p>"
        
        # Format the template
        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            key_metrics_html=key_metrics_html,
            performance_summary_html=performance_summary_html,
            profit_analysis_html=profit_analysis_html,
            performance_metrics_html=performance_metrics_html,
            flexibility_services_html=flexibility_services_html,
            statistical_tests_html=statistical_tests_html
        )
    
    def _create_interactive_figures(self, summary: ComparisonSummary, 
                                  episode_results: List[EpisodeResult],
                                  statistical_report: Optional[ComprehensiveStatisticalReport] = None) -> Dict[str, go.Figure]:
        """Create interactive Plotly figures for dashboard"""
        
        figures = {}
        
        # 1. Profit Analysis Figure
        ppo_results = [r for r in episode_results if r.algorithm == 'PPO']
        milp_results = [r for r in episode_results if r.algorithm == 'MILP']
        
        fig_profit = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Profit Distribution', 'Profit by Error Distribution', 
                          'Cumulative Profit Advantage', 'Execution Time Comparison'],
            specs=[[{"type": "box"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Profit distribution
        ppo_profits = [r.net_profit for r in ppo_results if r.net_profit is not None]
        milp_profits = [r.net_profit for r in milp_results if r.net_profit is not None]
        
        fig_profit.add_trace(go.Box(y=ppo_profits, name='PPO', marker_color='#FF6B6B'), row=1, col=1)
        fig_profit.add_trace(go.Box(y=milp_profits, name='MILP', marker_color='#4ECDC4'), row=1, col=1)
        
        # Profit by distribution
        for dist in summary.config.error_distributions:
            ppo_dist = [r.net_profit for r in ppo_results if r.error_distribution == dist and r.net_profit is not None]
            milp_dist = [r.net_profit for r in milp_results if r.error_distribution == dist and r.net_profit is not None]
            
            fig_profit.add_trace(go.Box(y=ppo_dist, name=f'PPO-{dist}', marker_color='#FF6B6B', 
                                      showlegend=False), row=1, col=2)
            fig_profit.add_trace(go.Box(y=milp_dist, name=f'MILP-{dist}', marker_color='#4ECDC4', 
                                      showlegend=False), row=1, col=2)
        
        # Profit advantage
        distributions = list(summary.profit_advantage.keys())
        advantages = list(summary.profit_advantage.values())
        colors = ['green' if adv > 0 else 'red' for adv in advantages]
        
        fig_profit.add_trace(go.Bar(x=distributions, y=advantages, marker_color=colors, 
                                  name='MILP Advantage', showlegend=False), row=2, col=1)
        
        # Execution time ratios
        time_ratios = list(summary.execution_time_ratio.values())
        fig_profit.add_trace(go.Bar(x=distributions, y=time_ratios, marker_color='#4ECDC4', 
                                  name='Time Ratio', showlegend=False), row=2, col=2)
        
        fig_profit.update_layout(height=800, title_text="Profit Analysis Dashboard")
        figures['profit_analysis'] = fig_profit
        
        # 2. Performance Metrics Figure
        fig_performance = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Battery Utilization', 'Final SOH Distribution', 
                          'Execution Time Distribution', 'Algorithm Efficiency'],
            specs=[[{"type": "violin"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Battery utilization
        ppo_util = [r.battery_utilization * 100 for r in ppo_results if r.battery_utilization is not None]
        milp_util = [r.battery_utilization * 100 for r in milp_results if r.battery_utilization is not None]
        
        fig_performance.add_trace(go.Violin(y=ppo_util, name='PPO', line_color='#FF6B6B'), row=1, col=1)
        fig_performance.add_trace(go.Violin(y=milp_util, name='MILP', line_color='#4ECDC4'), row=1, col=1)
        
        # Final SOH
        ppo_soh = [r.final_soh * 100 for r in ppo_results if r.final_soh is not None]
        milp_soh = [r.final_soh * 100 for r in milp_results if r.final_soh is not None]
        
        fig_performance.add_trace(go.Histogram(x=ppo_soh, name='PPO SOH', marker_color='#FF6B6B', 
                                             opacity=0.7), row=1, col=2)
        fig_performance.add_trace(go.Histogram(x=milp_soh, name='MILP SOH', marker_color='#4ECDC4', 
                                             opacity=0.7), row=1, col=2)
        
        fig_performance.update_layout(height=800, title_text="Performance Metrics Dashboard")
        figures['performance_metrics'] = fig_performance
        
        # 3. Flexibility Services Figure
        fig_flexibility = go.Figure()
        
        # Aggregate flexibility usage by service type
        service_usage = {'FCR': 0, 'aFRR': 0, 'mFRR': 0}
        
        for result in episode_results:
            if result.flexibility_reservations:
                for service, reservations in result.flexibility_reservations.items():
                    if reservations and service in service_usage:
                        service_usage[service] += sum(r for r in reservations if r > 0)
        
        fig_flexibility.add_trace(go.Bar(
            x=list(service_usage.keys()),
            y=list(service_usage.values()),
            marker_color=['#FFD93D', '#6BCF7F', '#4D96FF'],
            name='Service Usage'
        ))
        
        fig_flexibility.update_layout(
            title="Flexibility Service Usage Analysis",
            xaxis_title="Service Type",
            yaxis_title="Total Usage (MW·h)",
            height=600
        )
        
        figures['flexibility_services'] = fig_flexibility
        
        return figures
    
    def _generate_key_metrics_html(self, summary: ComparisonSummary) -> str:
        """Generate key metrics HTML section"""
        
        # Calculate key metrics
        total_episodes = summary.total_episodes
        success_rate = (summary.successful_episodes / total_episodes * 100) if total_episodes > 0 else 0
        
        # Get profit advantage
        avg_profit_advantage = np.mean(list(summary.profit_advantage.values())) if summary.profit_advantage else 0
        
        # Get execution time ratio
        avg_time_ratio = np.mean(list(summary.execution_time_ratio.values())) if summary.execution_time_ratio else 0
        
        return f"""
        <div class="metric-card">
            <div class="metric-value">{total_episodes}</div>
            <div class="metric-label">Total Episodes Analyzed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_profit_advantage:+.0f} EUR</div>
            <div class="metric-label">Avg MILP Profit Advantage</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_time_ratio:.1f}x</div>
            <div class="metric-label">Avg MILP/PPO Time Ratio</div>
        </div>
        """
    
    def _generate_performance_summary_html(self, summary: ComparisonSummary) -> str:
        """Generate performance summary HTML section"""
        
        ppo_profit = summary.ppo_results.get('profit', {}).get('mean', 0)
        milp_profit = summary.milp_results.get('profit', {}).get('mean', 0)
        
        ppo_time = summary.ppo_results.get('execution_time', {}).get('mean', 0)
        milp_time = summary.milp_results.get('execution_time', {}).get('mean', 0)
        
        return f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <h4 style="margin: 0 0 10px 0; color: #FF6B6B;">PPO Performance</h4>
                <p><strong>Avg Profit:</strong> {ppo_profit:.0f} EUR</p>
                <p><strong>Avg Time:</strong> {ppo_time:.3f} sec</p>
            </div>
            <div>
                <h4 style="margin: 0 0 10px 0; color: #4ECDC4;">MILP Performance</h4>
                <p><strong>Avg Profit:</strong> {milp_profit:.0f} EUR</p>
                <p><strong>Avg Time:</strong> {milp_time:.3f} sec</p>
            </div>
        </div>
        """
    
    def _generate_statistical_tests_html(self, statistical_report: ComprehensiveStatisticalReport) -> str:
        """Generate statistical tests HTML section"""
        
        html = "<div class='section-title'>Statistical Significance Tests</div>"
        
        for test in statistical_report.significance_tests:
            significance_color = "green" if test.significant else "red"
            significance_text = "Significant" if test.significant else "Not Significant"
            
            html += f"""
            <div class="metric-card">
                <h4>{test.test_name}</h4>
                <p><strong>Statistic:</strong> {test.statistic:.4f}</p>
                <p><strong>p-value:</strong> {test.p_value:.4f}</p>
                <p><strong>Result:</strong> <span style="color: {significance_color}; font-weight: bold;">{significance_text}</span></p>
                <p><strong>Interpretation:</strong> {test.interpretation}</p>
            </div>
            """
        
        return html
    
    def _create_dashboard_assets(self, dashboard_dir: Path):
        """Create CSS and JS assets for dashboard"""
        
        # Create CSS file
        css_content = """
        /* Additional dashboard styles */
        .plotly-graph-div {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }
        
        .tab:hover {
            background-color: rgba(102, 126, 234, 0.1);
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .tabs {
                flex-direction: column;
            }
        }
        """
        
        css_path = dashboard_dir / "dashboard_styles.css"
        with open(css_path, 'w') as f:
            f.write(css_content)
    
    def _export_episode_results_csv(self, episode_results: List[EpisodeResult]) -> Optional[str]:
        """Export episode results as CSV"""
        
        try:
            # Convert episode results to DataFrame
            data = []
            for result in episode_results:
                row = {
                    'episode_id': result.episode_id,
                    'error_distribution': result.error_distribution,
                    'algorithm': result.algorithm,
                    'arbitrage_profit': result.arbitrage_profit,
                    'flexibility_revenue': result.flexibility_revenue,
                    'degradation_cost': result.degradation_cost,
                    'net_profit': result.net_profit,
                    'execution_time': result.execution_time,
                    'battery_utilization': result.battery_utilization,
                    'final_soh': result.final_soh,
                    'solver_status': result.solver_status
                }
                
                # Add flexibility reservations as separate columns
                if result.flexibility_reservations:
                    for service, reservations in result.flexibility_reservations.items():
                        if reservations:
                            row[f'{service}_total_reservation'] = sum(reservations)
                            row[f'{service}_avg_reservation'] = np.mean(reservations)
                            row[f'{service}_max_reservation'] = max(reservations)
                
                data.append(row)
            
            df = pd.DataFrame(data)
            csv_path = self.data_dir / "episode_results.csv"
            df.to_csv(csv_path, index=False)
            
            return str(csv_path)
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return None
    
    def _export_summary_json(self, summary: ComparisonSummary) -> Optional[str]:
        """Export comparison summary as JSON"""
        
        try:
            # Convert summary to serializable format
            summary_dict = asdict(summary)
            
            json_path = self.data_dir / "comparison_summary.json"
            with open(json_path, 'w') as f:
                json.dump(summary_dict, f, indent=2, default=str)
            
            return str(json_path)
            
        except Exception as e:
            print(f"Error exporting summary JSON: {e}")
            return None
    
    def _export_statistical_report_json(self, statistical_report: ComprehensiveStatisticalReport) -> Optional[str]:
        """Export statistical report as JSON"""
        
        try:
            # Convert report to serializable format
            report_dict = asdict(statistical_report)
            
            json_path = self.data_dir / "statistical_report.json"
            with open(json_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            return str(json_path)
            
        except Exception as e:
            print(f"Error exporting statistical report JSON: {e}")
            return None
    
    def _create_excel_workbook(self, summary: ComparisonSummary, 
                             episode_results: List[EpisodeResult],
                             statistical_report: Optional[ComprehensiveStatisticalReport] = None) -> Optional[str]:
        """Create comprehensive Excel workbook with multiple sheets"""
        
        try:
            excel_path = self.data_dir / "comprehensive_analysis.xlsx"
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                
                # Sheet 1: Episode Results
                episode_data = []
                for result in episode_results:
                    row = asdict(result)
                    # Flatten flexibility reservations
                    if result.flexibility_reservations:
                        for service, reservations in result.flexibility_reservations.items():
                            if reservations:
                                row[f'{service}_total'] = sum(reservations)
                                row[f'{service}_avg'] = np.mean(reservations)
                    
                    # Remove complex nested data for Excel compatibility
                    row.pop('soc_trajectory', None)
                    row.pop('power_trajectory', None)
                    row.pop('flexibility_reservations', None)
                    row.pop('error_sequence', None)
                    
                    episode_data.append(row)
                
                df_episodes = pd.DataFrame(episode_data)
                df_episodes.to_excel(writer, sheet_name='Episode Results', index=False)
                
                # Sheet 2: Summary Statistics
                summary_data = {
                    'Metric': [],
                    'PPO': [],
                    'MILP': []
                }
                
                ppo_stats = summary.ppo_results.get('profit', {})
                milp_stats = summary.milp_results.get('profit', {})
                
                for metric in ['mean', 'std', 'min', 'max', 'median']:
                    summary_data['Metric'].append(f'Profit {metric.title()}')
                    summary_data['PPO'].append(ppo_stats.get(metric, 0))
                    summary_data['MILP'].append(milp_stats.get(metric, 0))
                
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary Statistics', index=False)
                
                # Sheet 3: Profit Advantage by Distribution
                advantage_data = {
                    'Error Distribution': list(summary.profit_advantage.keys()),
                    'MILP Profit Advantage (EUR)': list(summary.profit_advantage.values()),
                    'Execution Time Ratio (MILP/PPO)': list(summary.execution_time_ratio.values())
                }
                
                df_advantage = pd.DataFrame(advantage_data)
                df_advantage.to_excel(writer, sheet_name='Profit Advantage', index=False)
                
                # Sheet 4: Statistical Tests (if available)
                if statistical_report and statistical_report.significance_tests:
                    test_data = []
                    for test in statistical_report.significance_tests:
                        test_data.append({
                            'Test Name': test.test_name,
                            'Statistic': test.statistic,
                            'p-value': test.p_value,
                            'Significant': test.significant,
                            'Effect Size': test.effect_size,
                            'Interpretation': test.interpretation
                        })
                    
                    df_tests = pd.DataFrame(test_data)
                    df_tests.to_excel(writer, sheet_name='Statistical Tests', index=False)
            
            return str(excel_path)
            
        except Exception as e:
            print(f"Error creating Excel workbook: {e}")
            return None
    
    def _create_export_metadata(self, visualization_files: Dict[str, str],
                              pdf_exports: Dict[str, str],
                              data_exports: Dict[str, str],
                              dashboard_path: str) -> Dict[str, Any]:
        """Create metadata for export archive"""
        
        return {
            'export_timestamp': datetime.now().isoformat(),
            'export_version': '1.0',
            'description': 'PPO vs MILP Battery Trading Analysis Export',
            'contents': {
                'visualizations': {
                    'count': len(visualization_files),
                    'types': list(visualization_files.keys())
                },
                'pdf_exports': {
                    'count': len(pdf_exports),
                    'types': list(pdf_exports.keys())
                },
                'data_exports': {
                    'count': len(data_exports),
                    'formats': list(data_exports.keys())
                },
                'interactive_dashboard': {
                    'available': bool(dashboard_path),
                    'path': dashboard_path if dashboard_path else None
                }
            },
            'system_info': {
                'python_version': '3.12+',
                'required_packages': ['matplotlib', 'plotly', 'pandas', 'numpy', 'scipy']
            }
        }
    
    def _create_export_readme(self, metadata: Dict[str, Any]) -> str:
        """Create README content for export archive"""
        
        return f"""
PPO vs MILP Battery Trading Analysis Export
==========================================

Export Date: {metadata['export_timestamp']}
Version: {metadata['export_version']}

DESCRIPTION
-----------
{metadata['description']}

This archive contains a comprehensive analysis comparing PPO (Proximal Policy Optimization) 
and MILP (Mixed Integer Linear Programming) approaches for battery trading in the Italian 
flexibility market.

CONTENTS
--------
1. visualizations/ - Static visualization files (PNG format)
   - Contains {metadata['contents']['visualizations']['count']} visualization files
   - Types: {', '.join(metadata['contents']['visualizations']['types'])}

2. pdf_exports/ - High-resolution PDF versions of all visualizations
   - Contains {metadata['contents']['pdf_exports']['count']} PDF files
   - Suitable for publication and presentation

3. data_exports/ - Raw data in multiple formats
   - Contains {metadata['contents']['data_exports']['count']} data files
   - Formats: {', '.join(metadata['contents']['data_exports']['formats'])}

4. interactive/ - Interactive dashboard for detailed analysis
   - HTML dashboard with interactive plots
   - Open comprehensive_dashboard.html in a web browser

SYSTEM REQUIREMENTS
-------------------
- Python {metadata['system_info']['python_version']}
- Required packages: {', '.join(metadata['system_info']['required_packages'])}

USAGE
-----
1. For quick viewing: Open interactive/comprehensive_dashboard.html
2. For presentations: Use files in pdf_exports/
3. For further analysis: Import data from data_exports/
4. For publications: Use high-resolution PDFs

SUPPORT
-------
This export was generated by the Battery Flexibility Analysis System.
For questions or issues, refer to the system documentation.

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """.strip()


# Utility functions for export format validation

def validate_export_format_consistency(pdf_exports: Dict[str, str], 
                                     interactive_exports: Dict[str, str]) -> bool:
    """
    Validate that export formats are consistent
    Implements Property 23: Export Format Consistency validation
    
    Args:
        pdf_exports: Dictionary of PDF export files
        interactive_exports: Dictionary of interactive export files
        
    Returns:
        True if export formats are consistent
    """
    # Check PDF exports
    for export_type, filepath in pdf_exports.items():
        path = Path(filepath)
        if not path.exists():
            print(f"PDF export file missing: {filepath}")
            return False
        
        if path.suffix.lower() != '.pdf':
            print(f"PDF export has wrong format: {filepath}")
            return False
    
    # Check interactive exports
    for export_type, filepath in interactive_exports.items():
        path = Path(filepath)
        if not path.exists():
            print(f"Interactive export file missing: {filepath}")
            return False
        
        if path.suffix.lower() != '.html':
            print(f"Interactive export has wrong format: {filepath}")
            return False
    
    return True


def get_export_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about an exported file
    
    Args:
        filepath: Path to exported file
        
    Returns:
        Dictionary with file information
    """
    path = Path(filepath)
    
    if not path.exists():
        return {'exists': False}
    
    stat = path.stat()
    
    return {
        'exists': True,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'format': path.suffix.lower(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
    }