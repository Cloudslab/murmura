#!/usr/bin/env python3
"""
Analysis and Visualization Script for Resource-Aware Client Sampling Experiments

This script analyzes the results from resource_aware_sampling_experiments.py and generates
publication-ready plots and tables for the research paper:
"Resource-Aware Client Sampling Strategies for Differentially Private FL"

Author: Generated for Murmura Framework
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResourceExperimentAnalyzer:
    """Analyzer for resource-aware client sampling experiment results"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.plots_dir = self.experiment_dir / "analysis_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load experiment data
        self.load_experiment_data()
        
        logger.info(f"Loaded {len(self.df)} experiment results for analysis")
        logger.info(f"Plots will be saved to: {self.plots_dir}")
    
    def load_experiment_data(self):
        """Load and process experiment data"""
        # Load detailed analysis data
        analysis_file = self.experiment_dir / "detailed_analysis.json"
        if not analysis_file.exists():
            raise FileNotFoundError(f"Analysis file not found: {analysis_file}")
        
        with open(analysis_file, 'r') as f:
            self.analysis_data = json.load(f)
        
        # Convert to DataFrame
        rows = []
        for exp in self.analysis_data:
            if exp["execution"]["status"] == "success":
                base_data = {
                    "experiment_name": exp["experiment_name"],
                    "dataset": exp["dataset"],
                    "paradigm": exp["paradigm"],
                    "topology": exp["topology"],
                    "num_clients": exp["num_clients"],
                    "sampling_rate": exp["sampling_rate"],
                    "privacy_setting": exp["privacy_config"]["name"],
                    "dp_enabled": exp["privacy_config"]["enable_dp"],
                    "rounds": exp["training_config"]["rounds"],
                    "epochs": exp["training_config"]["epochs"],
                    "duration_seconds": exp["execution"]["duration_seconds"]
                }
                
                # Add privacy parameters if DP is enabled
                if exp["privacy_config"]["enable_dp"]:
                    base_data["target_epsilon"] = exp["privacy_config"]["target_epsilon_per_round"]
                    base_data["target_delta"] = exp["privacy_config"]["target_delta"]
                
                # Add resource analysis data
                if "resource_analysis" in exp:
                    ra = exp["resource_analysis"]
                    
                    # Resource summary data
                    if "resource_summary" in ra and "error" not in ra["resource_summary"]:
                        rs = ra["resource_summary"]
                        base_data.update({
                            "total_communication_bytes": rs.get("total_communication_bytes", 0),
                            "avg_round_time": rs.get("avg_round_time", 0),
                            "avg_efficiency_score": rs.get("avg_efficiency_score", 0),
                            "communication_savings_percent": rs.get("avg_communication_savings", 0)
                        })
                    
                    # Computation time data
                    if "computation_time" in ra and "error" not in ra["computation_time"]:
                        ct = ra["computation_time"]
                        base_data.update({
                            "total_training_time": ct.get("total_training_time", 0),
                            "avg_training_time_per_round": ct.get("avg_training_time_per_round", 0)
                        })
                    
                    # Sampling data
                    if "sampling" in ra and "error" not in ra["sampling"]:
                        sp = ra["sampling"]
                        base_data.update({
                            "actual_sampling_rate": sp.get("avg_actual_sampling_rate", 0),
                            "total_rounds_completed": sp.get("total_rounds", 0)
                        })
                
                rows.append(base_data)
        
        self.df = pd.DataFrame(rows)
        
        # Add derived metrics
        if len(self.df) > 0:
            self.df["communication_mb"] = self.df["total_communication_bytes"] / (1024 * 1024)
            self.df["time_per_client"] = self.df["avg_round_time"] / self.df["num_clients"]
            self.df["efficiency_per_mb"] = self.df["avg_efficiency_score"] / (self.df["communication_mb"] + 1e-6)
    
    def plot_sampling_rate_analysis(self):
        """Generate plots analyzing the effect of different sampling rates"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Impact of Client Sampling Rate on Resource Efficiency', fontsize=16, y=0.98)
        
        # 1. Communication vs Sampling Rate
        ax1 = axes[0, 0]
        for paradigm in self.df['paradigm'].unique():
            data = self.df[self.df['paradigm'] == paradigm]
            grouped = data.groupby('sampling_rate')['communication_mb'].mean()
            ax1.plot(grouped.index, grouped.values, marker='o', label=paradigm.title(), linewidth=2)
        
        ax1.set_xlabel('Client Sampling Rate')
        ax1.set_ylabel('Avg Communication (MB)')
        ax1.set_title('Communication vs Sampling Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training Time vs Sampling Rate
        ax2 = axes[0, 1]
        for paradigm in self.df['paradigm'].unique():
            data = self.df[self.df['paradigm'] == paradigm]
            grouped = data.groupby('sampling_rate')['avg_round_time'].mean()
            ax2.plot(grouped.index, grouped.values, marker='s', label=paradigm.title(), linewidth=2)
        
        ax2.set_xlabel('Client Sampling Rate')
        ax2.set_ylabel('Avg Round Time (seconds)')
        ax2.set_title('Training Time vs Sampling Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency Score vs Sampling Rate
        ax3 = axes[1, 0]
        for topology in self.df['topology'].unique():
            data = self.df[self.df['topology'] == topology]
            grouped = data.groupby('sampling_rate')['avg_efficiency_score'].mean()
            ax3.plot(grouped.index, grouped.values, marker='^', label=topology.title(), linewidth=2)
        
        ax3.set_xlabel('Client Sampling Rate')
        ax3.set_ylabel('Resource Efficiency Score')
        ax3.set_title('Efficiency vs Sampling Rate by Topology')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Communication Savings vs Sampling Rate
        ax4 = axes[1, 1]
        grouped = self.df.groupby('sampling_rate')['communication_savings_percent'].mean()
        ax4.bar(grouped.index, grouped.values, alpha=0.7, color='skyblue', edgecolor='navy')
        ax4.set_xlabel('Client Sampling Rate')
        ax4.set_ylabel('Communication Savings (%)')
        ax4.set_title('Communication Savings vs Sampling Rate')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sampling_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated sampling rate analysis plot")
    
    def plot_topology_comparison(self):
        """Generate plots comparing different network topologies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Network Topology Comparison', fontsize=16, y=0.98)
        
        # 1. Communication by Topology
        ax1 = axes[0, 0]
        topology_comm = self.df.groupby(['topology', 'paradigm'])['communication_mb'].mean().unstack()
        topology_comm.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_xlabel('Network Topology')
        ax1.set_ylabel('Avg Communication (MB)')
        ax1.set_title('Communication by Topology and Paradigm')
        ax1.legend(title='Learning Paradigm')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Training Time by Topology
        ax2 = axes[0, 1]
        topology_time = self.df.groupby(['topology', 'paradigm'])['avg_round_time'].mean().unstack()
        topology_time.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_xlabel('Network Topology')
        ax2.set_ylabel('Avg Round Time (seconds)')
        ax2.set_title('Training Time by Topology and Paradigm')
        ax2.legend(title='Learning Paradigm')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Efficiency Score Distribution by Topology
        ax3 = axes[1, 0]
        topology_data = [self.df[self.df['topology'] == topo]['avg_efficiency_score'].values 
                        for topo in self.df['topology'].unique()]
        ax3.boxplot(topology_data, labels=self.df['topology'].unique())
        ax3.set_xlabel('Network Topology')
        ax3.set_ylabel('Resource Efficiency Score')
        ax3.set_title('Efficiency Score Distribution by Topology')
        ax3.grid(True, alpha=0.3)
        
        # 4. Scalability Analysis (Time per Client)
        ax4 = axes[1, 1]
        for topology in self.df['topology'].unique():
            data = self.df[self.df['topology'] == topology]
            grouped = data.groupby('num_clients')['time_per_client'].mean()
            ax4.plot(grouped.index, grouped.values, marker='o', label=topology.title(), linewidth=2)
        
        ax4.set_xlabel('Number of Clients')
        ax4.set_ylabel('Time per Client (seconds)')
        ax4.set_title('Scalability: Time per Client vs Network Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'topology_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated topology comparison plot")
    
    def plot_privacy_impact(self):
        """Generate plots analyzing the impact of differential privacy"""
        # Filter data to include only experiments with DP variations
        dp_data = self.df[self.df['privacy_setting'].isin(['no_dp', 'high_privacy', 'medium_privacy', 'low_privacy'])]
        
        if len(dp_data) == 0:
            logger.warning("No differential privacy experiments found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Differential Privacy Impact on Resource Efficiency', fontsize=16, y=0.98)
        
        # 1. Training Time vs Privacy Level
        ax1 = axes[0, 0]
        privacy_order = ['no_dp', 'low_privacy', 'medium_privacy', 'high_privacy']
        privacy_time = dp_data.groupby('privacy_setting')['avg_round_time'].mean().reindex(privacy_order)
        bars1 = ax1.bar(range(len(privacy_time)), privacy_time.values, 
                       color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
        ax1.set_xticks(range(len(privacy_time)))
        ax1.set_xticklabels(['No DP', 'Low Privacy', 'Medium Privacy', 'High Privacy'], rotation=45)
        ax1.set_ylabel('Avg Round Time (seconds)')
        ax1.set_title('Training Time vs Privacy Level')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Communication vs Privacy Level 
        ax2 = axes[0, 1]
        privacy_comm = dp_data.groupby('privacy_setting')['communication_mb'].mean().reindex(privacy_order)
        bars2 = ax2.bar(range(len(privacy_comm)), privacy_comm.values,
                       color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
        ax2.set_xticks(range(len(privacy_comm)))
        ax2.set_xticklabels(['No DP', 'Low Privacy', 'Medium Privacy', 'High Privacy'], rotation=45)
        ax2.set_ylabel('Avg Communication (MB)')
        ax2.set_title('Communication vs Privacy Level')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Efficiency Score vs Privacy Level
        ax3 = axes[1, 0]
        privacy_eff = dp_data.groupby('privacy_setting')['avg_efficiency_score'].mean().reindex(privacy_order)
        bars3 = ax3.bar(range(len(privacy_eff)), privacy_eff.values,
                       color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
        ax3.set_xticks(range(len(privacy_eff)))
        ax3.set_xticklabels(['No DP', 'Low Privacy', 'Medium Privacy', 'High Privacy'], rotation=45)
        ax3.set_ylabel('Resource Efficiency Score')
        ax3.set_title('Efficiency vs Privacy Level')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Privacy-Utility-Resource Tradeoff (Sampling Rate Effect)
        ax4 = axes[1, 1]
        for privacy in ['no_dp', 'medium_privacy', 'high_privacy']:
            if privacy in dp_data['privacy_setting'].values:
                subset = dp_data[dp_data['privacy_setting'] == privacy]
                grouped = subset.groupby('sampling_rate')['avg_efficiency_score'].mean()
                ax4.plot(grouped.index, grouped.values, marker='o', 
                        label=privacy.replace('_', ' ').title(), linewidth=2)
        
        ax4.set_xlabel('Client Sampling Rate')
        ax4.set_ylabel('Resource Efficiency Score')
        ax4.set_title('Privacy-Sampling Rate Interaction')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'privacy_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated privacy impact analysis plot")
    
    def plot_scalability_analysis(self):
        """Generate plots analyzing scalability with number of clients"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scalability Analysis: Impact of Number of Clients', fontsize=16, y=0.98)
        
        # 1. Communication vs Number of Clients
        ax1 = axes[0, 0]
        for paradigm in self.df['paradigm'].unique():
            for topology in self.df['topology'].unique():
                subset = self.df[(self.df['paradigm'] == paradigm) & (self.df['topology'] == topology)]
                if len(subset) > 0:
                    grouped = subset.groupby('num_clients')['communication_mb'].mean()
                    ax1.plot(grouped.index, grouped.values, marker='o', 
                            label=f'{paradigm.title()}-{topology.title()}', linewidth=2)
        
        ax1.set_xlabel('Number of Clients')
        ax1.set_ylabel('Avg Communication (MB)')
        ax1.set_title('Communication vs Number of Clients')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Training Time vs Number of Clients
        ax2 = axes[0, 1]
        for sampling_rate in [0.2, 0.6, 1.0]:
            subset = self.df[self.df['sampling_rate'] == sampling_rate]
            grouped = subset.groupby('num_clients')['avg_round_time'].mean()
            ax2.plot(grouped.index, grouped.values, marker='s', 
                    label=f'Sampling Rate {sampling_rate}', linewidth=2)
        
        ax2.set_xlabel('Number of Clients')
        ax2.set_ylabel('Avg Round Time (seconds)')
        ax2.set_title('Training Time vs Number of Clients')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency vs Number of Clients
        ax3 = axes[1, 0]
        client_counts = sorted(self.df['num_clients'].unique())
        efficiency_data = [self.df[self.df['num_clients'] == count]['avg_efficiency_score'].values 
                          for count in client_counts]
        ax3.boxplot(efficiency_data, labels=client_counts)
        ax3.set_xlabel('Number of Clients')
        ax3.set_ylabel('Resource Efficiency Score')
        ax3.set_title('Efficiency Distribution vs Number of Clients')
        ax3.grid(True, alpha=0.3)
        
        # 4. Communication Savings vs Number of Clients
        ax4 = axes[1, 1]
        for sampling_rate in [0.2, 0.4, 0.6, 0.8]:
            subset = self.df[self.df['sampling_rate'] == sampling_rate]
            grouped = subset.groupby('num_clients')['communication_savings_percent'].mean()
            ax4.plot(grouped.index, grouped.values, marker='^', 
                    label=f'Sampling {sampling_rate}', linewidth=2)
        
        ax4.set_xlabel('Number of Clients')
        ax4.set_ylabel('Communication Savings (%)')
        ax4.set_title('Communication Savings vs Number of Clients')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated scalability analysis plot")
    
    def generate_summary_table(self):
        """Generate summary tables for the paper"""
        # Table 1: Resource Efficiency by Configuration
        summary_table = self.df.groupby(['paradigm', 'topology']).agg({
            'communication_mb': ['mean', 'std'],
            'avg_round_time': ['mean', 'std'],
            'avg_efficiency_score': ['mean', 'std'],
            'communication_savings_percent': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns.values]
        
        # Save to CSV
        summary_table.to_csv(self.plots_dir / 'resource_efficiency_summary.csv')
        
        # Table 2: Sampling Rate Impact
        sampling_table = self.df.groupby('sampling_rate').agg({
            'communication_savings_percent': 'mean',
            'avg_efficiency_score': 'mean',
            'avg_round_time': 'mean'
        }).round(3)
        
        sampling_table.to_csv(self.plots_dir / 'sampling_rate_impact.csv')
        
        # Table 3: Privacy Impact
        if 'privacy_setting' in self.df.columns:
            privacy_table = self.df.groupby('privacy_setting').agg({
                'avg_round_time': ['mean', 'std'],
                'avg_efficiency_score': ['mean', 'std'],
                'communication_mb': ['mean', 'std']
            }).round(3)
            
            privacy_table.columns = ['_'.join(col).strip() for col in privacy_table.columns.values]
            privacy_table.to_csv(self.plots_dir / 'privacy_impact_summary.csv')
        
        logger.info("Generated summary tables")
    
    def generate_all_plots(self):
        """Generate all analysis plots"""
        logger.info("Generating all analysis plots...")
        
        self.plot_sampling_rate_analysis()
        self.plot_topology_comparison()
        self.plot_privacy_impact()
        self.plot_scalability_analysis()
        self.generate_summary_table()
        
        logger.info(f"All plots and tables saved to: {self.plots_dir}")
        
        # Generate index file
        self.generate_plot_index()
    
    def generate_plot_index(self):
        """Generate an index file listing all generated plots"""
        plots = list(self.plots_dir.glob("*.png"))
        tables = list(self.plots_dir.glob("*.csv"))
        
        index_content = f"""# Resource-Aware Client Sampling Analysis Results

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Plots

"""
        
        for plot in sorted(plots):
            index_content += f"- [{plot.stem}]({plot.name})\n"
        
        index_content += "\n## Summary Tables\n\n"
        
        for table in sorted(tables):
            index_content += f"- [{table.stem}]({table.name})\n"
        
        index_content += f"""
## Experiment Summary

- Total experiments analyzed: {len(self.df)}
- Datasets: {', '.join(self.df['dataset'].unique())}
- Learning paradigms: {', '.join(self.df['paradigm'].unique())}
- Network topologies: {', '.join(self.df['topology'].unique())}
- Client counts: {', '.join(map(str, sorted(self.df['num_clients'].unique())))}
- Sampling rates: {', '.join(map(str, sorted(self.df['sampling_rate'].unique())))}

## Key Findings

1. **Sampling Rate Impact**: Communication savings range from {self.df['communication_savings_percent'].min():.1f}% to {self.df['communication_savings_percent'].max():.1f}%
2. **Topology Efficiency**: {self.df.loc[self.df['avg_efficiency_score'].idxmax(), 'topology'].title()} topology shows highest average efficiency
3. **Privacy Trade-offs**: Differential privacy impacts training time by up to {((self.df[self.df['privacy_setting'] == 'high_privacy']['avg_round_time'].mean() / self.df[self.df['privacy_setting'] == 'no_dp']['avg_round_time'].mean() - 1) * 100):.1f}%
"""
        
        with open(self.plots_dir / "README.md", "w") as f:
            f.write(index_content)
        
        logger.info("Generated analysis index file")


def main():
    parser = argparse.ArgumentParser(description="Analyze Resource-Aware Client Sampling Experiments")
    parser.add_argument("experiment_dir", help="Directory containing experiment results")
    parser.add_argument("--plots-only", action="store_true", 
                       help="Generate only plots (skip summary tables)")
    
    args = parser.parse_args()
    
    # Check if experiment directory exists
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return
    
    # Create analyzer
    analyzer = ResourceExperimentAnalyzer(args.experiment_dir)
    
    if len(analyzer.df) == 0:
        print("No successful experiments found for analysis")
        return
    
    # Generate analysis
    if args.plots_only:
        analyzer.plot_sampling_rate_analysis()
        analyzer.plot_topology_comparison()
        analyzer.plot_privacy_impact()
        analyzer.plot_scalability_analysis()
    else:
        analyzer.generate_all_plots()
    
    print(f"✅ Analysis complete! Results saved to: {analyzer.plots_dir}")


if __name__ == "__main__":
    main()