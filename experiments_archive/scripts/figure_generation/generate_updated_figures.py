#!/usr/bin/env python3
"""
Generate updated figures for the topology privacy leakage paper:
1. fig4_dataset_violin.pdf - Dataset vulnerability distributions with y-axis from 0
2. fig3_subsampling_flow.pdf - Complete subsampling impact assessment
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any

# Set up matplotlib for high-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_experimental_data() -> Tuple[List[Dict], List[Dict]]:
    """Load Phase 1 and Phase 2 experimental data."""
    
    # Load Phase 1 (baseline, no subsampling)
    with open('results_phase1/rerun_attack_results.json', 'r') as f:
        phase1_data = json.load(f)
    
    # Load Phase 2 (subsampling experiments)
    with open('results_phase2/rerun_attack_results.json', 'r') as f:
        phase2_data = json.load(f)
    
    return phase1_data, phase2_data

def extract_attack_success_data(experiments: List[Dict]) -> pd.DataFrame:
    """Extract attack success metrics from experimental data."""
    
    data = []
    for exp in experiments:
        config = exp['config']
        attack_results = exp['attack_results']['attack_results'][0]
        
        data.append({
            'experiment_name': exp['experiment_name'],
            'dataset': config['dataset'],
            'fl_type': config['fl_type'],
            'topology': config['topology'],
            'node_count': config['node_count'],
            'dp_enabled': config['dp_setting']['enabled'],
            'dp_epsilon': config['dp_setting']['epsilon'],
            'dp_name': config['dp_setting']['name'],
            'attack_success_metric': attack_results['attack_success_metric'],
            'attack_name': attack_results['attack_name']
        })
    
    return pd.DataFrame(data)

def extract_subsampling_level(experiment_name: str) -> str:
    """Extract subsampling level from experiment name."""
    if 'baseline' in experiment_name or ('no_dp' in experiment_name and 'sampling' not in experiment_name):
        return 'baseline'
    elif 'moderate_sampling' in experiment_name:
        return 'moderate'
    elif 'strong_sampling' in experiment_name:
        return 'strong'
    elif 'very_strong_sampling' in experiment_name:
        return 'very_strong'
    else:
        # For debugging
        print(f"Unknown subsampling pattern in: {experiment_name}")
        return 'unknown'

def generate_dataset_violin_plot(phase1_df: pd.DataFrame, phase2_df: pd.DataFrame, 
                                output_dir: Path):
    """Generate Figure 4: Dataset vulnerability distributions with y-axis from 0."""
    
    # Combine datasets and add phase information
    phase1_df = phase1_df.copy()
    phase2_df = phase2_df.copy()
    phase1_df['phase'] = 'Phase 1 (Baseline)'
    phase2_df['phase'] = 'Phase 2 (Subsampling)'
    
    combined_df = pd.concat([phase1_df, phase2_df], ignore_index=True)
    
    # Create violin plot
    plt.figure(figsize=(12, 8))
    
    # Define color palette
    dataset_colors = {
        'mnist': '#E74C3C',      # Red
        'ham10000': '#3498DB'    # Blue
    }
    
    # Create subplots for each dataset
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    
    datasets = ['mnist', 'ham10000']
    dataset_labels = ['MNIST', 'HAM10000']
    
    for idx, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
        dataset_data = combined_df[combined_df['dataset'] == dataset]
        
        # Create violin plot
        violin_parts = axes[idx].violinplot([
            dataset_data[dataset_data['phase'] == 'Phase 1 (Baseline)']['attack_success_metric'].values,
            dataset_data[dataset_data['phase'] == 'Phase 2 (Subsampling)']['attack_success_metric'].values
        ], positions=[1, 2], widths=0.6, showmeans=True, showmedians=True)
        
        # Customize violin colors
        for pc in violin_parts['bodies']:
            pc.set_facecolor(dataset_colors[dataset])
            pc.set_alpha(0.7)
        
        # Set labels and title
        axes[idx].set_title(f'{label} Dataset\nVulnerability Distribution', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Experimental Phase', fontsize=12)
        if idx == 0:
            axes[idx].set_ylabel('Attack Success Rate', fontsize=12)
        
        # Set x-axis labels
        axes[idx].set_xticks([1, 2])
        axes[idx].set_xticklabels(['Baseline\n(No Subsampling)', 'Subsampling\nVariants'], fontsize=10)
        
        # Set y-axis to start from 0
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics
        baseline_mean = dataset_data[dataset_data['phase'] == 'Phase 1 (Baseline)']['attack_success_metric'].mean()
        subsampling_mean = dataset_data[dataset_data['phase'] == 'Phase 2 (Subsampling)']['attack_success_metric'].mean()
        
        axes[idx].text(0.02, 0.98, f'Baseline Mean: {baseline_mean:.3f}\nSubsampling Mean: {subsampling_mean:.3f}', 
                      transform=axes[idx].transAxes, fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Dataset Vulnerability Comparison:\nBaseline vs. Subsampling Impact', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'fig4_dataset_violin.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    print(f"Saved Figure 4 to {output_path}")
    plt.close()

def generate_subsampling_flow_chart(phase1_df: pd.DataFrame, phase2_df: pd.DataFrame, 
                                   output_dir: Path):
    """Generate Figure 3: Complete subsampling impact assessment."""
    
    # Extract subsampling levels for Phase 2
    phase2_df = phase2_df.copy()
    phase2_df['subsampling_level'] = phase2_df['experiment_name'].apply(extract_subsampling_level)
    
    # Calculate statistics for each subsampling level
    subsampling_stats = []
    
    # Baseline (Phase 1)
    baseline_mean = phase1_df['attack_success_metric'].mean()
    baseline_std = phase1_df['attack_success_metric'].std()
    subsampling_stats.append({
        'level': 'Baseline',
        'mean': baseline_mean,
        'std': baseline_std,
        'count': len(phase1_df),
        'order': 0
    })
    
    # Subsampling levels (Phase 2)
    sampling_order = {'moderate': 1, 'strong': 2, 'very_strong': 3}
    available_levels = phase2_df['subsampling_level'].unique()
    available_levels = [level for level in available_levels if level != 'unknown']
    
    for level in ['moderate', 'strong', 'very_strong']:
        if level in available_levels:
            level_data = phase2_df[phase2_df['subsampling_level'] == level]
            if len(level_data) > 0:
                subsampling_stats.append({
                    'level': level.replace('_', ' ').title(),
                    'mean': level_data['attack_success_metric'].mean(),
                    'std': level_data['attack_success_metric'].std(),
                    'count': len(level_data),
                    'order': sampling_order[level]
                })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(subsampling_stats).sort_values('order')
    
    # Create comprehensive flow chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Main progression line plot
    colors = ['#2C3E50', '#E74C3C', '#F39C12', '#8E44AD']
    ax1.plot(stats_df['order'], stats_df['mean'], 'o-', linewidth=3, markersize=10, 
             color='#3498DB', alpha=0.8)
    ax1.fill_between(stats_df['order'], 
                     stats_df['mean'] - stats_df['std'], 
                     stats_df['mean'] + stats_df['std'], 
                     alpha=0.3, color='#3498DB')
    
    ax1.set_xticks(stats_df['order'])
    ax1.set_xticklabels(stats_df['level'], rotation=45, ha='right')
    ax1.set_ylabel('Attack Success Rate', fontsize=12)
    ax1.set_title('Subsampling Impact Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # 2. Bar plot with error bars
    bars = ax2.bar(range(len(stats_df)), stats_df['mean'], 
                   yerr=stats_df['std'], capsize=5, alpha=0.8,
                   color=colors[:len(stats_df)])
    ax2.set_xticks(range(len(stats_df)))
    ax2.set_xticklabels(stats_df['level'], rotation=45, ha='right')
    ax2.set_ylabel('Attack Success Rate', fontsize=12)
    ax2.set_title('Subsampling Level Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, stats_df['mean']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Dataset-specific breakdown
    if len(phase2_df) > 0:
        dataset_breakdown = []
        for dataset in ['mnist', 'ham10000']:
            # Baseline
            data = phase1_df[phase1_df['dataset'] == dataset]
            if len(data) > 0:
                dataset_breakdown.append({
                    'dataset': dataset.upper(),
                    'level': 'Baseline',
                    'mean': data['attack_success_metric'].mean(),
                    'order': 0
                })
            
            # Available subsampling levels
            for level in available_levels:
                if level in sampling_order:
                    data = phase2_df[(phase2_df['dataset'] == dataset) & 
                                   (phase2_df['subsampling_level'] == level)]
                    if len(data) > 0:
                        dataset_breakdown.append({
                            'dataset': dataset.upper(),
                            'level': level.replace('_', ' ').title(),
                            'mean': data['attack_success_metric'].mean(),
                            'order': sampling_order[level]
                        })
        
        breakdown_df = pd.DataFrame(dataset_breakdown)
        
        # Plot dataset breakdown
        for dataset in ['MNIST', 'HAM10000']:
            dataset_data = breakdown_df[breakdown_df['dataset'] == dataset]
            ax3.plot(dataset_data['order'], dataset_data['mean'], 'o-', 
                    linewidth=2, markersize=8, label=dataset, alpha=0.8)
        
        # Set x-ticks based on available data
        x_labels = ['Baseline']
        if 'moderate' in available_levels:
            x_labels.append('Moderate')
        if 'strong' in available_levels:
            x_labels.append('Strong')
        if 'very_strong' in available_levels:
            x_labels.append('Very Strong')
            
        ax3.set_xticks(range(len(x_labels)))
        ax3.set_xticklabels(x_labels, rotation=45, ha='right')
        ax3.set_ylabel('Attack Success Rate', fontsize=12)
        ax3.set_title('Dataset-Specific Subsampling Impact', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.0)
    
    # 4. Statistical summary table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for _, row in stats_df.iterrows():
        reduction = ((stats_df.iloc[0]['mean'] - row['mean']) / stats_df.iloc[0]['mean'] * 100) if row['order'] > 0 else 0
        table_data.append([
            row['level'],
            f"{row['mean']:.3f} ± {row['std']:.3f}",
            f"{row['count']}",
            f"{reduction:.1f}%" if reduction > 0 else "Baseline"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Subsampling Level', 'Attack Success\n(Mean ± Std)', 'Experiments', 'Reduction from\nBaseline'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#3498DB')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
    
    ax4.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Subsampling Impact Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'fig3_subsampling_flow.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    print(f"Saved Figure 3 to {output_path}")
    plt.close()

def main():
    """Main function to generate both figures."""
    
    print("Loading experimental data...")
    phase1_data, phase2_data = load_experimental_data()
    
    print("Extracting attack success metrics...")
    phase1_df = extract_attack_success_data(phase1_data)
    phase2_df = extract_attack_success_data(phase2_data)
    
    print(f"Phase 1: {len(phase1_df)} experiments")
    print(f"Phase 2: {len(phase2_df)} experiments")
    
    # Set output directory
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating Figure 4: Dataset vulnerability distributions...")
    generate_dataset_violin_plot(phase1_df, phase2_df, output_dir)
    
    print("Generating Figure 3: Subsampling impact assessment...")
    generate_subsampling_flow_chart(phase1_df, phase2_df, output_dir)
    
    print("All figures generated successfully!")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Phase 1 (Baseline) - Attack Success Rate: {phase1_df['attack_success_metric'].mean():.3f} ± {phase1_df['attack_success_metric'].std():.3f}")
    print(f"Phase 2 (Subsampling) - Attack Success Rate: {phase2_df['attack_success_metric'].mean():.3f} ± {phase2_df['attack_success_metric'].std():.3f}")
    
    print("\nDataset breakdown:")
    for dataset in ['mnist', 'ham10000']:
        p1_dataset = phase1_df[phase1_df['dataset'] == dataset]['attack_success_metric']
        p2_dataset = phase2_df[phase2_df['dataset'] == dataset]['attack_success_metric']
        print(f"{dataset.upper()}:")
        print(f"  Phase 1: {p1_dataset.mean():.3f} ± {p1_dataset.std():.3f}")
        print(f"  Phase 2: {p2_dataset.mean():.3f} ± {p2_dataset.std():.3f}")

if __name__ == "__main__":
    main()