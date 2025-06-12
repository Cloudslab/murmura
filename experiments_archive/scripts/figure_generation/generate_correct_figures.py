#!/usr/bin/env python3
"""
Generate correct figures for the topology privacy leakage paper:
1. fig4_dataset_violin.pdf - Dataset vulnerability distributions with 3 attacks for each dataset
2. fig3_subsampling_flow.pdf - Complete subsampling impact assessment
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def load_experimental_data():
    """Load Phase 1 and Phase 2 experimental data."""
    
    # Load Phase 1 (baseline, no subsampling)
    with open('results_phase1/rerun_attack_results.json', 'r') as f:
        phase1_data = json.load(f)
    
    # Load Phase 2 (subsampling experiments)  
    with open('results_phase2/rerun_attack_results.json', 'r') as f:
        phase2_data = json.load(f)
    
    return phase1_data, phase2_data

def extract_attack_success_by_strategy(experiments):
    """Extract attack success metrics by attack strategy and dataset."""
    
    data = []
    for exp in experiments:
        config = exp['config']
        
        # Get all attack results
        if 'attack_results' in exp and 'attack_results' in exp['attack_results']:
            attack_results = exp['attack_results']['attack_results']
            
            for attack in attack_results:
                if 'attack_success_metric' in attack and 'attack_name' in attack:
                    # Map attack names to standardized strategy names
                    attack_name = attack['attack_name'].lower()
                    if 'communication' in attack_name or 'pattern' in attack_name:
                        strategy = 'communication_pattern'
                    elif 'magnitude' in attack_name or 'parameter' in attack_name:
                        strategy = 'parameter_magnitude'
                    elif 'topology' in attack_name or 'structure' in attack_name:
                        strategy = 'topology_structure'
                    else:
                        continue  # Skip unknown attack types
                    
                    data.append({
                        'dataset': config['dataset'].lower(),
                        'attack_strategy': strategy,
                        'attack_success': attack['attack_success_metric'],
                        'experiment_name': exp['experiment_name']
                    })
    
    return pd.DataFrame(data)

def extract_subsampling_data(experiments):
    """Extract subsampling progression data."""
    
    data = []
    for exp in experiments:
        config = exp['config']
        
        # Extract subsampling level from experiment name - more comprehensive patterns
        exp_name = exp['experiment_name'].lower()
        
        # Check for subsampling patterns
        if 'baseline' in exp_name or 'no_subsampling' in exp_name or 'no_dp' in exp_name:
            subsampling_level = 'baseline'
        elif 'moderate' in exp_name or '50' in exp_name and '80' in exp_name:
            subsampling_level = 'moderate'
        elif 'strong' in exp_name and 'very' not in exp_name:
            if '30' in exp_name and '60' in exp_name:
                subsampling_level = 'strong'
            else:
                subsampling_level = 'strong'  # Assume strong if just "strong" mentioned
        elif 'very_strong' in exp_name or 'very strong' in exp_name or ('20' in exp_name and '50' in exp_name):
            subsampling_level = 'very_strong'
        else:
            # Print unknown patterns for debugging
            print(f"Unknown subsampling pattern: {exp_name}")
            subsampling_level = 'unknown'
        
        # Get all attack results
        if 'attack_results' in exp and 'attack_results' in exp['attack_results']:
            attack_results = exp['attack_results']['attack_results']
            
            for attack in attack_results:
                if 'attack_success_metric' in attack:
                    data.append({
                        'dataset': config['dataset'].lower(),
                        'subsampling_level': subsampling_level,
                        'attack_success': attack['attack_success_metric'],
                        'experiment_name': exp['experiment_name']
                    })
    
    return pd.DataFrame(data)

def generate_dataset_violin_plot(phase1_data, phase2_data, output_dir):
    """Generate Figure 4: Dataset vulnerability distributions with 3 attacks for each dataset."""
    
    print("Generating Figure 4: Dataset vulnerability distributions...")
    
    # Extract attack strategy data from both phases
    phase1_df = extract_attack_success_by_strategy(phase1_data)
    phase2_df = extract_attack_success_by_strategy(phase2_data)
    
    # Combine data
    combined_df = pd.concat([phase1_df, phase2_df], ignore_index=True)
    
    if len(combined_df) == 0:
        print("No attack strategy data found!")
        return
    
    print(f"Found {len(combined_df)} attack results")
    print(f"Datasets: {combined_df['dataset'].unique()}")
    print(f"Attack strategies: {combined_df['attack_strategy'].unique()}")
    
    # Debug: Show data range for each attack type
    for strategy in combined_df['attack_strategy'].unique():
        strategy_data = combined_df[combined_df['attack_strategy'] == strategy]['attack_success']
        print(f"{strategy}: min={strategy_data.min():.3f}, max={strategy_data.max():.3f}, mean={strategy_data.mean():.3f}")
        print(f"  Zero values: {(strategy_data == 0).sum()} out of {len(strategy_data)}")
        print(f"  Range: {strategy_data.min():.3f} to {strategy_data.max():.3f}")
    
    # Create the violin plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define attack strategies and datasets
    attack_strategies = ['communication_pattern', 'parameter_magnitude', 'topology_structure']
    strategy_labels = ['Communication\nPattern', 'Parameter\nMagnitude', 'Topology\nStructure']
    datasets = ['mnist', 'ham10000']
    dataset_labels = ['MNIST', 'HAM10000']
    
    # Colors for different attacks
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Prepare data for violin plots
    violin_data = []
    positions = []
    labels = []
    colors_list = []
    
    pos = 1
    for i, dataset in enumerate(datasets):
        for j, strategy in enumerate(attack_strategies):
            # Get data for this combination
            data = combined_df[(combined_df['dataset'] == dataset) & 
                             (combined_df['attack_strategy'] == strategy)]['attack_success'].values
            
            if len(data) > 0:
                violin_data.append(data)
                positions.append(pos)
                labels.append(f"{strategy_labels[j]}\n({dataset_labels[i]})")
                colors_list.append(colors[j])
                pos += 1
        
        # Add space between datasets
        if i < len(datasets) - 1:
            pos += 0.5
    
    if len(violin_data) == 0:
        print("No data to plot!")
        return
    
    # Create violin plots
    violin_parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True, widths=0.6)
    
    # Color the violin plots
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Style the other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in violin_parts:
            violin_parts[partname].set_color('black')
            violin_parts[partname].set_linewidth(1)
    
    # Customize the plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Attack Success Rate', fontsize=14)
    ax.set_title('Dataset Vulnerability Distributions by Attack Type', fontsize=16)
    ax.set_ylim(0, 1.05)  # Y-axis from 0
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add dataset separator line (shifted left)
    if len(datasets) > 1:
        separator_pos = positions[len(attack_strategies)] - 0.5
        ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'fig4_dataset_violin.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    print(f"Saved Figure 4 to {output_path}")
    plt.close()

def generate_subsampling_flow_chart(phase1_data, phase2_data, output_dir):
    """Generate Figure 3: Subsampling impact assessment with simple bar chart."""
    
    print("Generating Figure 3: Subsampling impact assessment...")
    
    # Phase 1 is baseline (no subsampling)
    phase1_df = extract_subsampling_data(phase1_data)
    phase1_df['subsampling_level'] = 'baseline'  # Force all Phase 1 to be baseline
    
    # Phase 2 has subsampling - let's check what levels exist
    phase2_df = extract_subsampling_data(phase2_data)
    
    # Combine data
    combined_df = pd.concat([phase1_df, phase2_df], ignore_index=True)
    
    if len(combined_df) == 0:
        print("No subsampling data found!")
        return
    
    print(f"Found {len(combined_df)} subsampling results")
    print(f"Subsampling levels: {combined_df['subsampling_level'].unique()}")
    
    # Define expected subsampling configurations in order
    expected_configs = [
        {'level': 'baseline', 'label': 'No Subsampling', 'order': 0},
        {'level': 'moderate', 'label': 'Moderate\n(50% clients, 80% data)', 'order': 1},
        {'level': 'strong', 'label': 'Strong\n(30% clients, 60% data)', 'order': 2},
        {'level': 'very_strong', 'label': 'Very Strong\n(20% clients, 50% data)', 'order': 3}
    ]
    
    # Calculate statistics for each available level
    stats = []
    for config in expected_configs:
        level_data = combined_df[combined_df['subsampling_level'] == config['level']]
        if len(level_data) > 0:
            stats.append({
                'level': config['level'],
                'label': config['label'],
                'order': config['order'],
                'mean': level_data['attack_success'].mean(),
                'std': level_data['attack_success'].std(),
                'count': len(level_data)
            })
            print(f"Found {len(level_data)} experiments for {config['level']}")
        else:
            print(f"No data found for {config['level']}")
    
    if len(stats) == 0:
        print("No valid subsampling data to plot!")
        return
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats).sort_values('order')
    
    # Print detailed analysis
    print(f"\nSubsampling Results Analysis:")
    for _, row in stats_df.iterrows():
        print(f"{row['label']}: {row['mean']:.3f} ± {row['std']:.3f} ({row['count']} experiments)")
        if row['level'] != 'baseline':
            baseline_mean = stats_df[stats_df['level'] == 'baseline']['mean'].iloc[0]
            change = ((row['mean'] - baseline_mean) / baseline_mean) * 100
            print(f"  Change from baseline: {change:+.1f}%")
    
    # Create line chart (like original figure)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create line chart with error bars
    ax.errorbar(range(len(stats_df)), stats_df['mean'], yerr=stats_df['std'], 
                marker='o', linewidth=3, markersize=10, capsize=8, capthick=2,
                color='#e74c3c', markerfacecolor='#c0392b', markeredgecolor='white', 
                markeredgewidth=2)
    
    # Fill area under curve
    ax.fill_between(range(len(stats_df)), 
                    stats_df['mean'] - stats_df['std'], 
                    stats_df['mean'] + stats_df['std'], 
                    alpha=0.3, color='#e74c3c')
    
    # Customize the plot
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df['label'], fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=14)
    ax.set_title('Privacy Amplification Through Subsampling', fontsize=16)
    ax.set_ylim(0, 1.0)  # Y-axis from 0
    ax.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, row in enumerate(stats_df.itertuples()):
        ax.annotate(f'{row.mean:.3f}', 
                   (i, row.mean), 
                   textcoords="offset points", 
                   xytext=(0,15), 
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add percentage change from baseline (if not baseline)
        if row.level != 'baseline' and len(stats_df) > 0:
            baseline_mean = stats_df[stats_df['level'] == 'baseline']['mean']
            if len(baseline_mean) > 0:
                baseline_val = baseline_mean.iloc[0]
                change = ((row.mean - baseline_val) / baseline_val) * 100
                ax.annotate(f'{change:+.1f}%', 
                           (i, row.mean), 
                           textcoords="offset points", 
                           xytext=(0,-25), 
                           ha='center', fontsize=10, 
                           color='red' if change > 0 else 'green',
                           fontweight='bold')
    
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
    
    print(f"Phase 1: {len(phase1_data)} experiments")
    print(f"Phase 2: {len(phase2_data)} experiments")
    
    # Set output directory
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Generate figures
    generate_dataset_violin_plot(phase1_data, phase2_data, output_dir)
    generate_subsampling_flow_chart(phase1_data, phase2_data, output_dir)
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()