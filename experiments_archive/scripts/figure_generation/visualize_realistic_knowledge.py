#!/usr/bin/env python3
"""
Generate visualizations for realistic partial topology knowledge scenarios.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any


def load_results(results_file: str) -> Dict[str, Any]:
    """Load realistic scenario analysis results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_scenario_comparison_plot(summary: Dict[str, Any], output_dir: str) -> None:
    """Create comprehensive comparison of all scenarios."""
    
    # Set up the data
    scenarios = [
        "Complete Knowledge (Baseline)",
        "Neighborhood 1-hop",
        "Neighborhood 2-hop", 
        "Statistical Knowledge",
        "Organizational 3-groups",
        "Organizational 5-groups"
    ]
    
    attacks = ["Communication Pattern Attack", "Parameter Magnitude Attack", "Topology Structure Attack"]
    attack_labels = ["Communication\nPattern", "Parameter\nMagnitude", "Topology\nStructure"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    colors = ['#2E8B57', '#4169E1', '#FF6347', '#FFD700', '#9370DB', '#20B2AA']
    
    for attack_idx, (attack, attack_label) in enumerate(zip(attacks, attack_labels)):
        ax = axes[attack_idx]
        
        success_rates = []
        scenario_labels = []
        scenario_colors = []
        
        for i, scenario in enumerate(scenarios):
            if scenario in summary["scenario_effectiveness"]:
                if attack in summary["scenario_effectiveness"][scenario]:
                    success = summary["scenario_effectiveness"][scenario][attack]["average_success"]
                    success_rates.append(success)
                    scenario_labels.append(scenario.replace(" (Baseline)", "").replace("Neighborhood ", "").replace("Organizational ", "Org "))
                    scenario_colors.append(colors[i])
        
        # Create bar plot
        bars = ax.bar(range(len(success_rates)), success_rates, color=scenario_colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=14)
        
        # Formatting
        ax.set_title(attack_label, fontsize=20)
        ax.set_ylabel("Attack Success Rate" if attack_idx == 0 else "", fontsize=18)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(scenario_labels)))
        ax.set_xticklabels(scenario_labels, rotation=45, ha='right', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/realistic_scenarios_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/realistic_scenarios_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_degradation_heatmap(summary: Dict[str, Any], output_dir: str) -> None:
    """Create heatmap showing performance degradation from baseline."""
    
    scenarios = [
        "Neighborhood 1-hop",
        "Neighborhood 2-hop", 
        "Statistical Knowledge",
        "Organizational 3-groups",
        "Organizational 5-groups"
    ]
    
    attacks = ["Communication Pattern Attack", "Parameter Magnitude Attack", "Topology Structure Attack"]
    attack_labels = ["Communication", "Parameter", "Topology"]
    
    # Build degradation matrix
    degradation_matrix = []
    
    for scenario in scenarios:
        row = []
        for attack in attacks:
            if (scenario in summary["scenario_effectiveness"] and 
                attack in summary["scenario_effectiveness"][scenario]):
                reduction = summary["scenario_effectiveness"][scenario][attack]["reduction_from_baseline"]
                row.append(reduction)
            else:
                row.append(0)
        degradation_matrix.append(row)
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    
    # Use diverging colormap
    sns.heatmap(degradation_matrix,
                xticklabels=attack_labels,
                yticklabels=[s.replace("Neighborhood ", "").replace("Organizational ", "Org ") for s in scenarios],
                annot=True,
                fmt='.1f',
                cmap='RdBu',
                center=0,
                cbar_kws={'label': 'Reduction from Baseline (%)'},
                vmin=-100,
                vmax=100)
    
    plt.title("Attack Success Reduction with Partial Knowledge", fontsize=20)
    plt.xlabel("Attack Type", fontsize=18)
    plt.ylabel("Knowledge Scenario", fontsize=18)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degradation_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/degradation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_paper_figure(summary: Dict[str, Any], output_dir: str) -> None:
    """Create publication-ready figure for the paper."""
    
    # Set style for publication
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(12, 8))
    
    # Create 2x2 subplot layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Scenario comparison (top, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    
    scenarios = ["Complete Knowledge (Baseline)", "Neighborhood 1-hop", "Neighborhood 2-hop", 
                "Statistical Knowledge", "Organizational 3-groups"]
    scenario_labels = ["Complete", "1-hop", "2-hop", "Statistical", "Org-3"]
    
    attacks = ["Communication Pattern Attack", "Parameter Magnitude Attack", "Topology Structure Attack"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    x = np.arange(len(scenario_labels))
    width = 0.25
    
    for i, attack in enumerate(attacks):
        success_rates = []
        for scenario in scenarios:
            if (scenario in summary["scenario_effectiveness"] and 
                attack in summary["scenario_effectiveness"][scenario]):
                success_rates.append(summary["scenario_effectiveness"][scenario][attack]["average_success"])
            else:
                success_rates.append(0)
        
        ax1.bar(x + i*width, success_rates, width, 
               label=attack.replace(" Attack", ""), color=colors[i], alpha=0.8)
    
    ax1.set_xlabel("Knowledge Scenario", fontsize=18)
    ax1.set_ylabel("Attack Success Rate", fontsize=18)
    ax1.set_title("(a) Attack Success Across Knowledge Scenarios", fontsize=20)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenario_labels)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Neighborhood knowledge effectiveness (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    
    neighborhood_scenarios = ["Complete Knowledge (Baseline)", "Neighborhood 1-hop", "Neighborhood 2-hop"]
    neighborhood_labels = ["Complete", "1-hop", "2-hop"]
    
    # Focus on Communication Pattern Attack for neighborhood
    comm_success = []
    for scenario in neighborhood_scenarios:
        if (scenario in summary["scenario_effectiveness"] and 
            "Communication Pattern Attack" in summary["scenario_effectiveness"][scenario]):
            comm_success.append(summary["scenario_effectiveness"][scenario]["Communication Pattern Attack"]["average_success"])
        else:
            comm_success.append(0)
    
    bars = ax2.bar(neighborhood_labels, comm_success, color='#1f77b4', alpha=0.8)
    for bar, val in zip(bars, comm_success):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=14)
    
    ax2.set_ylabel("Success Rate", fontsize=18)
    ax2.set_title("(b) Neighborhood Knowledge\n(Communication Attack)", fontsize=18)
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Organizational knowledge (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    org_scenarios = ["Complete Knowledge (Baseline)", "Organizational 3-groups", "Organizational 5-groups"]
    org_labels = ["Complete", "3-groups", "5-groups"]
    
    # Show all attacks for organizational scenarios
    for i, attack in enumerate(attacks):
        org_success = []
        for scenario in org_scenarios:
            if (scenario in summary["scenario_effectiveness"] and 
                attack in summary["scenario_effectiveness"][scenario]):
                org_success.append(summary["scenario_effectiveness"][scenario][attack]["average_success"])
            else:
                org_success.append(0)
        
        x_pos = np.arange(len(org_labels)) + i*0.25
        ax3.bar(x_pos, org_success, 0.25, 
               label=attack.replace(" Attack", ""), color=colors[i], alpha=0.8)
    
    ax3.set_ylabel("Success Rate", fontsize=18)
    ax3.set_title("(c) Organizational Knowledge", fontsize=18)
    ax3.set_xticks(np.arange(len(org_labels)) + 0.25)
    ax3.set_xticklabels(org_labels)
    ax3.legend(fontsize=14)
    ax3.set_ylim(0, 1.0)
    ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f"{output_dir}/fig_realistic_knowledge_paper.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig_realistic_knowledge_paper.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Generate all visualizations."""
    
    # Paths
    results_dir = "experiments_archive/results/attack_results/realistic_knowledge_full_analysis"
    summary_file = f"{results_dir}/realistic_scenario_summary.json"
    output_dir = "experiments_archive/figures/realistic_knowledge_analysis"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    if not Path(summary_file).exists():
        print(f"Error: Summary file not found at {summary_file}")
        print("Please run the realistic knowledge analysis first.")
        return
    
    summary = load_results(summary_file)
    
    # Generate plots
    print("Generating scenario comparison plot...")
    create_scenario_comparison_plot(summary, output_dir)
    
    print("Generating degradation heatmap...")
    create_degradation_heatmap(summary, output_dir)
    
    print("Generating paper figure...")
    create_paper_figure(summary, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")
    
    # Print paper-ready summary
    print("\nSummary for Paper:")
    print("-" * 60)
    print("Realistic Partial Knowledge Scenarios:")
    
    baseline_success = {}
    if "Complete Knowledge (Baseline)" in summary["scenario_effectiveness"]:
        for attack, data in summary["scenario_effectiveness"]["Complete Knowledge (Baseline)"].items():
            baseline_success[attack] = data["average_success"]
    
    key_scenarios = [
        ("Neighborhood 1-hop", "Local adversary (1-hop neighborhood)"),
        ("Statistical Knowledge", "External adversary (topology type only)"),
        ("Organizational 3-groups", "Insider knowledge (group structure)")
    ]
    
    for scenario, description in key_scenarios:
        print(f"\n{description}:")
        if scenario in summary["scenario_effectiveness"]:
            for attack in ["Communication Pattern Attack", "Parameter Magnitude Attack", "Topology Structure Attack"]:
                if attack in summary["scenario_effectiveness"][scenario]:
                    success = summary["scenario_effectiveness"][scenario][attack]["average_success"]
                    baseline = baseline_success.get(attack, 0)
                    reduction = ((baseline - success) / baseline * 100) if baseline > 0 else 0
                    effective = "✓" if success > 0.3 else "✗"
                    
                    print(f"  {attack.replace(' Attack', '')}: {success:.2f} ({effective}) (-{reduction:.0f}%)")


if __name__ == "__main__":
    main()