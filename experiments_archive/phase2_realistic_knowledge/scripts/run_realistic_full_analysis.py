#!/usr/bin/env python3
"""Run realistic scenarios on ALL experiments with >5 nodes in phase 1."""

from pathlib import Path
from rerun_attacks_realistic_partial_knowledge import run_experiments_with_realistic_scenarios

def main():
    """Run comprehensive realistic knowledge analysis on all eligible experiments."""
    
    print("Starting comprehensive realistic knowledge analysis...")
    print("Target: All experiments in phase 1 with >5 nodes")
    
    # First, let's see how many experiments we have
    viz_base = Path("experiments_archive/results/training_visualizations/visualizations_phase1")
    
    if not viz_base.exists():
        print(f"Error: Base directory not found: {viz_base}")
        return
    
    # Find all experiments with >5 nodes (including up to 30 nodes)
    all_experiments = []
    node_counts = {}
    
    for exp_dir in viz_base.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
            exp_name = exp_dir.name
            # Extract node count from name
            parts = exp_name.split('_')
            for part in parts:
                if part.endswith('n'):
                    try:
                        node_count = int(part[:-1])
                        if node_count > 5:
                            all_experiments.append(exp_name)
                            node_counts[node_count] = node_counts.get(node_count, 0) + 1
                        break
                    except ValueError:
                        continue
    
    print(f"Found {len(all_experiments)} experiments with >5 nodes")
    print("Node count distribution:")
    for nodes in sorted(node_counts.keys()):
        print(f"  {nodes} nodes: {node_counts[nodes]} experiments")
    print(f"This will generate approximately {len(all_experiments)} × 6 scenarios = {len(all_experiments) * 6} attack evaluations")
    
    # Run the full analysis
    run_experiments_with_realistic_scenarios(
        visualization_base_dir="experiments_archive/results/training_visualizations/visualizations_phase1",
        output_dir="experiments_archive/results/attack_results/realistic_knowledge_full_analysis",
        experiment_filter=None,  # Process all eligible experiments
        min_nodes=6  # Only experiments with >5 nodes
    )
    
    print("Comprehensive analysis complete!")
    print("Results saved to: experiments_archive/results/attack_results/realistic_knowledge_full_analysis/")

if __name__ == "__main__":
    main()