#!/usr/bin/env python3
"""
Re-run topology attacks on existing visualization data with improved metrics.
This script processes existing experiment visualization data without re-running the expensive training.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from murmura.attacks.topology_attacks import run_topology_attacks


def load_existing_results(results_file: str) -> List[Dict[str, Any]]:
    """Load existing experiment results."""
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return []


def extract_experiment_config_from_name(exp_name: str) -> Dict[str, Any]:
    """Extract experiment configuration from experiment name."""
    # Example: exp_0001_mnist_federated_star_5n_no_dp
    parts = exp_name.split('_')
    
    try:
        exp_id = int(parts[1])
        dataset = parts[2]
        fl_type = parts[3]
        topology = parts[4]
        node_count = int(parts[5].replace('n', ''))
        
        # Parse DP setting
        dp_parts = parts[6:]
        if 'no' in dp_parts and 'dp' in dp_parts:
            dp_setting = {"enabled": False, "epsilon": None, "name": "no_dp"}
        elif 'weak' in dp_parts:
            dp_setting = {"enabled": True, "epsilon": 16.0, "name": "weak_dp"}
        elif 'medium' in dp_parts:
            dp_setting = {"enabled": True, "epsilon": 8.0, "name": "medium_dp"}
        elif 'strong' in dp_parts:
            dp_setting = {"enabled": True, "epsilon": 4.0, "name": "strong_dp"}
        elif 'very' in dp_parts and 'strong' in dp_parts:
            dp_setting = {"enabled": True, "epsilon": 1.0, "name": "very_strong_dp"}
        else:
            dp_setting = {"enabled": False, "epsilon": None, "name": "unknown"}
        
        # Determine attack strategy from experiment range
        if exp_id <= 105:
            attack_strategy = "sensitive_groups"
        elif exp_id <= 200:
            attack_strategy = "topology_correlated"
        else:
            attack_strategy = "imbalanced_sensitive"
        
        return {
            "config_id": exp_id,
            "dataset": dataset,
            "attack_strategy": attack_strategy,
            "fl_type": fl_type,
            "topology": topology,
            "node_count": node_count,
            "dp_setting": dp_setting,
            "expected_runtime": 60  # Default value
        }
    
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse experiment name {exp_name}: {e}")
        return {
            "config_id": 0,
            "dataset": "unknown",
            "attack_strategy": "unknown",
            "fl_type": "unknown",
            "topology": "unknown",
            "node_count": 5,
            "dp_setting": {"enabled": False, "epsilon": None, "name": "unknown"},
            "expected_runtime": 60
        }


def run_attacks_on_visualization_data(viz_dir: str, exp_name: str) -> Dict[str, Any]:
    """Run topology attacks on visualization data from a specific experiment."""
    try:
        print(f"Processing {exp_name}...")
        
        # Check if visualization directory exists
        if not os.path.exists(viz_dir):
            return {
                "error": f"Visualization directory not found: {viz_dir}",
                "attack_results": [],
                "status": "failed"
            }
        
        # Run attacks using the existing function
        attack_results = run_topology_attacks(viz_dir)
        
        # Extract just the attack results array
        if 'attack_results' in attack_results:
            return {
                "attack_results": attack_results['attack_results'],
                "status": "success"
            }
        else:
            return {
                "attack_results": [],
                "status": "failed",
                "error": "No attack results returned"
            }
    
    except Exception as e:
        print(f"Error processing {exp_name}: {e}")
        return {
            "attack_results": [],
            "status": "failed",
            "error": str(e)
        }


def rerun_all_attacks(
    visualization_base_dir: str = "paper_experiments/visualizations",
    results_file: str = "paper_experiments/results/rerun_attack_results.json",
    experiment_filter: Optional[List[str]] = None
) -> None:
    """
    Re-run attacks on all existing visualization data.
    
    Args:
        visualization_base_dir: Base directory containing experiment visualization folders
        results_file: Output file for updated results
        experiment_filter: Optional list of experiment names to process (process all if None)
    """
    
    print("Starting attack re-run on existing visualization data...")
    
    # Get list of experiment directories
    viz_base_path = Path(visualization_base_dir)
    if not viz_base_path.exists():
        print(f"Error: Visualization base directory not found: {visualization_base_dir}")
        return
    
    experiment_dirs = [d for d in viz_base_path.iterdir() if d.is_dir() and d.name.startswith('exp_')]
    experiment_dirs.sort(key=lambda x: int(x.name.split('_')[1]))  # Sort by experiment number
    
    # Apply filter if provided
    if experiment_filter:
        experiment_dirs = [d for d in experiment_dirs if d.name in experiment_filter]
    
    print(f"Found {len(experiment_dirs)} experiments to process")
    
    # Process each experiment
    updated_results = []
    failed_experiments = []
    
    for i, exp_dir in enumerate(experiment_dirs):
        exp_name = exp_dir.name
        viz_dir = str(exp_dir)
        
        print(f"[{i+1}/{len(experiment_dirs)}] Processing {exp_name}")
        
        # Extract config from experiment name
        config = extract_experiment_config_from_name(exp_name)
        
        # Run attacks
        start_time = time.time()
        attack_result = run_attacks_on_visualization_data(viz_dir, exp_name)
        runtime = time.time() - start_time
        
        # Create result entry
        result_entry = {
            "experiment_id": config["config_id"],
            "experiment_name": exp_name,
            "config": config,
            "status": attack_result["status"],
            "runtime_seconds": runtime,
            "attack_results": attack_result
        }
        
        # Add error info if failed
        if "error" in attack_result:
            result_entry["error"] = attack_result["error"]
            failed_experiments.append(exp_name)
        
        updated_results.append(result_entry)
        
        # Save intermediate results every 10 experiments
        if (i + 1) % 10 == 0:
            print(f"Saving intermediate results after {i+1} experiments...")
            save_results(updated_results, results_file)
    
    # Save final results
    print(f"\nSaving final results to {results_file}")
    save_results(updated_results, results_file)
    
    # Print summary
    successful = len([r for r in updated_results if r["status"] == "success"])
    print(f"\nSummary:")
    print(f"  Total experiments: {len(updated_results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"  Failed experiments: {failed_experiments[:5]}{'...' if len(failed_experiments) > 5 else ''}")


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results to JSON file."""
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def compare_old_vs_new_metrics(
    old_results_file: str = "paper_experiments/results/intermediate_results.json",
    new_results_file: str = "paper_experiments/results/rerun_attack_results.json"
) -> None:
    """Compare old vs new attack success metrics."""
    
    print("Comparing old vs new metrics...")
    
    # Load both result sets
    with open(old_results_file, 'r') as f:
        old_results = json.load(f)
    
    with open(new_results_file, 'r') as f:
        new_results = json.load(f)
    
    # Create comparison
    comparisons = []
    
    for old_result in old_results:
        exp_name = old_result["experiment_name"]
        
        # Find corresponding new result
        new_result = next((r for r in new_results if r["experiment_name"] == exp_name), None)
        
        if new_result and new_result["status"] == "success":
            # Extract parameter magnitude attack metrics
            old_mag_attack = None
            new_mag_attack = None
            
            for attack in old_result["attack_results"]["attack_results"]:
                if attack["attack_name"] == "Parameter Magnitude Attack":
                    old_mag_attack = attack["attack_success_metric"]
                    break
            
            for attack in new_result["attack_results"]["attack_results"]:
                if attack["attack_name"] == "Parameter Magnitude Attack":
                    new_mag_attack = attack["attack_success_metric"]
                    break
            
            if old_mag_attack is not None and new_mag_attack is not None:
                comparisons.append({
                    "experiment": exp_name,
                    "old_metric": old_mag_attack,
                    "new_metric": new_mag_attack,
                    "improvement": new_mag_attack - old_mag_attack,
                    "improvement_ratio": new_mag_attack / old_mag_attack if old_mag_attack > 0 else float('inf')
                })
    
    # Print comparison summary
    if comparisons:
        improvements = [c["improvement"] for c in comparisons]
        ratios = [c["improvement_ratio"] for c in comparisons if c["improvement_ratio"] != float('inf')]
        
        print(f"\nMetric Comparison Summary:")
        print(f"  Experiments compared: {len(comparisons)}")
        print(f"  Average improvement: {sum(improvements)/len(improvements):.6f}")
        print(f"  Average improvement ratio: {sum(ratios)/len(ratios):.2f}x" if ratios else "N/A")
        
        # Show top improvements
        top_improvements = sorted(comparisons, key=lambda x: x["improvement"], reverse=True)[:5]
        print(f"\nTop 5 improvements:")
        for comp in top_improvements:
            print(f"    {comp['experiment']}: {comp['old_metric']:.6f} → {comp['new_metric']:.6f} ({comp['improvement']:.6f})")


if __name__ == "__main__":
    import sys
    
    # Command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Just run comparison
            compare_old_vs_new_metrics()
        elif sys.argv[1] == "--filter":
            # Run specific experiments
            experiment_names = sys.argv[2:]
            rerun_all_attacks(experiment_filter=experiment_names)
        else:
            print("Usage:")
            print("  python rerun_attacks.py                    # Process all experiments")
            print("  python rerun_attacks.py --filter exp_001 exp_002  # Process specific experiments")
            print("  python rerun_attacks.py --compare         # Compare old vs new metrics")
    else:
        # Run all experiments
        rerun_all_attacks()
        
        # Also run comparison if old results exist
        old_results_file = "paper_experiments/results/intermediate_results.json"
        if os.path.exists(old_results_file):
            compare_old_vs_new_metrics()