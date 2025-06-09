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
import numpy as np
from murmura.attacks.topology_attacks import run_topology_attacks


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def load_existing_results(results_file: str) -> List[Dict[str, Any]]:
    """Load existing experiment results_phase1."""
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
        
        # Extract just the attack results_phase1 array
        if 'attack_results' in attack_results:
            return {
                "attack_results": attack_results['attack_results'],
                "status": "success"
            }
        else:
            return {
                "attack_results": [],
                "status": "failed",
                "error": "No attack results_phase1 returned"
            }
    
    except Exception as e:
        print(f"Error processing {exp_name}: {e}")
        return {
            "attack_results": [],
            "status": "failed",
            "error": str(e)
        }


def rerun_all_attacks(
    visualization_base_dir: str = "paper_experiments/visualizations_phase1",
    results_file: str = "paper_experiments/results_phase1/rerun_attack_results.json",
    experiment_filter: Optional[List[str]] = None,
    experiment_phase: int = 1
) -> None:
    """
    Re-run attacks on all existing visualization data.
    
    Args:
        visualization_base_dir: Base directory containing experiment visualization folders
        results_file: Output file for updated results
        experiment_filter: Optional list of experiment names to process (process all if None)
        experiment_phase: Experimental phase (1 or 2) to determine directory structure
    """
    
    print(f"Starting attack re-run on existing Phase {experiment_phase} visualization data...")
    
    # Determine paths based on phase
    if experiment_phase == 2:
        if "phase1" in visualization_base_dir:
            visualization_base_dir = visualization_base_dir.replace("phase1", "phase2")
        if "phase1" in results_file:
            results_file = results_file.replace("phase1", "phase2")
    
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
        
        # Add phase-specific evaluation for phase 2
        if config.get("phase") == 2:
            result_entry["phase2_analysis"] = analyze_phase2_results(attack_result, config)
        
        updated_results.append(result_entry)
        
        # Save intermediate results_phase1 every 10 experiments
        if (i + 1) % 10 == 0:
            print(f"Saving intermediate results_phase1 after {i+1} experiments...")
            save_results(updated_results, results_file)
    
    # Save final results_phase1
    print(f"\nSaving final results_phase1 to {results_file}")
    save_results(updated_results, results_file)
    
    # Print summary
    successful = len([r for r in updated_results if r["status"] == "success"])
    print(f"\nSummary:")
    print(f"  Total experiments: {len(updated_results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"  Failed experiments: {failed_experiments[:5]}{'...' if len(failed_experiments) > 5 else ''}")


def convert_keys_to_str(obj):
    """Recursively convert dictionary keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    else:
        return obj


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results_phase1 to JSON file."""
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert all keys to strings and handle numpy types
    results_converted = convert_keys_to_str(results)
    
    # Save results_phase1
    with open(output_file, 'w') as f:
        json.dump(results_converted, f, indent=2, cls=NumpyEncoder)


def compare_old_vs_new_metrics(
    old_results_path: str = "paper_experiments/results_phase1/intermediate_results.json",
    new_results_path: str = "paper_experiments/results_phase1/rerun_attack_results.json"
) -> None:
    """Compare old vs new attack success metrics."""
    
    print("Comparing old vs new metrics...")
    
    # Load both result sets
    with open(old_results_path, 'r') as f:
        old_results = json.load(f)
    
    with open(new_results_path, 'r') as f:
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


def analyze_phase2_results(attack_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Phase 2 specific results with sampling effects."""
    
    analysis = {
        "sampling_scenario": config.get("sampling_scenario", "unknown"),
        "client_sampling_rate": config.get("client_sampling_rate", 1.0),
        "data_sampling_rate": config.get("data_sampling_rate", 1.0),
        "sampling_impact_metrics": {}
    }
    
    if attack_result["status"] == "success":
        # Analyze how sampling affects attack success
        for attack in attack_result.get("attack_results", []):
            attack_name = attack.get("attack_name", "")
            success_metric = attack.get("attack_success_metric", 0.0)
            
            # Calculate sampling impact score
            # Lower participation should generally reduce attack effectiveness
            participation_factor = config.get("client_sampling_rate", 1.0) * config.get("data_sampling_rate", 1.0)
            expected_reduction = 1.0 - participation_factor
            
            analysis["sampling_impact_metrics"][attack_name] = {
                "success_metric": success_metric,
                "participation_factor": participation_factor,
                "expected_reduction": expected_reduction,
                "relative_effectiveness": success_metric / max(participation_factor, 0.1)  # Avoid division by zero
            }
    
    return analysis


def compare_phase1_vs_phase2(
    phase1_results_file: str = "paper_experiments/results_phase1/rerun_attack_results.json",
    phase2_results_file: str = "paper_experiments/results_phase2/rerun_attack_results.json"
) -> None:
    """Compare Phase 1 vs Phase 2 results to analyze sampling effects."""
    
    print("Comparing Phase 1 vs Phase 2 results...")
    
    try:
        with open(phase1_results_file, 'r') as f:
            phase1_results = json.load(f)
        
        with open(phase2_results_file, 'r') as f:
            phase2_results = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find results file: {e}")
        return
    
    # Group Phase 1 results by configuration (ignoring sampling)
    phase1_by_config = {}
    for result in phase1_results:
        if result["status"] == "success":
            config = result["config"]
            key = f"{config['dataset']}_{config['fl_type']}_{config['topology']}_{config['node_count']}_{config['dp_setting']['name']}_{config['attack_strategy']}"
            phase1_by_config[key] = result
    
    # Compare with Phase 2 results
    comparisons = []
    
    for phase2_result in phase2_results:
        if phase2_result["status"] == "success":
            config = phase2_result["config"]
            key = f"{config['dataset']}_{config['fl_type']}_{config['topology']}_{config['node_count']}_{config['dp_setting']['name']}_{config['attack_strategy']}"
            
            if key in phase1_by_config:
                phase1_result = phase1_by_config[key]
                
                # Extract attack success metrics
                def get_attack_metric(res, attack_name):
                    for attack in res["attack_results"]["attack_results"]:
                        if attack["attack_name"] == attack_name:
                            return attack["attack_success_metric"]
                    return 0.0
                
                phase1_metric = get_attack_metric(phase1_result, "Parameter Magnitude Attack")
                phase2_metric = get_attack_metric(phase2_result, "Parameter Magnitude Attack")
                
                comparisons.append({
                    "config_key": key,
                    "sampling_scenario": config.get("sampling_scenario", "unknown"),
                    "client_sampling_rate": config.get("client_sampling_rate", 1.0),
                    "data_sampling_rate": config.get("data_sampling_rate", 1.0),
                    "phase1_metric": phase1_metric,
                    "phase2_metric": phase2_metric,
                    "metric_reduction": phase1_metric - phase2_metric,
                    "relative_reduction": (phase1_metric - phase2_metric) / max(phase1_metric, 0.001)
                })
    
    # Analyze results
    if comparisons:
        print(f"\nPhase 1 vs Phase 2 Comparison Summary:")
        print(f"  Configurations compared: {len(comparisons)}")
        
        # Group by sampling scenario
        by_scenario = {}
        for comp in comparisons:
            scenario = comp["sampling_scenario"]
            if scenario not in by_scenario:
                by_scenario[scenario] = []
            by_scenario[scenario].append(comp)
        
        for scenario, comps in by_scenario.items():
            avg_reduction = sum(c["metric_reduction"] for c in comps) / len(comps)
            avg_relative = sum(c["relative_reduction"] for c in comps) / len(comps)
            
            print(f"\n  {scenario}:")
            print(f"    Experiments: {len(comps)}")
            print(f"    Avg metric reduction: {avg_reduction:.6f}")
            print(f"    Avg relative reduction: {avg_relative:.1%}")
            
            # Show example
            example = comps[0]
            print(f"    Example: {example['phase1_metric']:.6f} → {example['phase2_metric']:.6f} (C={example['client_sampling_rate']:.1f}, D={example['data_sampling_rate']:.1f})")
    
    else:
        print("No matching configurations found between Phase 1 and Phase 2 results.")


if __name__ == "__main__":
    import sys
    
    # Command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Just run comparison
            compare_old_vs_new_metrics()
        elif sys.argv[1] == "--compare-phases":
            # Compare Phase 1 vs Phase 2
            compare_phase1_vs_phase2()
        elif sys.argv[1] == "--phase2":
            # Process Phase 2 experiments
            rerun_all_attacks(
                visualization_base_dir="paper_experiments/visualizations_phase2",
                results_file="paper_experiments/results_phase2/rerun_attack_results.json",
                experiment_phase=2
            )
        elif sys.argv[1] == "--filter":
            # Run specific experiments
            phase = 2 if "--phase2" in sys.argv else 1
            experiment_names = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
            
            if phase == 2:
                rerun_all_attacks(
                    visualization_base_dir="paper_experiments/visualizations_phase2",
                    results_file="paper_experiments/results_phase2/rerun_attack_results.json",
                    experiment_filter=experiment_names,
                    experiment_phase=2
                )
            else:
                rerun_all_attacks(experiment_filter=experiment_names)
        else:
            print("Usage:")
            print("  python rerun_attacks.py                           # Process all Phase 1 experiments")
            print("  python rerun_attacks.py --phase2                  # Process all Phase 2 experiments")
            print("  python rerun_attacks.py --filter exp_001 exp_002  # Process specific Phase 1 experiments")
            print("  python rerun_attacks.py --filter --phase2 exp_2001 # Process specific Phase 2 experiments")
            print("  python rerun_attacks.py --compare                 # Compare old vs new metrics")
            print("  python rerun_attacks.py --compare-phases          # Compare Phase 1 vs Phase 2 results")
    else:
        # Run all Phase 1 experiments
        rerun_all_attacks()
        
        # Also run comparison if old results_phase1 exist
        old_results_file = "paper_experiments/results_phase1/intermediate_results.json"
        if os.path.exists(old_results_file):
            compare_old_vs_new_metrics()