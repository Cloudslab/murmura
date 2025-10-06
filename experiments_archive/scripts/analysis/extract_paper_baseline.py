#!/usr/bin/env python3
"""
Extract baseline performance statistics from original paper results.
"""

import json

import numpy as np


def extract_baseline_stats():
    """Extract baseline attack success rates from original paper results."""
    
    # Load original results
    with open("experiments_archive/results/attack_results/results_phase1/rerun_attack_results.json", 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} experiments from original paper results")
    
    # Group attacks by type
    attack_stats = {
        "Communication Pattern Attack": [],
        "Parameter Magnitude Attack": [],
        "Topology Structure Attack": []
    }
    
    # Extract success metrics
    for exp in results:
        if exp["status"] == "success" and "attack_results" in exp["attack_results"]:
            for attack in exp["attack_results"]["attack_results"]:
                attack_name = attack["attack_name"]
                success_metric = attack.get("attack_success_metric", 0.0)
                
                if attack_name in attack_stats:
                    attack_stats[attack_name].append(success_metric)
    
    print("\nOriginal Paper Baseline Performance:")
    print("="*50)
    
    baseline_stats = {}
    for attack_name, metrics in attack_stats.items():
        if metrics:
            mean_success = np.mean(metrics)
            std_success = np.std(metrics)
            n_experiments = len(metrics)
            
            # Calculate 95% confidence interval
            se = std_success / np.sqrt(n_experiments)
            ci_margin = 1.96 * se
            ci_lower = mean_success - ci_margin
            ci_upper = mean_success + ci_margin
            
            threshold_status = "✓" if mean_success > 0.3 else "✗"
            
            print(f"{attack_name}:")
            print(f"  Success Rate: {mean_success:.3f} ({mean_success*100:.1f}%) {threshold_status}")
            print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"  Sample Size: n={n_experiments}")
            print(f"  Std Dev: {std_success:.3f}")
            print()
            
            baseline_stats[attack_name] = {
                "mean": mean_success,
                "std": std_success,
                "n": n_experiments,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            }
    
    return baseline_stats

if __name__ == "__main__":
    baseline_stats = extract_baseline_stats()