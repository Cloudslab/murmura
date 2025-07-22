#!/usr/bin/env python3
"""
Simple test of defense mechanisms on a few experiments without complex features.
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add murmura to path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from defense_mechanisms import (
    DefenseConfig, StructuralNoiseInjection, TopologyAwareDifferentialPrivacy
)
from murmura.attacks.topology_attacks import (
    CommunicationPatternAttack, ParameterMagnitudeAttack, TopologyStructureAttack
)


def load_experiment_data(experiment_dir: Path) -> dict:
    """Load experiment data from directory."""
    data = {}
    
    data_files = {
        'communications': 'training_data_communications.csv',
        'parameter_updates': 'training_data_parameter_updates.csv',
        'topology': 'training_data_topology.csv',
        'metrics': 'training_data_metrics.csv'
    }
    
    for data_type, filename in data_files.items():
        filepath = experiment_dir / filename
        if filepath.exists():
            try:
                data[data_type] = pd.read_csv(filepath)
                print(f"Loaded {filename}: {len(data[data_type])} rows")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    
    return data


def run_attacks_on_data(data: dict) -> dict:
    """Run topology attacks on data."""
    attacks = [
        CommunicationPatternAttack(),
        ParameterMagnitudeAttack(), 
        TopologyStructureAttack()
    ]
    
    attack_results = []
    
    for attack in attacks:
        try:
            result = attack.execute_attack(data)
            result['attack_name'] = attack.name
            attack_results.append(result)
            print(f"{attack.name}: Success = {result.get('attack_success_metric', 0):.3f}")
        except Exception as e:
            print(f"Attack {attack.name} failed: {e}")
            attack_results.append({
                'attack_name': attack.name,
                'error': str(e),
                'attack_success_metric': 0.0
            })
    
    return {'attack_results': attack_results}


def test_defenses():
    """Test defense mechanisms on a single experiment."""
    
    # Load data from one experiment
    data_dir = Path('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/experiments_archive/phase1_baseline_analysis/training_data')
    
    # Find a star topology experiment
    exp_dirs = [d for d in data_dir.iterdir() if d.is_dir() and 'star' in d.name and 'mnist' in d.name]
    
    if not exp_dirs:
        print("No suitable experiments found")
        return
    
    experiment_dir = exp_dirs[0]
    print(f"Testing on experiment: {experiment_dir.name}")
    
    # Load original data
    original_data = load_experiment_data(experiment_dir)
    
    if not original_data:
        print("No data loaded")
        return
        
    print("\nOriginal data loaded:")
    for key, df in original_data.items():
        print(f"  {key}: {len(df)} rows")
    
    # Run attacks on original data
    print("\n=== ORIGINAL DATA ATTACKS ===")
    original_results = run_attacks_on_data(original_data)
    
    # Test Structural Noise Injection
    print("\n=== TESTING STRUCTURAL NOISE INJECTION ===")
    
    config = DefenseConfig(
        enable_comm_noise=True,
        comm_noise_rate=0.2,
        enable_timing_noise=True,
        timing_noise_std=0.15,
        enable_magnitude_noise=True,
        magnitude_noise_multiplier=0.1,
        enable_topology_reconfig=False,
        enable_topology_aware_dp=False
    )
    
    noise_defense = StructuralNoiseInjection(config)
    defended_data_noise = noise_defense.apply_defense(original_data)
    
    print("Defended data (noise injection):")
    for key, df in defended_data_noise.items():
        print(f"  {key}: {len(df)} rows")
    
    print("\nAttacks on noise-defended data:")
    noise_results = run_attacks_on_data(defended_data_noise)
    
    # Test Topology-Aware DP
    print("\n=== TESTING TOPOLOGY-AWARE DP ===")
    
    config_dp = DefenseConfig(
        enable_comm_noise=False,
        enable_timing_noise=False,
        enable_magnitude_noise=False,
        enable_topology_reconfig=False,
        enable_topology_aware_dp=True,
        structural_amplification_factor=1.5,
        neighbor_correlation_weight=0.1
    )
    
    dp_defense = TopologyAwareDifferentialPrivacy(config_dp)
    defended_data_dp = dp_defense.apply_defense(original_data)
    
    print("Attacks on DP-defended data:")
    dp_results = run_attacks_on_data(defended_data_dp)
    
    # Compare results
    print("\n=== DEFENSE EFFECTIVENESS SUMMARY ===")
    
    def print_attack_comparison(original, defended, defense_name):
        print(f"\n{defense_name}:")
        for i, (orig_attack, def_attack) in enumerate(zip(
            original['attack_results'], defended['attack_results']
        )):
            attack_name = orig_attack.get('attack_name', f'Attack_{i}')
            orig_success = orig_attack.get('attack_success_metric', 0.0)
            def_success = def_attack.get('attack_success_metric', 0.0)
            
            reduction = orig_success - def_success
            reduction_pct = (reduction / max(orig_success, 0.001)) * 100
            
            print(f"  {attack_name}:")
            print(f"    Original: {orig_success:.3f}")
            print(f"    Defended: {def_success:.3f}")
            print(f"    Reduction: {reduction:.3f} ({reduction_pct:.1f}%)")
    
    print_attack_comparison(original_results, noise_results, "Structural Noise Injection")
    print_attack_comparison(original_results, dp_results, "Topology-Aware DP")
    
    # Save results to file
    output_dir = Path('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/defense_test_results')
    output_dir.mkdir(exist_ok=True)
    
    results_summary = {
        'experiment': experiment_dir.name,
        'original_results': original_results,
        'noise_injection_results': noise_results,
        'topology_aware_dp_results': dp_results
    }
    
    # Convert numpy types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        else:
            return obj
    
    results_clean = clean_for_json(results_summary)
    
    with open(output_dir / 'simple_defense_test_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'simple_defense_test_results.json'}")


if __name__ == '__main__':
    test_defenses()