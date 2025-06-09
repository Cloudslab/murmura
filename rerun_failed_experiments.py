#!/usr/bin/env python3
"""
Script to rerun failed experiments from paper_experiments.py
Parses the log file to identify failed experiments and reruns them.
"""

import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def parse_failed_experiments(log_file: Path) -> List[Dict[str, Any]]:
    """Parse log file to extract failed experiment configurations."""
    failed_experiments = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern to match failed experiments
    # Example: [276/520] ❌ ham10000/federated/star/7n | DP(ε=4.0) | sensitive-groups | EXPERIMENT FAILED
    pattern = r'\[(\d+)/\d+\] ❌ (\w+)/(\w+)/(\w+)/(\d+)n \| DP\(ε=([0-9.]+)\) \| ([\w-]+) \| EXPERIMENT FAILED'
    no_dp_pattern = r'\[(\d+)/\d+\] ❌ (\w+)/(\w+)/(\w+)/(\d+)n \| no-DP \| ([\w-]+) \| EXPERIMENT FAILED'
    
    # Find all failed experiments with DP
    for match in re.finditer(pattern, content):
        exp_num, dataset, fl_type, topology, node_count, epsilon, attack = match.groups()
        
        # Map epsilon to dp_setting
        dp_setting_map = {
            '16.0': 'weak_dp',
            '8.0': 'medium_dp', 
            '4.0': 'strong_dp',
            '1.0': 'very_strong_dp'
        }
        
        failed_experiments.append({
            'experiment_number': int(exp_num),
            'dataset': dataset,
            'fl_type': fl_type,
            'topology': topology,
            'node_count': int(node_count),
            'dp_setting': dp_setting_map.get(epsilon, 'unknown'),
            'epsilon': float(epsilon),
            'attack_strategy': attack.replace('-', '_')
        })
    
    # Find all failed experiments without DP
    for match in re.finditer(no_dp_pattern, content):
        exp_num, dataset, fl_type, topology, node_count, attack = match.groups()
        
        failed_experiments.append({
            'experiment_number': int(exp_num),
            'dataset': dataset,
            'fl_type': fl_type,
            'topology': topology,
            'node_count': int(node_count),
            'dp_setting': 'no_dp',
            'epsilon': None,
            'attack_strategy': attack.replace('-', '_')
        })
    
    return failed_experiments

def run_failed_experiments(failed_experiments: List[Dict[str, Any]], dry_run: bool = False):
    """Run the failed experiments using paper_experiments.py"""
    
    print(f"\nFound {len(failed_experiments)} failed experiments:")
    for exp in failed_experiments:
        print(f"  - Experiment {exp['experiment_number']}: {exp['dataset']}/{exp['fl_type']}/{exp['topology']}/{exp['node_count']}n "
              f"| DP: {exp['dp_setting']} | Attack: {exp['attack_strategy']}")
    
    if dry_run:
        print("\nDry run mode - not executing experiments")
        return
    
    # Build command to run paper_experiments.py with specific configurations
    for i, exp in enumerate(failed_experiments):
        print(f"\n[{i+1}/{len(failed_experiments)}] Rerunning experiment {exp['experiment_number']}...")
        
        # Build filter arguments
        cmd = [
            "python", "paper_experiments.py",
            "--datasets", exp['dataset'],
            "--fl-types", exp['fl_type'],
            "--topologies", exp['topology'],
            "--node-counts", str(exp['node_count']),
            "--dp-settings", exp['dp_setting'],
            "--attack-strategies", exp['attack_strategy'],
            "--max-parallel", "1"  # Run one at a time to avoid resource issues
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Successfully completed")
            else:
                print(f"❌ Failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"❌ Exception occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Rerun failed experiments from paper_experiments.py")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("paper_experiments/logs/paper_experiments_20250608_142742.log"),
        help="Path to the log file containing experiment results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and show failed experiments without running them"
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Export failed experiments to JSON file"
    )
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1
    
    # Parse failed experiments
    failed_experiments = parse_failed_experiments(args.log_file)
    
    if args.export_json:
        with open(args.export_json, 'w') as f:
            json.dump(failed_experiments, f, indent=2)
        print(f"Exported {len(failed_experiments)} failed experiments to {args.export_json}")
    
    # Run experiments
    run_failed_experiments(failed_experiments, dry_run=args.dry_run)
    
    return 0

if __name__ == "__main__":
    exit(main())