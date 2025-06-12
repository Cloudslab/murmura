"""
Parallel version of scalability simulator for faster execution.
"""

from typing import Dict, List, Tuple, Any
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from .scalability_simulator import (
    NetworkConfig, LargeScaleAttackSimulator, 
    analyze_scalability_results, _convert_for_json
)


def run_single_experiment(args: Tuple[NetworkConfig, str, int]) -> Dict[str, Any]:
    """Run a single experiment in a separate process."""
    config, experiment_key, experiment_id = args
    
    try:
        simulator = LargeScaleAttackSimulator(config)
        result = simulator.simulate_attack_execution()
        result['experiment_id'] = experiment_id
        result['experiment_key'] = experiment_key
        result['status'] = 'success'
        
        # Extract key metrics for logging
        eval_result = result['evaluation']
        attack_success = eval_result.get('overall_success', False)
        max_signal = eval_result.get('attack_indicators', {}).get('max_signal', 0.0)
        
        return {
            'result': result,
            'success': True,
            'message': f"Attack {'SUCCESS' if attack_success else 'FAILED'} (signal: {max_signal:.3f})"
        }
        
    except Exception as e:
        return {
            'result': {
                'experiment_id': experiment_id,
                'experiment_key': experiment_key,
                'config': config.__dict__,
                'status': 'failed',
                'error': str(e)
            },
            'success': False,
            'message': f"Experiment failed: {str(e)}"
        }


def run_scalability_experiments_parallel(network_sizes: List[int], 
                                       topologies: List[str],
                                       attack_strategies: List[str],
                                       dp_settings: List[Dict[str, Any]],
                                       output_dir: str = "./scalability_results",
                                       num_workers: int = 4) -> Dict[str, Any]:
    """Run scalability experiments in parallel for faster execution."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Files for incremental saving
    results_file = output_path / "scalability_results.json"
    checkpoint_file = output_path / "experiment_checkpoint.json"
    
    # Load existing results and checkpoint if available
    all_results = []
    completed_experiments = set()
    experiment_id = 0
    
    if results_file.exists():
        print(f"📂 Loading existing results from {results_file}")
        try:
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            print(f"   Loaded {len(all_results)} existing results")
        except Exception as e:
            print(f"   ⚠️  Error loading results: {e}, starting fresh")
            all_results = []
    
    if checkpoint_file.exists():
        print(f"📂 Loading checkpoint from {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed_experiments = set(checkpoint.get('completed_experiments', []))
                experiment_id = checkpoint.get('last_experiment_id', 0)
            print(f"   Resuming from experiment {experiment_id}")
        except Exception as e:
            print(f"   ⚠️  Error loading checkpoint: {e}, starting fresh")
    
    print(f"🚀 Starting parallel scalability experiments with {num_workers} workers...")
    print(f"   Network sizes: {network_sizes}")
    print(f"   Topologies: {topologies}")
    print(f"   Attack strategies: {attack_strategies}")
    print(f"   DP settings: {len(dp_settings)}")
    
    # Generate all experiment configurations
    experiment_configs = []
    for size in network_sizes:
        for topology in topologies:
            for attack_strategy in attack_strategies:
                for dp_setting in dp_settings:
                    experiment_key = f"{size}_{topology}_{attack_strategy}_{dp_setting.get('name', 'no_dp')}"
                    
                    # Skip if already completed
                    if experiment_key in completed_experiments:
                        continue
                    
                    try:
                        config = NetworkConfig(
                            num_nodes=size,
                            topology=topology,
                            attack_strategy=attack_strategy,
                            dp_enabled=dp_setting.get('enabled', False),
                            dp_epsilon=dp_setting.get('epsilon'),
                            num_rounds=5  # Reduced for large-scale simulation
                        )
                        
                        experiment_configs.append((config, experiment_key, experiment_id))
                        experiment_id += 1
                        
                    except ValueError as e:
                        # Skip invalid configurations
                        print(f"   ⚠️  Skipping invalid config: {experiment_key} - {e}")
    
    total_experiments = len(experiment_configs) + len(completed_experiments)
    print(f"   Total experiments: {total_experiments} ({len(experiment_configs)} to run, {len(completed_experiments)} already completed)")
    
    if not experiment_configs:
        print("✅ All experiments already completed!")
        return {
            'results_file': str(results_file),
            'analysis_file': str(output_path / "scalability_analysis.json"),
            'total_experiments': total_experiments,
            'successful_experiments': len(all_results),
            'skipped_experiments': len(completed_experiments)
        }
    
    # Run experiments in parallel
    start_time = time.time()
    successful = 0
    failed = 0
    
    if num_workers == 1:
        # Sequential execution for debugging
        print("🐌 Running in sequential mode (num_workers=1)")
        for i, args in enumerate(experiment_configs):
            config, exp_key, exp_id = args
            progress_pct = (len(completed_experiments) + i) / total_experiments * 100
            print(f"📊 Experiment {exp_id + 1} ({progress_pct:.1f}% complete): "
                  f"{config.num_nodes} nodes, {config.topology}, {config.attack_strategy}, "
                  f"DP={'on' if config.dp_enabled else 'off'}")
            
            result = run_single_experiment(args)
            print(f"   {result['message']}")
            
            all_results.append(result['result'])
            completed_experiments.add(exp_key)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
            
            # Save incrementally
            _save_incremental_results(all_results, completed_experiments, 
                                    experiment_id, total_experiments,
                                    results_file, checkpoint_file)
    else:
        # Parallel execution
        print(f"🚀 Running in parallel mode with {num_workers} workers")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(run_single_experiment, args): args 
                for args in experiment_configs
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_config)):
                config, exp_key, exp_id = future_to_config[future]
                progress_pct = (len(completed_experiments) + i + 1) / total_experiments * 100
                
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per experiment
                    print(f"📊 [{progress_pct:5.1f}%] Experiment {exp_id + 1}: "
                          f"{config.num_nodes} nodes, {config.topology}, {config.attack_strategy}, "
                          f"DP={'on' if config.dp_enabled else 'off'} - {result['message']}")
                    
                    all_results.append(result['result'])
                    completed_experiments.add(exp_key)
                    
                    if result['success']:
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    print(f"❌ [{progress_pct:5.1f}%] Experiment {exp_id + 1} failed: {e}")
                    all_results.append({
                        'experiment_id': exp_id,
                        'experiment_key': exp_key,
                        'config': config.__dict__,
                        'status': 'failed',
                        'error': str(e)
                    })
                    completed_experiments.add(exp_key)
                    failed += 1
                
                # Save incrementally every 10 experiments
                if (i + 1) % 10 == 0:
                    _save_incremental_results(all_results, completed_experiments,
                                            exp_id, total_experiments,
                                            results_file, checkpoint_file)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Parallel experiments completed in {elapsed_time/60:.1f} minutes")
    print(f"   Successful: {successful}, Failed: {failed}")
    
    # Final save
    _save_incremental_results(all_results, completed_experiments,
                            experiment_id, total_experiments,
                            results_file, checkpoint_file)
    
    # Generate analysis
    print("📊 Generating analysis...")
    analysis = analyze_scalability_results(all_results)
    analysis_file = output_path / "scalability_analysis.json"
    
    try:
        with open(analysis_file, 'w') as f:
            json.dump(_convert_for_json(analysis), f, indent=2)
        print(f"   Analysis saved to: {analysis_file}")
    except Exception as e:
        print(f"   ⚠️  Error saving analysis: {e}")
    
    return {
        'results_file': str(results_file),
        'analysis_file': str(analysis_file),
        'total_experiments': total_experiments,
        'successful_experiments': successful,
        'failed_experiments': failed,
        'elapsed_time_seconds': elapsed_time
    }


def _save_incremental_results(all_results: List[Dict], 
                            completed_experiments: set,
                            last_experiment_id: int,
                            total_experiments: int,
                            results_file: Path,
                            checkpoint_file: Path):
    """Save results and checkpoint incrementally."""
    try:
        # Save results
        with open(results_file, 'w') as f:
            json_results = [_convert_for_json(r) for r in all_results]
            json.dump(json_results, f, indent=2)
        
        # Save checkpoint
        checkpoint_data = {
            'last_experiment_id': last_experiment_id,
            'completed_experiments': list(completed_experiments),
            'total_experiments': total_experiments,
            'timestamp': str(pd.Timestamp.now())
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
    except Exception as e:
        print(f"   ⚠️  Warning: Failed to save incremental results: {e}")