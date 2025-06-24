#!/usr/bin/env python3
"""
Comprehensive evaluation of all defense mechanisms across ALL phase1 configurations.
Tests 520 experiments across MNIST and HAM10000 datasets with complete defense evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Add murmura to path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from defense_mechanisms import (
    DefenseConfig, StructuralNoiseInjection, DynamicTopologyReconfiguration,
    TopologyAwareDifferentialPrivacy, DefenseEvaluator
)
from murmura.attacks.topology_attacks import (
    CommunicationPatternAttack, ParameterMagnitudeAttack, TopologyStructureAttack
)


class ComprehensivePhase1Evaluator:
    """Comprehensive evaluator for all phase1 configurations."""
    
    def __init__(self, data_dir: str, output_dir: str, max_workers: int = 4):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'comprehensive_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Define ALL defense configurations to test
        self.defense_configs = {
            # Structural Noise Injection variants
            'structural_noise_weak': DefenseConfig(
                enable_comm_noise=True, comm_noise_rate=0.1,
                enable_timing_noise=True, timing_noise_std=0.05,
                enable_magnitude_noise=True, magnitude_noise_multiplier=0.05,
                enable_topology_reconfig=False, enable_topology_aware_dp=False
            ),
            'structural_noise_medium': DefenseConfig(
                enable_comm_noise=True, comm_noise_rate=0.2,
                enable_timing_noise=True, timing_noise_std=0.15,
                enable_magnitude_noise=True, magnitude_noise_multiplier=0.15,
                enable_topology_reconfig=False, enable_topology_aware_dp=False
            ),
            'structural_noise_strong': DefenseConfig(
                enable_comm_noise=True, comm_noise_rate=0.3,
                enable_timing_noise=True, timing_noise_std=0.3,
                enable_magnitude_noise=True, magnitude_noise_multiplier=0.3,
                enable_topology_reconfig=False, enable_topology_aware_dp=False
            ),
            
            # Topology-Aware DP variants
            'topology_aware_dp_weak': DefenseConfig(
                enable_comm_noise=False, enable_timing_noise=False, enable_magnitude_noise=False,
                enable_topology_reconfig=False, enable_topology_aware_dp=True,
                structural_amplification_factor=1.2, neighbor_correlation_weight=0.05
            ),
            'topology_aware_dp_medium': DefenseConfig(
                enable_comm_noise=False, enable_timing_noise=False, enable_magnitude_noise=False,
                enable_topology_reconfig=False, enable_topology_aware_dp=True,
                structural_amplification_factor=1.5, neighbor_correlation_weight=0.1
            ),
            'topology_aware_dp_strong': DefenseConfig(
                enable_comm_noise=False, enable_timing_noise=False, enable_magnitude_noise=False,
                enable_topology_reconfig=False, enable_topology_aware_dp=True,
                structural_amplification_factor=2.0, neighbor_correlation_weight=0.2
            ),
            
            # Dynamic Topology Reconfiguration variants
            'topology_reconfig_weak': DefenseConfig(
                enable_comm_noise=False, enable_timing_noise=False, enable_magnitude_noise=False,
                enable_topology_reconfig=True, reconfig_frequency=10, preserve_connectivity=True,
                enable_topology_aware_dp=False
            ),
            'topology_reconfig_medium': DefenseConfig(
                enable_comm_noise=False, enable_timing_noise=False, enable_magnitude_noise=False,
                enable_topology_reconfig=True, reconfig_frequency=5, preserve_connectivity=True,
                enable_topology_aware_dp=False
            ),
            'topology_reconfig_strong': DefenseConfig(
                enable_comm_noise=False, enable_timing_noise=False, enable_magnitude_noise=False,
                enable_topology_reconfig=True, reconfig_frequency=3, preserve_connectivity=True,
                enable_topology_aware_dp=False
            ),
            
            # Combined defense strategies
            'combined_weak': DefenseConfig(
                enable_comm_noise=True, comm_noise_rate=0.05,
                enable_timing_noise=True, timing_noise_std=0.05,
                enable_magnitude_noise=True, magnitude_noise_multiplier=0.05,
                enable_topology_reconfig=False, enable_topology_aware_dp=True,
                structural_amplification_factor=1.2, neighbor_correlation_weight=0.05
            ),
            'combined_medium': DefenseConfig(
                enable_comm_noise=True, comm_noise_rate=0.1,
                enable_timing_noise=True, timing_noise_std=0.1,
                enable_magnitude_noise=True, magnitude_noise_multiplier=0.1,
                enable_topology_reconfig=False, enable_topology_aware_dp=True,
                structural_amplification_factor=1.5, neighbor_correlation_weight=0.1
            ),
            'combined_strong': DefenseConfig(
                enable_comm_noise=True, comm_noise_rate=0.15,
                enable_timing_noise=True, timing_noise_std=0.15,
                enable_magnitude_noise=True, magnitude_noise_multiplier=0.15,
                enable_topology_reconfig=True, reconfig_frequency=5, preserve_connectivity=True,
                enable_topology_aware_dp=True,
                structural_amplification_factor=1.8, neighbor_correlation_weight=0.15
            ),
        }
        
        self.logger.info(f"Initialized evaluator with {len(self.defense_configs)} defense configurations")
        
    def load_experiment_data(self, experiment_dir: Path) -> Dict[str, pd.DataFrame]:
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
                except Exception as e:
                    self.logger.warning(f"Failed to load {filename} from {experiment_dir.name}: {e}")
        
        return data
    
    def run_attacks_on_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
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
            except Exception as e:
                # Log error but continue with other attacks
                attack_results.append({
                    'attack_name': attack.name,
                    'error': str(e),
                    'attack_success_metric': 0.0
                })
        
        return {'attack_results': attack_results}
    
    def apply_defense(self, data: Dict[str, pd.DataFrame], config: DefenseConfig) -> Dict[str, pd.DataFrame]:
        """Apply defense mechanism to data."""
        defended_data = data.copy()
        
        try:
            # Apply structural noise injection
            if config.enable_comm_noise or config.enable_timing_noise or config.enable_magnitude_noise:
                noise_defense = StructuralNoiseInjection(config)
                defended_data = noise_defense.apply_defense(defended_data)
            
            # Apply dynamic topology reconfiguration
            if config.enable_topology_reconfig:
                reconfig_defense = DynamicTopologyReconfiguration(config)
                defended_data = reconfig_defense.apply_defense(defended_data)
            
            # Apply topology-aware DP
            if config.enable_topology_aware_dp:
                dp_defense = TopologyAwareDifferentialPrivacy(config)
                defended_data = dp_defense.apply_defense(defended_data)
                
        except Exception as e:
            self.logger.error(f"Defense application failed: {e}")
            return data  # Return original data if defense fails
        
        return defended_data
    
    def evaluate_single_experiment(self, experiment_path: Tuple[Path, str]) -> Dict[str, Any]:
        """Evaluate a single experiment with all defense mechanisms."""
        experiment_dir, defense_name = experiment_path
        experiment_name = experiment_dir.name
        
        try:
            # Load original data
            original_data = self.load_experiment_data(experiment_dir)
            
            if len(original_data) < 3:  # Need at least communications, parameters, topology
                return {
                    'experiment_name': experiment_name,
                    'defense_name': defense_name,
                    'error': 'Insufficient data files',
                    'success': False
                }
            
            # Run attacks on original data
            original_results = self.run_attacks_on_data(original_data)
            
            # Apply defense
            config = self.defense_configs[defense_name]
            defended_data = self.apply_defense(original_data, config)
            
            # Run attacks on defended data
            defended_results = self.run_attacks_on_data(defended_data)
            
            # Calculate defense effectiveness
            effectiveness_metrics = self._calculate_effectiveness(
                original_results['attack_results'],
                defended_results['attack_results']
            )
            
            return {
                'experiment_name': experiment_name,
                'defense_name': defense_name,
                'original_results': original_results,
                'defended_results': defended_results,
                'effectiveness_metrics': effectiveness_metrics,
                'data_summary': {k: len(v) for k, v in original_data.items()},
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {experiment_name} with {defense_name}: {e}")
            return {
                'experiment_name': experiment_name,
                'defense_name': defense_name,
                'error': str(e),
                'success': False
            }
    
    def _calculate_effectiveness(self, original_attacks: List[Dict], defended_attacks: List[Dict]) -> Dict[str, Any]:
        """Calculate defense effectiveness metrics."""
        metrics = {
            'attack_reductions': {},
            'overall_effectiveness': 0.0,
            'valid_attacks': 0
        }
        
        total_reduction = 0.0
        
        for orig_attack, def_attack in zip(original_attacks, defended_attacks):
            attack_name = orig_attack.get('attack_name', 'Unknown')
            orig_success = orig_attack.get('attack_success_metric', 0.0)
            def_success = def_attack.get('attack_success_metric', 0.0)
            
            if orig_success > 0:
                reduction_pct = max(0.0, (orig_success - def_success) / orig_success) * 100
                metrics['attack_reductions'][attack_name] = {
                    'original_success': orig_success,
                    'defended_success': def_success,
                    'reduction_percentage': reduction_pct,
                    'absolute_reduction': orig_success - def_success
                }
                total_reduction += reduction_pct
                metrics['valid_attacks'] += 1
        
        if metrics['valid_attacks'] > 0:
            metrics['overall_effectiveness'] = total_reduction / metrics['valid_attacks']
        
        return metrics
    
    def run_comprehensive_evaluation(self, max_experiments: int = None, 
                                   dataset_filter: str = None) -> Dict[str, Any]:
        """Run comprehensive evaluation across all phase1 configurations."""
        start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPREHENSIVE PHASE1 DEFENSE EVALUATION")
        self.logger.info("=" * 80)
        
        # Find all experiment directories
        training_data_dir = self.data_dir / 'training_data'
        if not training_data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")
        
        experiment_dirs = [d for d in training_data_dir.iterdir() if d.is_dir()]
        
        # Apply dataset filter if specified
        if dataset_filter:
            experiment_dirs = [d for d in experiment_dirs if dataset_filter in d.name]
            self.logger.info(f"Filtered to {dataset_filter} experiments: {len(experiment_dirs)}")
        
        # Limit experiments if specified
        if max_experiments:
            experiment_dirs = experiment_dirs[:max_experiments]
            self.logger.info(f"Limited to {max_experiments} experiments")
        
        # Create all experiment-defense combinations
        experiment_defense_pairs = []
        for exp_dir in experiment_dirs:
            for defense_name in self.defense_configs.keys():
                experiment_defense_pairs.append((exp_dir, defense_name))
        
        total_evaluations = len(experiment_defense_pairs)
        self.logger.info(f"Total evaluations to run: {total_evaluations}")
        self.logger.info(f"Experiments: {len(experiment_dirs)}")
        self.logger.info(f"Defense mechanisms: {len(self.defense_configs)}")
        
        # Initialize results tracking
        all_results = []
        summary_stats = {
            'total_evaluations': total_evaluations,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'start_time': datetime.now().isoformat(),
            'defense_effectiveness': {},
            'attack_reductions': {},
            'experiment_summary': {},
            'dataset_breakdown': {}
        }
        
        # Process evaluations in batches to manage memory
        batch_size = 50
        num_batches = (total_evaluations + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_evaluations)
            batch_pairs = experiment_defense_pairs[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_idx + 1}/{num_batches} "
                           f"({len(batch_pairs)} evaluations)")
            
            # Use process pool for parallel evaluation
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.evaluate_single_experiment, pair) 
                          for pair in batch_pairs]
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per evaluation
                        all_results.append(result)
                        
                        if result['success']:
                            summary_stats['successful_evaluations'] += 1
                            self._update_summary_stats(summary_stats, result)
                        else:
                            summary_stats['failed_evaluations'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Evaluation failed: {e}")
                        summary_stats['failed_evaluations'] += 1
            
            # Save intermediate results
            if (batch_idx + 1) % 5 == 0:  # Save every 5 batches
                self._save_intermediate_results(all_results, summary_stats, batch_idx + 1)
        
        # Finalize summary statistics
        self._finalize_summary_stats(summary_stats)
        
        # Calculate completion time
        total_time = time.time() - start_time
        summary_stats['total_time_seconds'] = total_time
        summary_stats['end_time'] = datetime.now().isoformat()
        
        # Create comprehensive results
        comprehensive_results = {
            'summary_stats': summary_stats,
            'detailed_results': all_results,
            'defense_configs': {k: str(v.__dict__) for k, v in self.defense_configs.items()},
            'evaluation_metadata': {
                'total_experiments': len(experiment_dirs),
                'total_defenses': len(self.defense_configs),
                'max_workers': self.max_workers,
                'dataset_filter': dataset_filter,
                'max_experiments': max_experiments
            }
        }
        
        # Save final results
        self._save_final_results(comprehensive_results)
        
        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE EVALUATION COMPLETED")
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        self.logger.info(f"Successful evaluations: {summary_stats['successful_evaluations']}")
        self.logger.info(f"Failed evaluations: {summary_stats['failed_evaluations']}")
        self.logger.info("=" * 80)
        
        return comprehensive_results
    
    def _update_summary_stats(self, summary_stats: Dict[str, Any], result: Dict[str, Any]):
        """Update summary statistics with single result."""
        defense_name = result['defense_name']
        experiment_name = result['experiment_name']
        effectiveness = result['effectiveness_metrics']
        
        # Initialize defense stats if not exists
        if defense_name not in summary_stats['defense_effectiveness']:
            summary_stats['defense_effectiveness'][defense_name] = []
            summary_stats['attack_reductions'][defense_name] = {
                'Communication Pattern Attack': [],
                'Parameter Magnitude Attack': [],
                'Topology Structure Attack': []
            }
        
        # Add overall effectiveness
        summary_stats['defense_effectiveness'][defense_name].append(
            effectiveness['overall_effectiveness']
        )
        
        # Add attack-specific reductions
        for attack_name, reduction_info in effectiveness['attack_reductions'].items():
            if attack_name in summary_stats['attack_reductions'][defense_name]:
                summary_stats['attack_reductions'][defense_name][attack_name].append(
                    reduction_info['reduction_percentage']
                )
        
        # Track experiment types
        if 'mnist' in experiment_name:
            dataset = 'mnist'
        elif 'ham10000' in experiment_name:
            dataset = 'ham10000'
        else:
            dataset = 'unknown'
        
        if dataset not in summary_stats['dataset_breakdown']:
            summary_stats['dataset_breakdown'][dataset] = 0
        summary_stats['dataset_breakdown'][dataset] += 1
    
    def _finalize_summary_stats(self, summary_stats: Dict[str, Any]):
        """Calculate final summary statistics."""
        # Calculate average effectiveness for each defense
        for defense_name, effectiveness_list in summary_stats['defense_effectiveness'].items():
            if effectiveness_list:
                summary_stats['defense_effectiveness'][defense_name] = {
                    'mean': float(np.mean(effectiveness_list)),
                    'std': float(np.std(effectiveness_list)),
                    'min': float(np.min(effectiveness_list)),
                    'max': float(np.max(effectiveness_list)),
                    'count': len(effectiveness_list)
                }
        
        # Calculate average attack reductions
        for defense_name, attack_reductions in summary_stats['attack_reductions'].items():
            for attack_name, reduction_list in attack_reductions.items():
                if reduction_list:
                    summary_stats['attack_reductions'][defense_name][attack_name] = {
                        'mean': float(np.mean(reduction_list)),
                        'std': float(np.std(reduction_list)),
                        'min': float(np.min(reduction_list)),
                        'max': float(np.max(reduction_list)),
                        'count': len(reduction_list)
                    }
    
    def _save_intermediate_results(self, results: List[Dict], summary_stats: Dict, batch_num: int):
        """Save intermediate results after each batch."""
        intermediate_file = self.output_dir / f'intermediate_results_batch_{batch_num}.json'
        
        intermediate_data = {
            'batch_number': batch_num,
            'results_so_far': len(results),
            'summary_stats': summary_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(intermediate_file, 'w') as f:
            json.dump(intermediate_data, f, indent=2)
        
        self.logger.info(f"Saved intermediate results to {intermediate_file}")
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final comprehensive results."""
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
        
        results_clean = clean_for_json(results)
        
        # Save comprehensive results
        final_file = self.output_dir / 'comprehensive_phase1_evaluation_results.json'
        with open(final_file, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        # Save summary only
        summary_file = self.output_dir / 'evaluation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'summary_stats': results_clean['summary_stats'],
                'evaluation_metadata': results_clean['evaluation_metadata']
            }, f, indent=2)
        
        self.logger.info(f"Final results saved to {final_file}")
        self.logger.info(f"Summary saved to {summary_file}")


def main():
    """Main function for running comprehensive evaluation."""
    parser = argparse.ArgumentParser(description='Comprehensive Phase1 Defense Evaluation')
    parser.add_argument('--data-dir', type=str, 
                       default='/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/experiments_archive/phase1_baseline_analysis',
                       help='Path to phase1 baseline analysis data')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/comprehensive_phase1_results',
                       help='Output directory for results')
    parser.add_argument('--max-experiments', type=int, default=None,
                       help='Maximum number of experiments to evaluate (default: all)')
    parser.add_argument('--dataset-filter', type=str, choices=['mnist', 'ham10000'],
                       help='Filter experiments by dataset')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ComprehensivePhase1Evaluator(
        args.data_dir, 
        args.output_dir,
        max_workers=args.max_workers
    )
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        max_experiments=args.max_experiments,
        dataset_filter=args.dataset_filter
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total evaluations: {results['summary_stats']['total_evaluations']}")
    print(f"Successful: {results['summary_stats']['successful_evaluations']}")
    print(f"Failed: {results['summary_stats']['failed_evaluations']}")
    print(f"Success rate: {results['summary_stats']['successful_evaluations'] / results['summary_stats']['total_evaluations'] * 100:.1f}%")
    
    # Print top performing defenses
    defense_effectiveness = results['summary_stats']['defense_effectiveness']
    if defense_effectiveness:
        print(f"\nTop Performing Defense Mechanisms:")
        sorted_defenses = sorted(
            [(name, stats['mean']) for name, stats in defense_effectiveness.items() 
             if isinstance(stats, dict) and 'mean' in stats],
            key=lambda x: x[1], reverse=True
        )
        
        for i, (defense_name, effectiveness) in enumerate(sorted_defenses[:5], 1):
            print(f"{i}. {defense_name}: {effectiveness:.2f}% avg attack reduction")
    
    print(f"\nResults saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())