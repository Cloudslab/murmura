#!/usr/bin/env python3
"""
Empirical evaluation of topology-aware defense mechanisms using phase1 baseline data.

This script evaluates the effectiveness of proposed defense mechanisms against
topology-based privacy attacks using existing experimental data from phase1_baseline_analysis.
"""

import sys
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add murmura to path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from defense_mechanisms import (
    StructuralNoiseInjection, DynamicTopologyReconfiguration,
    TopologyAwareDifferentialPrivacy, DefenseEvaluator, create_defense_config
)
from murmura.attacks.topology_attacks import (
    CommunicationPatternAttack, ParameterMagnitudeAttack, TopologyStructureAttack,
    AttackEvaluator
)


class DefenseExperimentRunner:
    """Runner for defense mechanism evaluation experiments."""
    
    def __init__(self, phase1_data_dir: str, output_dir: str):
        self.phase1_data_dir = Path(phase1_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'defense_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize defense mechanisms
        self.defense_configs = {
            'structural_noise_weak': create_defense_config('structural_noise', 'weak'),
            'structural_noise_medium': create_defense_config('structural_noise', 'medium'),
            'structural_noise_strong': create_defense_config('structural_noise', 'strong'),
            'topology_reconfig_weak': create_defense_config('topology_reconfig', 'weak'),
            'topology_reconfig_medium': create_defense_config('topology_reconfig', 'medium'),
            'topology_reconfig_strong': create_defense_config('topology_reconfig', 'strong'),
            'topology_aware_dp_weak': create_defense_config('topology_aware_dp', 'weak'),
            'topology_aware_dp_medium': create_defense_config('topology_aware_dp', 'medium'),
            'topology_aware_dp_strong': create_defense_config('topology_aware_dp', 'strong'),
            'balanced_weak': create_defense_config('balanced', 'weak'),
            'balanced_medium': create_defense_config('balanced', 'medium'),
            'balanced_strong': create_defense_config('balanced', 'strong'),
        }
        
        # Initialize attacks
        self.attacks = [
            CommunicationPatternAttack(),
            ParameterMagnitudeAttack(),
            TopologyStructureAttack()
        ]
        
        self.evaluator = DefenseEvaluator()
        
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
                    self.logger.debug(f"Loaded {filename}: {len(data[data_type])} rows")
                except Exception as e:
                    self.logger.warning(f"Failed to load {filename}: {e}")
        
        return data
    
    def apply_defense_mechanisms(
        self, 
        original_data: Dict[str, pd.DataFrame], 
        defense_name: str
    ) -> Dict[str, pd.DataFrame]:
        """Apply specified defense mechanism to data."""
        config = self.defense_configs[defense_name]
        
        # Initialize defense mechanisms
        defenses = []
        
        if config.enable_comm_noise or config.enable_timing_noise or config.enable_magnitude_noise:
            defenses.append(StructuralNoiseInjection(config))
        
        if config.enable_topology_reconfig:
            defenses.append(DynamicTopologyReconfiguration(config))
        
        if config.enable_topology_aware_dp:
            defenses.append(TopologyAwareDifferentialPrivacy(config))
        
        # Apply defenses sequentially
        defended_data = original_data.copy()
        for defense in defenses:
            defended_data = defense.apply_defense(defended_data)
        
        return defended_data
    
    def run_attacks_on_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run all topology attacks on given data."""
        attack_results = []
        
        for attack in self.attacks:
            try:
                result = attack.execute_attack(data)
                result['attack_name'] = attack.name
                attack_results.append(result)
            except Exception as e:
                self.logger.error(f"Attack {attack.name} failed: {e}")
                attack_results.append({
                    'attack_name': attack.name,
                    'error': str(e),
                    'attack_success_metric': 0.0
                })
        
        # Evaluate attack results
        attack_evaluator = AttackEvaluator()
        evaluation = attack_evaluator.evaluate_attacks(attack_results)
        
        return {
            'attack_results': attack_results,
            'evaluation': evaluation
        }
    
    def evaluate_single_experiment(
        self, 
        experiment_dir: Path, 
        defense_name: str
    ) -> Dict[str, Any]:
        """Evaluate defense effectiveness on a single experiment."""
        self.logger.info(f"Evaluating {defense_name} on {experiment_dir.name}")
        
        # Load original data
        original_data = self.load_experiment_data(experiment_dir)
        
        if not original_data:
            self.logger.warning(f"No data loaded for {experiment_dir.name}")
            return None
        
        # Run attacks on original data
        original_attack_results = self.run_attacks_on_data(original_data)
        
        # Apply defense mechanisms
        defended_data = self.apply_defense_mechanisms(original_data, defense_name)
        
        # Run attacks on defended data
        defended_attack_results = self.run_attacks_on_data(defended_data)
        
        # Evaluate defense effectiveness
        defense_evaluation = self.evaluator.evaluate_defense_effectiveness(
            original_data,
            defended_data,
            original_attack_results,
            defended_attack_results
        )
        
        return {
            'experiment_name': experiment_dir.name,
            'defense_name': defense_name,
            'original_attack_results': original_attack_results,
            'defended_attack_results': defended_attack_results,
            'defense_evaluation': defense_evaluation,
            'data_summary': {k: len(v) for k, v in original_data.items()},
            'defended_data_summary': {k: len(v) for k, v in defended_data.items()}
        }
    
    def run_comprehensive_evaluation(
        self, 
        max_experiments: int = 50,
        filter_criteria: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across multiple experiments and defenses."""
        self.logger.info("Starting comprehensive defense evaluation")
        
        # Find experiment directories
        training_data_dir = self.phase1_data_dir / 'training_data'
        if not training_data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")
        
        experiment_dirs = [d for d in training_data_dir.iterdir() if d.is_dir()]
        
        # Apply filters if specified
        if filter_criteria:
            experiment_dirs = self._filter_experiments(experiment_dirs, filter_criteria)
        
        # Limit number of experiments
        experiment_dirs = experiment_dirs[:max_experiments]
        
        self.logger.info(f"Evaluating {len(experiment_dirs)} experiments with {len(self.defense_configs)} defenses")
        
        all_results = []
        summary_stats = {
            'total_experiments': len(experiment_dirs),
            'total_defenses': len(self.defense_configs),
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'defense_effectiveness': {},
            'attack_degradation_by_defense': {},
        }
        
        for exp_dir in experiment_dirs:
            for defense_name in self.defense_configs.keys():
                try:
                    result = self.evaluate_single_experiment(exp_dir, defense_name)
                    if result:
                        all_results.append(result)
                        summary_stats['successful_evaluations'] += 1
                        
                        # Update summary statistics
                        self._update_summary_stats(summary_stats, result)
                    else:
                        summary_stats['failed_evaluations'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {exp_dir.name} with {defense_name}: {e}")
                    summary_stats['failed_evaluations'] += 1
        
        # Calculate overall statistics
        self._finalize_summary_stats(summary_stats, all_results)
        
        # Save results
        results_summary = {
            'summary_stats': summary_stats,
            'detailed_results': all_results,
            'evaluation_timestamp': datetime.now().isoformat(),
            'configuration': {
                'max_experiments': max_experiments,
                'filter_criteria': filter_criteria,
                'defense_configs': {k: str(v.__dict__) for k, v in self.defense_configs.items()}
            }
        }
        
        # Convert dict keys to strings before JSON serialization
        results_summary_clean = self._convert_dict_keys_to_str(results_summary)
        
        # Save to JSON (with better serialization)
        with open(self.output_dir / 'comprehensive_evaluation_results.json', 'w') as f:
            json.dump(results_summary_clean, f, indent=2, default=self._json_serializer)
        
        # Generate visualizations
        self.generate_visualizations(results_summary)
        
        return results_summary
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and pandas objects."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            # Convert dict keys to strings
            return {str(k): v for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)
    
    def _convert_dict_keys_to_str(self, obj):
        """Recursively convert dictionary keys to strings."""
        if isinstance(obj, dict):
            return {str(k): self._convert_dict_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_dict_keys_to_str(item) for item in obj]
        else:
            return obj
    
    def _filter_experiments(
        self, 
        experiment_dirs: List[Path], 
        filter_criteria: Dict[str, Any]
    ) -> List[Path]:
        """Filter experiments based on criteria."""
        filtered_dirs = []
        
        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name
            
            # Apply filters
            include = True
            
            if 'topology' in filter_criteria:
                if filter_criteria['topology'] not in exp_name:
                    include = False
            
            if 'dataset' in filter_criteria:
                if filter_criteria['dataset'] not in exp_name:
                    include = False
            
            if 'dp_level' in filter_criteria:
                if filter_criteria['dp_level'] not in exp_name:
                    include = False
            
            if include:
                filtered_dirs.append(exp_dir)
        
        return filtered_dirs
    
    def _update_summary_stats(self, summary_stats: Dict[str, Any], result: Dict[str, Any]):
        """Update summary statistics with single result."""
        defense_name = result['defense_name']
        defense_eval = result['defense_evaluation']
        
        # Initialize defense stats if not exists
        if defense_name not in summary_stats['defense_effectiveness']:
            summary_stats['defense_effectiveness'][defense_name] = []
            summary_stats['attack_degradation_by_defense'][defense_name] = {
                'Communication Pattern Attack': [],
                'Parameter Magnitude Attack': [],
                'Topology Structure Attack': []
            }
        
        # Add overall effectiveness
        summary_stats['defense_effectiveness'][defense_name].append(
            defense_eval['overall_effectiveness']
        )
        
        # Add attack-specific degradation
        for attack_name, degradation_info in defense_eval['attack_degradation'].items():
            if attack_name in summary_stats['attack_degradation_by_defense'][defense_name]:
                summary_stats['attack_degradation_by_defense'][defense_name][attack_name].append(
                    degradation_info['percentage_degradation']
                )
    
    def _finalize_summary_stats(self, summary_stats: Dict[str, Any], all_results: List[Dict[str, Any]]):
        """Calculate final summary statistics."""
        # Calculate average effectiveness for each defense
        for defense_name, effectiveness_list in summary_stats['defense_effectiveness'].items():
            if effectiveness_list:
                summary_stats['defense_effectiveness'][defense_name] = {
                    'mean': np.mean(effectiveness_list),
                    'std': np.std(effectiveness_list),
                    'min': np.min(effectiveness_list),
                    'max': np.max(effectiveness_list),
                    'count': len(effectiveness_list)
                }
        
        # Calculate average attack degradation
        for defense_name, attack_degradations in summary_stats['attack_degradation_by_defense'].items():
            for attack_name, degradation_list in attack_degradations.items():
                if degradation_list:
                    summary_stats['attack_degradation_by_defense'][defense_name][attack_name] = {
                        'mean': np.mean(degradation_list),
                        'std': np.std(degradation_list),
                        'min': np.min(degradation_list),
                        'max': np.max(degradation_list),
                        'count': len(degradation_list)
                    }
    
    def generate_visualizations(self, results_summary: Dict[str, Any]):
        """Generate visualization plots for defense evaluation."""
        plt.style.use('default')
        
        # 1. Defense Effectiveness Comparison
        self._plot_defense_effectiveness(results_summary)
        
        # 2. Attack Degradation by Defense Type
        self._plot_attack_degradation(results_summary)
        
        # 3. Defense Mechanism Comparison Heatmap
        self._plot_defense_heatmap(results_summary)
        
        plt.close('all')
    
    def _plot_defense_effectiveness(self, results_summary: Dict[str, Any]):
        """Plot overall defense effectiveness comparison."""
        defense_stats = results_summary['summary_stats']['defense_effectiveness']
        
        defense_names = []
        means = []
        stds = []
        
        for defense_name, stats in defense_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                defense_names.append(defense_name.replace('_', '\n'))
                means.append(stats['mean'])
                stds.append(stats['std'])
        
        if not defense_names:
            return
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(defense_names, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Defense Mechanism Effectiveness Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Defense Mechanism', fontsize=12)
        plt.ylabel('Overall Effectiveness (0-1)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Color bars by effectiveness level
        for bar, mean in zip(bars, means):
            if mean > 0.7:
                bar.set_color('green')
            elif mean > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'defense_effectiveness_comparison.png', dpi=300, bbox_inches='tight')
        self.logger.info("Generated defense effectiveness comparison plot")
    
    def _plot_attack_degradation(self, results_summary: Dict[str, Any]):
        """Plot attack degradation by defense mechanism."""
        attack_degradation = results_summary['summary_stats']['attack_degradation_by_defense']
        
        # Prepare data for grouped bar chart
        attack_names = ['Communication Pattern Attack', 'Parameter Magnitude Attack', 'Topology Structure Attack']
        defense_names = list(attack_degradation.keys())
        
        # Create matrix of degradation values
        degradation_matrix = []
        for defense_name in defense_names:
            defense_degradations = []
            for attack_name in attack_names:
                attack_stats = attack_degradation[defense_name].get(attack_name, {})
                if isinstance(attack_stats, dict) and 'mean' in attack_stats:
                    defense_degradations.append(attack_stats['mean'])
                else:
                    defense_degradations.append(0.0)
            degradation_matrix.append(defense_degradations)
        
        if not degradation_matrix:
            return
        
        # Create grouped bar chart
        x = np.arange(len(attack_names))
        width = 0.8 / len(defense_names)
        
        plt.figure(figsize=(15, 10))
        
        for i, (defense_name, degradations) in enumerate(zip(defense_names, degradation_matrix)):
            plt.bar(x + i * width, degradations, width, 
                   label=defense_name.replace('_', ' ').title(), alpha=0.8)
        
        plt.title('Attack Success Degradation by Defense Mechanism', fontsize=16, fontweight='bold')
        plt.xlabel('Attack Type', fontsize=12)
        plt.ylabel('Success Degradation (%)', fontsize=12)
        plt.xticks(x + width * (len(defense_names) - 1) / 2, 
                  [name.replace(' Attack', '') for name in attack_names])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attack_degradation_by_defense.png', dpi=300, bbox_inches='tight')
        self.logger.info("Generated attack degradation plot")
    
    def _plot_defense_heatmap(self, results_summary: Dict[str, Any]):
        """Plot heatmap of defense mechanism performance."""
        attack_degradation = results_summary['summary_stats']['attack_degradation_by_defense']
        
        # Prepare data for heatmap
        attack_names = ['Communication\nPattern', 'Parameter\nMagnitude', 'Topology\nStructure']
        defense_names = []
        heatmap_data = []
        
        for defense_name, attack_stats in attack_degradation.items():
            defense_names.append(defense_name.replace('_', '\n'))
            row_data = []
            
            for attack_name in ['Communication Pattern Attack', 'Parameter Magnitude Attack', 'Topology Structure Attack']:
                stats = attack_stats.get(attack_name, {})
                if isinstance(stats, dict) and 'mean' in stats:
                    row_data.append(stats['mean'])
                else:
                    row_data.append(0.0)
            
            heatmap_data.append(row_data)
        
        if not heatmap_data:
            return
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(heatmap_data, 
                   xticklabels=attack_names,
                   yticklabels=defense_names,
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn',
                   center=50,
                   cbar_kws={'label': 'Attack Success Degradation (%)'})
        
        plt.title('Defense Mechanism Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Attack Type', fontsize=12)
        plt.ylabel('Defense Mechanism', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'defense_performance_heatmap.png', dpi=300, bbox_inches='tight')
        self.logger.info("Generated defense performance heatmap")


def main():
    """Main function for running defense evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate topology-aware defense mechanisms')
    parser.add_argument('--data-dir', type=str, 
                       default='/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/experiments_archive/phase1_baseline_analysis',
                       help='Path to phase1 baseline analysis data')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/defense_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--max-experiments', type=int, default=30,
                       help='Maximum number of experiments to evaluate')
    parser.add_argument('--filter-topology', type=str, choices=['star', 'ring', 'complete', 'line'],
                       help='Filter experiments by topology')
    parser.add_argument('--filter-dataset', type=str, choices=['mnist', 'ham10000'],
                       help='Filter experiments by dataset')
    parser.add_argument('--filter-dp', type=str, choices=['no_dp', 'weak_dp', 'medium_dp', 'strong_dp'],
                       help='Filter experiments by DP level')
    
    args = parser.parse_args()
    
    # Build filter criteria
    filter_criteria = {}
    if args.filter_topology:
        filter_criteria['topology'] = args.filter_topology
    if args.filter_dataset:
        filter_criteria['dataset'] = args.filter_dataset
    if args.filter_dp:
        filter_criteria['dp_level'] = args.filter_dp
    
    # Run evaluation
    runner = DefenseExperimentRunner(args.data_dir, args.output_dir)
    
    try:
        results = runner.run_comprehensive_evaluation(
            max_experiments=args.max_experiments,
            filter_criteria=filter_criteria if filter_criteria else None
        )
        
        print("\nDefense Evaluation Complete!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Total experiments evaluated: {results['summary_stats']['successful_evaluations']}")
        print(f"Failed evaluations: {results['summary_stats']['failed_evaluations']}")
        
        # Print top performing defenses
        defense_effectiveness = results['summary_stats']['defense_effectiveness']
        if defense_effectiveness:
            print("\nTop Performing Defense Mechanisms:")
            sorted_defenses = sorted(
                [(name, stats['mean']) for name, stats in defense_effectiveness.items() if isinstance(stats, dict)],
                key=lambda x: x[1], reverse=True
            )
            
            for i, (defense_name, effectiveness) in enumerate(sorted_defenses[:5], 1):
                print(f"{i}. {defense_name}: {effectiveness:.3f} effectiveness")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())