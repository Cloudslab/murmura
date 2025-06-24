#!/usr/bin/env python3
"""
Comprehensive defense mechanism evaluation across multiple experiments.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Add murmura to path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from defense_mechanisms import (
    DefenseConfig, StructuralNoiseInjection, TopologyAwareDifferentialPrivacy, 
    DefenseEvaluator
)
from murmura.attacks.topology_attacks import (
    CommunicationPatternAttack, ParameterMagnitudeAttack, TopologyStructureAttack
)


class ComprehensiveDefenseTest:
    """Comprehensive test of defense mechanisms."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.attacks = [
            CommunicationPatternAttack(),
            ParameterMagnitudeAttack(), 
            TopologyStructureAttack()
        ]
        
        # Define defense configurations to test
        self.defense_configs = {
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
            'combined_weak': DefenseConfig(
                enable_comm_noise=True, comm_noise_rate=0.05,
                enable_timing_noise=True, timing_noise_std=0.05,
                enable_magnitude_noise=True, magnitude_noise_multiplier=0.05,
                enable_topology_reconfig=False, enable_topology_aware_dp=True,
                structural_amplification_factor=1.2, neighbor_correlation_weight=0.05
            ),
            'combined_strong': DefenseConfig(
                enable_comm_noise=True, comm_noise_rate=0.15,
                enable_timing_noise=True, timing_noise_std=0.15,
                enable_magnitude_noise=True, magnitude_noise_multiplier=0.15,
                enable_topology_reconfig=False, enable_topology_aware_dp=True,
                structural_amplification_factor=1.8, neighbor_correlation_weight=0.15
            )
        }
    
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
                    print(f"Failed to load {filename}: {e}")
        
        return data
    
    def run_attacks_on_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run topology attacks on data."""
        attack_results = []
        
        for attack in self.attacks:
            try:
                result = attack.execute_attack(data)
                result['attack_name'] = attack.name
                attack_results.append(result)
            except Exception as e:
                attack_results.append({
                    'attack_name': attack.name,
                    'error': str(e),
                    'attack_success_metric': 0.0
                })
        
        return {'attack_results': attack_results}
    
    def apply_defense(self, data: Dict[str, pd.DataFrame], config: DefenseConfig) -> Dict[str, pd.DataFrame]:
        """Apply defense mechanism to data."""
        defended_data = data.copy()
        
        # Apply structural noise injection
        if config.enable_comm_noise or config.enable_timing_noise or config.enable_magnitude_noise:
            noise_defense = StructuralNoiseInjection(config)
            defended_data = noise_defense.apply_defense(defended_data)
        
        # Apply topology-aware DP
        if config.enable_topology_aware_dp:
            dp_defense = TopologyAwareDifferentialPrivacy(config)
            defended_data = dp_defense.apply_defense(defended_data)
        
        return defended_data
    
    def test_single_experiment(self, experiment_dir: Path) -> Dict[str, Any]:
        """Test all defenses on a single experiment."""
        experiment_name = experiment_dir.name
        print(f"\nTesting {experiment_name}")
        
        # Load original data
        original_data = self.load_experiment_data(experiment_dir)
        
        if not original_data:
            return None
        
        # Run attacks on original data
        original_results = self.run_attacks_on_data(original_data)
        
        # Test each defense configuration
        defense_results = {}
        
        for defense_name, config in self.defense_configs.items():
            try:
                # Apply defense
                defended_data = self.apply_defense(original_data, config)
                
                # Run attacks on defended data
                defense_attack_results = self.run_attacks_on_data(defended_data)
                
                defense_results[defense_name] = defense_attack_results
                
            except Exception as e:
                print(f"  Defense {defense_name} failed: {e}")
                defense_results[defense_name] = {'attack_results': []}
        
        return {
            'experiment_name': experiment_name,
            'original_results': original_results,
            'defense_results': defense_results,
            'data_summary': {k: len(v) for k, v in original_data.items()}
        }
    
    def run_comprehensive_test(self, max_experiments: int = 15, filter_pattern: str = None) -> Dict[str, Any]:
        """Run comprehensive test across multiple experiments."""
        print("Starting comprehensive defense evaluation...")
        
        # Find experiment directories
        training_data_dir = self.data_dir / 'training_data'
        experiment_dirs = [d for d in training_data_dir.iterdir() if d.is_dir()]
        
        # Apply filter if specified
        if filter_pattern:
            experiment_dirs = [d for d in experiment_dirs if filter_pattern in d.name]
        
        # Limit number of experiments
        experiment_dirs = experiment_dirs[:max_experiments]
        
        print(f"Testing {len(experiment_dirs)} experiments with {len(self.defense_configs)} defenses")
        
        all_results = []
        summary_stats = {
            'total_experiments': len(experiment_dirs),
            'successful_tests': 0,
            'failed_tests': 0,
            'defense_effectiveness': {},
            'attack_reductions': {}
        }
        
        for exp_dir in experiment_dirs:
            try:
                result = self.test_single_experiment(exp_dir)
                if result:
                    all_results.append(result)
                    summary_stats['successful_tests'] += 1
                    self._update_summary_stats(summary_stats, result)
                else:
                    summary_stats['failed_tests'] += 1
            except Exception as e:
                print(f"Failed to test {exp_dir.name}: {e}")
                summary_stats['failed_tests'] += 1
        
        # Calculate final statistics
        self._finalize_summary_stats(summary_stats)
        
        # Create comprehensive results
        comprehensive_results = {
            'summary_stats': summary_stats,
            'detailed_results': all_results,
            'defense_configs': {k: str(v.__dict__) for k, v in self.defense_configs.items()}
        }
        
        # Save results
        self._save_results(comprehensive_results)
        
        # Generate visualizations
        self._generate_visualizations(comprehensive_results)
        
        return comprehensive_results
    
    def _update_summary_stats(self, summary_stats: Dict[str, Any], result: Dict[str, Any]):
        """Update summary statistics with single experiment result."""
        original_results = result['original_results']['attack_results']
        defense_results = result['defense_results']
        
        # Initialize structures if needed
        for defense_name in defense_results.keys():
            if defense_name not in summary_stats['defense_effectiveness']:
                summary_stats['defense_effectiveness'][defense_name] = []
                summary_stats['attack_reductions'][defense_name] = {
                    'Communication Pattern Attack': [],
                    'Parameter Magnitude Attack': [],
                    'Topology Structure Attack': []
                }
        
        # Calculate effectiveness for each defense
        for defense_name, defense_attack_results in defense_results.items():
            if 'attack_results' not in defense_attack_results:
                continue
                
            total_reduction = 0.0
            valid_attacks = 0
            
            for orig_attack, def_attack in zip(original_results, defense_attack_results['attack_results']):
                attack_name = orig_attack.get('attack_name', 'Unknown')
                orig_success = orig_attack.get('attack_success_metric', 0.0)
                def_success = def_attack.get('attack_success_metric', 0.0)
                
                if orig_success > 0:
                    reduction_pct = max(0.0, (orig_success - def_success) / orig_success) * 100
                    total_reduction += reduction_pct
                    valid_attacks += 1
                    
                    # Store individual attack reductions
                    if attack_name in summary_stats['attack_reductions'][defense_name]:
                        summary_stats['attack_reductions'][defense_name][attack_name].append(reduction_pct)
            
            # Calculate average effectiveness for this defense on this experiment
            if valid_attacks > 0:
                avg_effectiveness = total_reduction / valid_attacks
                summary_stats['defense_effectiveness'][defense_name].append(avg_effectiveness)
    
    def _finalize_summary_stats(self, summary_stats: Dict[str, Any]):
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
        
        # Calculate average attack reductions
        for defense_name, attack_reductions in summary_stats['attack_reductions'].items():
            for attack_name, reduction_list in attack_reductions.items():
                if reduction_list:
                    summary_stats['attack_reductions'][defense_name][attack_name] = {
                        'mean': np.mean(reduction_list),
                        'std': np.std(reduction_list),
                        'min': np.min(reduction_list),
                        'max': np.max(reduction_list),
                        'count': len(reduction_list)
                    }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
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
        
        with open(self.output_dir / 'comprehensive_defense_results.json', 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"Results saved to: {self.output_dir / 'comprehensive_defense_results.json'}")
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualization plots."""
        summary_stats = results['summary_stats']
        
        # 1. Defense Effectiveness Bar Chart
        plt.figure(figsize=(12, 8))
        
        defense_names = []
        effectiveness_means = []
        effectiveness_stds = []
        
        for defense_name, stats in summary_stats['defense_effectiveness'].items():
            if isinstance(stats, dict) and 'mean' in stats:
                defense_names.append(defense_name.replace('_', ' ').title())
                effectiveness_means.append(stats['mean'])
                effectiveness_stds.append(stats['std'])
        
        if defense_names:
            bars = plt.bar(defense_names, effectiveness_means, yerr=effectiveness_stds, 
                          capsize=5, alpha=0.7, color='steelblue')
            plt.title('Defense Mechanism Effectiveness\n(Average Attack Success Reduction)', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Defense Mechanism', fontsize=12)
            plt.ylabel('Attack Success Reduction (%)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, effectiveness_means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'defense_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Attack-specific reduction heatmap
        attack_reductions = summary_stats['attack_reductions']
        
        if attack_reductions:
            # Prepare data for heatmap
            defense_names = list(attack_reductions.keys())
            attack_names = ['Communication Pattern', 'Parameter Magnitude', 'Topology Structure']
            
            heatmap_data = []
            for defense_name in defense_names:
                row_data = []
                for attack_full_name in ['Communication Pattern Attack', 'Parameter Magnitude Attack', 'Topology Structure Attack']:
                    attack_stats = attack_reductions[defense_name].get(attack_full_name, {})
                    if isinstance(attack_stats, dict) and 'mean' in attack_stats:
                        row_data.append(attack_stats['mean'])
                    else:
                        row_data.append(0.0)
                heatmap_data.append(row_data)
            
            if heatmap_data:
                plt.figure(figsize=(10, 8))
                sns.heatmap(heatmap_data,
                           xticklabels=attack_names,
                           yticklabels=[name.replace('_', ' ').title() for name in defense_names],
                           annot=True,
                           fmt='.1f',
                           cmap='RdYlGn',
                           center=25,
                           cbar_kws={'label': 'Attack Success Reduction (%)'})
                
                plt.title('Defense Mechanism Performance by Attack Type', 
                         fontsize=16, fontweight='bold')
                plt.xlabel('Attack Type', fontsize=12)
                plt.ylabel('Defense Mechanism', fontsize=12)
                plt.tight_layout()
                plt.savefig(self.output_dir / 'defense_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Visualizations saved to: {self.output_dir}")


def main():
    """Main function."""
    data_dir = '/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/experiments_archive/phase1_baseline_analysis'
    output_dir = '/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/comprehensive_defense_results'
    
    tester = ComprehensiveDefenseTest(data_dir, output_dir)
    
    # Run comprehensive test on MNIST star topology experiments
    results = tester.run_comprehensive_test(
        max_experiments=15, 
        filter_pattern='mnist_federated_star'
    )
    
    print(f"\n=== COMPREHENSIVE DEFENSE EVALUATION SUMMARY ===")
    print(f"Total experiments tested: {results['summary_stats']['successful_tests']}")
    print(f"Failed tests: {results['summary_stats']['failed_tests']}")
    
    # Print top performing defenses
    defense_effectiveness = results['summary_stats']['defense_effectiveness']
    if defense_effectiveness:
        print(f"\nTop Performing Defense Mechanisms:")
        sorted_defenses = sorted(
            [(name, stats['mean']) for name, stats in defense_effectiveness.items() 
             if isinstance(stats, dict)],
            key=lambda x: x[1], reverse=True
        )
        
        for i, (defense_name, effectiveness) in enumerate(sorted_defenses, 1):
            print(f"{i}. {defense_name.replace('_', ' ').title()}: {effectiveness:.1f}% avg reduction")


if __name__ == '__main__':
    main()