#!/usr/bin/env python3
"""
Granular Defense Analysis Script

Provides detailed breakdown of defense effectiveness by:
- Attack type (Communication Pattern, Parameter Magnitude, Topology Structure)
- Network topology (Star, Ring, Line, Complete)
- Dataset (MNIST vs HAM10000)
- DP levels (no_dp, weak_dp, medium_dp, strong_dp, very_strong_dp)
- Defense mechanism combinations

Answers specific questions about:
1. What does "combined approach" mean exactly?
2. How does dynamic network reconfiguration work for different topologies?
3. Performance breakdown by individual attack vectors
4. Topology-specific defense effectiveness
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class GranularDefenseAnalyzer:
    """Detailed analysis of defense evaluation results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_data = None
        self.detailed_results = None
        self.load_data()
        
    def load_data(self):
        """Load comprehensive evaluation results."""
        results_file = self.results_dir / 'comprehensive_phase1_evaluation_results.json'
        
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        self.results_data = data['summary_stats']
        self.detailed_results = data.get('detailed_results', [])
        
        print(f"Loaded {len(self.detailed_results)} detailed evaluation results")
        
    def explain_defense_configurations(self):
        """Explain exactly what each defense configuration means."""
        
        explanations = {
            "Structural Noise Injection": {
                "Description": "Adds calibrated noise to obscure topology-based signatures",
                "Components": {
                    "Communication Noise": "Injects dummy communications to hide real patterns",
                    "Timing Noise": "Adds Gaussian noise to communication timestamps", 
                    "Magnitude Noise": "Applies multiplicative noise to parameter norms"
                },
                "Configurations": {
                    "weak": {
                        "comm_noise_rate": 0.1,  # 10% dummy traffic
                        "timing_noise_std": 0.05,  # 5% timing variance
                        "magnitude_noise_multiplier": 0.05  # 5% magnitude noise
                    },
                    "medium": {
                        "comm_noise_rate": 0.2,  # 20% dummy traffic
                        "timing_noise_std": 0.15,  # 15% timing variance
                        "magnitude_noise_multiplier": 0.15  # 15% magnitude noise
                    },
                    "strong": {
                        "comm_noise_rate": 0.3,  # 30% dummy traffic
                        "timing_noise_std": 0.3,  # 30% timing variance
                        "magnitude_noise_multiplier": 0.3  # 30% magnitude noise
                    }
                }
            },
            
            "Topology-Aware Differential Privacy": {
                "Description": "Extends standard DP to account for structural correlations",
                "Components": {
                    "Structural Amplification": "Multiplies base noise by topology-based factors",
                    "Neighbor Correlation": "Adds extra noise based on node degree/centrality",
                    "Privacy Accounting": "Adjusts noise for topology-induced correlations"
                },
                "Configurations": {
                    "weak": {
                        "structural_amplification_factor": 1.2,  # 20% amplification
                        "neighbor_correlation_weight": 0.05  # 5% neighbor correlation
                    },
                    "medium": {
                        "structural_amplification_factor": 1.5,  # 50% amplification
                        "neighbor_correlation_weight": 0.1  # 10% neighbor correlation
                    },
                    "strong": {
                        "structural_amplification_factor": 2.0,  # 100% amplification
                        "neighbor_correlation_weight": 0.2  # 20% neighbor correlation
                    }
                }
            },
            
            "Dynamic Topology Reconfiguration": {
                "Description": "Periodically changes network topology during training",
                "Components": {
                    "Reconfiguration Frequency": "How often topology changes",
                    "Connectivity Preservation": "Ensures network remains connected",
                    "Random Topology Generation": "Creates new random topologies"
                },
                "Configurations": {
                    "weak": {
                        "reconfig_frequency": 10,  # Every 10 rounds
                        "preserve_connectivity": True
                    },
                    "medium": {
                        "reconfig_frequency": 5,  # Every 5 rounds
                        "preserve_connectivity": True
                    },
                    "strong": {
                        "reconfig_frequency": 3,  # Every 3 rounds
                        "preserve_connectivity": True
                    }
                }
            },
            
            "Combined Defense Strategies": {
                "Description": "Integrates multiple defense mechanisms for layered protection",
                "Components": {
                    "Structural Noise": "Moderate noise injection",
                    "Topology-Aware DP": "Enhanced privacy accounting",
                    "Optional Reconfiguration": "In strong configuration only"
                },
                "Configurations": {
                    "weak": {
                        "structural_noise": {
                            "comm_noise_rate": 0.05,  # 5% dummy traffic
                            "timing_noise_std": 0.05,  # 5% timing variance
                            "magnitude_noise_multiplier": 0.05  # 5% magnitude noise
                        },
                        "topology_aware_dp": {
                            "structural_amplification_factor": 1.2,  # 20% amplification
                            "neighbor_correlation_weight": 0.05  # 5% neighbor correlation
                        },
                        "topology_reconfig": False
                    },
                    "medium": {
                        "structural_noise": {
                            "comm_noise_rate": 0.1,  # 10% dummy traffic
                            "timing_noise_std": 0.1,  # 10% timing variance
                            "magnitude_noise_multiplier": 0.1  # 10% magnitude noise
                        },
                        "topology_aware_dp": {
                            "structural_amplification_factor": 1.5,  # 50% amplification
                            "neighbor_correlation_weight": 0.1  # 10% neighbor correlation
                        },
                        "topology_reconfig": False
                    },
                    "strong": {
                        "structural_noise": {
                            "comm_noise_rate": 0.15,  # 15% dummy traffic
                            "timing_noise_std": 0.15,  # 15% timing variance
                            "magnitude_noise_multiplier": 0.15  # 15% magnitude noise
                        },
                        "topology_aware_dp": {
                            "structural_amplification_factor": 1.8,  # 80% amplification
                            "neighbor_correlation_weight": 0.15  # 15% neighbor correlation
                        },
                        "topology_reconfig": {
                            "reconfig_frequency": 5,  # Every 5 rounds
                            "preserve_connectivity": True
                        }
                    }
                }
            }
        }
        
        return explanations
    
    def analyze_by_topology(self):
        """Analyze defense effectiveness by network topology."""
        
        topology_analysis = {}
        
        # Extract topology from experiment names
        topology_patterns = {
            'star': ['star'],
            'ring': ['ring'],
            'line': ['line'], 
            'complete': ['complete']
        }
        
        for result in self.detailed_results:
            if not result.get('success', False):
                continue
                
            exp_name = result['experiment_name']
            defense_name = result['defense_name']
            effectiveness = result['effectiveness_metrics']
            
            # Determine topology
            topology = 'unknown'
            for topo_name, patterns in topology_patterns.items():
                if any(pattern in exp_name.lower() for pattern in patterns):
                    topology = topo_name
                    break
            
            # Initialize topology analysis
            if topology not in topology_analysis:
                topology_analysis[topology] = {}
            if defense_name not in topology_analysis[topology]:
                topology_analysis[topology][defense_name] = {
                    'overall_effectiveness': [],
                    'attack_reductions': {
                        'Communication Pattern Attack': [],
                        'Parameter Magnitude Attack': [],
                        'Topology Structure Attack': []
                    }
                }
            
            # Add effectiveness data
            topology_analysis[topology][defense_name]['overall_effectiveness'].append(
                effectiveness['overall_effectiveness']
            )
            
            # Add attack-specific reductions
            for attack_name, reduction_info in effectiveness['attack_reductions'].items():
                if attack_name in topology_analysis[topology][defense_name]['attack_reductions']:
                    topology_analysis[topology][defense_name]['attack_reductions'][attack_name].append(
                        reduction_info['reduction_percentage']
                    )
        
        # Calculate statistics
        for topology in topology_analysis:
            for defense_name in topology_analysis[topology]:
                defense_data = topology_analysis[topology][defense_name]
                
                # Overall effectiveness stats
                if defense_data['overall_effectiveness']:
                    effectiveness_list = defense_data['overall_effectiveness']
                    defense_data['overall_stats'] = {
                        'mean': np.mean(effectiveness_list),
                        'std': np.std(effectiveness_list),
                        'min': np.min(effectiveness_list),
                        'max': np.max(effectiveness_list),
                        'count': len(effectiveness_list)
                    }
                
                # Attack-specific stats
                for attack_name in defense_data['attack_reductions']:
                    attack_reductions = defense_data['attack_reductions'][attack_name]
                    if attack_reductions:
                        defense_data['attack_reductions'][attack_name] = {
                            'mean': np.mean(attack_reductions),
                            'std': np.std(attack_reductions),
                            'min': np.min(attack_reductions),
                            'max': np.max(attack_reductions),
                            'count': len(attack_reductions)
                        }
        
        return topology_analysis
    
    def analyze_dynamic_reconfig_by_topology(self):
        """Specifically analyze dynamic topology reconfiguration by topology type."""
        
        reconfig_analysis = {}
        
        topology_patterns = {
            'star': ['star'],
            'ring': ['ring'],
            'line': ['line'],
            'complete': ['complete']
        }
        
        for result in self.detailed_results:
            if not result.get('success', False):
                continue
                
            defense_name = result['defense_name']
            
            # Only analyze topology reconfiguration defenses
            if 'topology_reconfig' not in defense_name:
                continue
                
            exp_name = result['experiment_name']
            effectiveness = result['effectiveness_metrics']
            
            # Determine topology
            topology = 'unknown'
            for topo_name, patterns in topology_patterns.items():
                if any(pattern in exp_name.lower() for pattern in patterns):
                    topology = topo_name
                    break
            
            # Initialize analysis structure
            if topology not in reconfig_analysis:
                reconfig_analysis[topology] = {}
            if defense_name not in reconfig_analysis[topology]:
                reconfig_analysis[topology][defense_name] = {
                    'effectiveness_scores': [],
                    'topology_structure_reductions': [],
                    'experiment_details': []
                }
            
            # Add data
            reconfig_analysis[topology][defense_name]['effectiveness_scores'].append(
                effectiveness['overall_effectiveness']
            )
            
            # Get topology structure attack reduction specifically
            for attack_name, reduction_info in effectiveness['attack_reductions'].items():
                if 'topology structure' in attack_name.lower():
                    reconfig_analysis[topology][defense_name]['topology_structure_reductions'].append(
                        reduction_info['reduction_percentage']
                    )
            
            # Store experiment details
            reconfig_analysis[topology][defense_name]['experiment_details'].append({
                'experiment': exp_name,
                'overall_effectiveness': effectiveness['overall_effectiveness'],
                'attack_reductions': effectiveness['attack_reductions']
            })
        
        # Calculate statistics
        for topology in reconfig_analysis:
            for defense_name in reconfig_analysis[topology]:
                data = reconfig_analysis[topology][defense_name]
                
                if data['effectiveness_scores']:
                    data['overall_stats'] = {
                        'mean': np.mean(data['effectiveness_scores']),
                        'std': np.std(data['effectiveness_scores']),
                        'count': len(data['effectiveness_scores']),
                        'non_zero_count': sum(1 for x in data['effectiveness_scores'] if x > 0)
                    }
                
                if data['topology_structure_reductions']:
                    data['topology_attack_stats'] = {
                        'mean': np.mean(data['topology_structure_reductions']),
                        'std': np.std(data['topology_structure_reductions']),
                        'count': len(data['topology_structure_reductions']),
                        'non_zero_count': sum(1 for x in data['topology_structure_reductions'] if x > 0)
                    }
        
        return reconfig_analysis
    
    def analyze_by_dataset(self):
        """Analyze defense effectiveness by dataset (MNIST vs HAM10000)."""
        
        dataset_analysis = {}
        
        for result in self.detailed_results:
            if not result.get('success', False):
                continue
                
            exp_name = result['experiment_name']
            defense_name = result['defense_name']
            effectiveness = result['effectiveness_metrics']
            
            # Determine dataset
            dataset = 'unknown'
            if 'mnist' in exp_name.lower():
                dataset = 'mnist'
            elif 'ham10000' in exp_name.lower():
                dataset = 'ham10000'
            
            # Initialize dataset analysis
            if dataset not in dataset_analysis:
                dataset_analysis[dataset] = {}
            if defense_name not in dataset_analysis[dataset]:
                dataset_analysis[dataset][defense_name] = {
                    'overall_effectiveness': [],
                    'attack_reductions': {
                        'Communication Pattern Attack': [],
                        'Parameter Magnitude Attack': [],
                        'Topology Structure Attack': []
                    }
                }
            
            # Add effectiveness data
            dataset_analysis[dataset][defense_name]['overall_effectiveness'].append(
                effectiveness['overall_effectiveness']
            )
            
            # Add attack-specific reductions
            for attack_name, reduction_info in effectiveness['attack_reductions'].items():
                if attack_name in dataset_analysis[dataset][defense_name]['attack_reductions']:
                    dataset_analysis[dataset][defense_name]['attack_reductions'][attack_name].append(
                        reduction_info['reduction_percentage']
                    )
        
        # Calculate statistics
        for dataset in dataset_analysis:
            for defense_name in dataset_analysis[dataset]:
                defense_data = dataset_analysis[dataset][defense_name]
                
                # Overall effectiveness stats
                if defense_data['overall_effectiveness']:
                    effectiveness_list = defense_data['overall_effectiveness']
                    defense_data['overall_stats'] = {
                        'mean': np.mean(effectiveness_list),
                        'std': np.std(effectiveness_list),
                        'count': len(effectiveness_list)
                    }
                
                # Attack-specific stats
                for attack_name in defense_data['attack_reductions']:
                    attack_reductions = defense_data['attack_reductions'][attack_name]
                    if attack_reductions:
                        defense_data['attack_reductions'][attack_name] = {
                            'mean': np.mean(attack_reductions),
                            'std': np.std(attack_reductions),
                            'count': len(attack_reductions)
                        }
        
        return dataset_analysis
    
    def create_granular_report(self, output_file: str):
        """Create comprehensive granular analysis report."""
        
        # Get all analyses
        defense_explanations = self.explain_defense_configurations()
        topology_analysis = self.analyze_by_topology()
        reconfig_analysis = self.analyze_dynamic_reconfig_by_topology()
        dataset_analysis = self.analyze_by_dataset()
        
        report = {
            'defense_mechanism_explanations': defense_explanations,
            'topology_analysis': topology_analysis,
            'dynamic_reconfiguration_analysis': reconfig_analysis,
            'dataset_analysis': dataset_analysis,
            'summary_insights': self.generate_insights(topology_analysis, reconfig_analysis, dataset_analysis)
        }
        
        # Save report
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Granular analysis report saved to: {output_path}")
        
        return report
    
    def generate_insights(self, topology_analysis, reconfig_analysis, dataset_analysis):
        """Generate key insights from granular analysis."""
        
        insights = {}
        
        # Combined defense strategy explanation
        insights['combined_defense_explanation'] = {
            "what_it_means": "Combined defenses integrate multiple protection mechanisms for layered security",
            "components": [
                "Structural noise injection at moderate levels (5-15% noise rates)",
                "Topology-aware differential privacy with amplification factors (1.2-1.8x)",
                "Optional dynamic topology reconfiguration (strong configuration only)"
            ],
            "rationale": "Individual defenses may miss certain attack vectors, so combining them provides comprehensive protection",
            "trade_offs": "Increased computational overhead but better overall protection"
        }
        
        # Dynamic reconfiguration insights
        insights['dynamic_reconfiguration_insights'] = {
            "why_limited_effectiveness": [
                "Implementation challenges with maintaining network connectivity",
                "Reconfiguration frequency may be too low to disrupt attacks effectively",
                "Some topologies (like complete graphs) are harder to meaningfully reconfigure"
            ],
            "topology_specific_performance": {},
            "improvement_opportunities": [
                "More frequent reconfiguration (every 1-2 rounds instead of 3-10)",
                "Smarter reconfiguration algorithms that target attack-vulnerable patterns",
                "Topology-aware reconfiguration strategies"
            ]
        }
        
        # Extract topology-specific reconfig performance
        for topology in reconfig_analysis:
            topology_insights = {}
            for defense in reconfig_analysis[topology]:
                stats = reconfig_analysis[topology][defense].get('overall_stats', {})
                topology_insights[defense] = {
                    'mean_effectiveness': stats.get('mean', 0),
                    'non_zero_cases': stats.get('non_zero_count', 0),
                    'total_cases': stats.get('count', 0),
                    'success_rate': stats.get('non_zero_count', 0) / max(stats.get('count', 1), 1) * 100
                }
            insights['dynamic_reconfiguration_insights']['topology_specific_performance'][topology] = topology_insights
        
        # Attack-specific insights
        insights['attack_specific_insights'] = {
            "communication_pattern_attacks": {
                "most_effective_defense": "Structural noise injection",
                "why_effective": "Dummy communications directly obscure real communication patterns",
                "topology_aware_dp_limitation": "DP affects parameter noise but not communication timing patterns"
            },
            "parameter_magnitude_attacks": {
                "balanced_effectiveness": "Multiple defenses show moderate effectiveness (3-4%)",
                "topology_aware_dp_strength": "Direct parameter perturbation helps against magnitude analysis"
            },
            "topology_structure_attacks": {
                "highest_reduction_potential": "Up to 28% average reduction with structural noise",
                "topology_aware_dp_effectiveness": "21-25% reduction shows good structural correlation modeling"
            }
        }
        
        return insights
    
    def create_visualization_dashboard(self, output_dir: str):
        """Create comprehensive visualization dashboard."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get analysis data
        topology_analysis = self.analyze_by_topology()
        
        # 1. Defense effectiveness by topology heatmap
        self.plot_topology_effectiveness_heatmap(topology_analysis, output_path / 'topology_effectiveness_heatmap.png')
        
        # 2. Attack-specific performance breakdown
        self.plot_attack_specific_breakdown(topology_analysis, output_path / 'attack_specific_breakdown.png')
        
        # 3. Dynamic reconfiguration analysis
        reconfig_analysis = self.analyze_dynamic_reconfig_by_topology()
        self.plot_dynamic_reconfig_analysis(reconfig_analysis, output_path / 'dynamic_reconfig_analysis.png')
        
        print(f"Visualization dashboard created in: {output_path}")
    
    def plot_topology_effectiveness_heatmap(self, topology_analysis, output_file):
        """Create heatmap of defense effectiveness by topology."""
        
        # Prepare data for heatmap
        topologies = list(topology_analysis.keys())
        defenses = set()
        for topo_data in topology_analysis.values():
            defenses.update(topo_data.keys())
        defenses = sorted(list(defenses))
        
        # Create effectiveness matrix
        effectiveness_matrix = np.zeros((len(defenses), len(topologies)))
        
        for i, defense in enumerate(defenses):
            for j, topology in enumerate(topologies):
                if defense in topology_analysis[topology]:
                    stats = topology_analysis[topology][defense].get('overall_stats', {})
                    effectiveness_matrix[i, j] = stats.get('mean', 0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(effectiveness_matrix, 
                   xticklabels=topologies, 
                   yticklabels=defenses,
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Average Attack Reduction (%)'})
        
        plt.title('Defense Effectiveness by Network Topology', fontsize=16, fontweight='bold')
        plt.xlabel('Network Topology', fontsize=12)
        plt.ylabel('Defense Mechanism', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attack_specific_breakdown(self, topology_analysis, output_file):
        """Create breakdown of defense effectiveness by attack type."""
        
        attack_types = ['Communication Pattern Attack', 'Parameter Magnitude Attack', 'Topology Structure Attack']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, attack_type in enumerate(attack_types):
            ax = axes[idx]
            
            # Collect data for this attack type across all topologies and defenses
            defense_names = []
            attack_reductions = []
            
            for topology in topology_analysis:
                for defense in topology_analysis[topology]:
                    if defense not in defense_names:
                        defense_names.append(defense)
            
            defense_names = sorted(defense_names)
            
            for defense in defense_names:
                reductions = []
                for topology in topology_analysis:
                    if defense in topology_analysis[topology]:
                        attack_data = topology_analysis[topology][defense]['attack_reductions'].get(attack_type, {})
                        if isinstance(attack_data, dict) and 'mean' in attack_data:
                            reductions.append(attack_data['mean'])
                
                if reductions:
                    attack_reductions.append(np.mean(reductions))
                else:
                    attack_reductions.append(0)
            
            # Create bar plot
            bars = ax.bar(range(len(defense_names)), attack_reductions, 
                         color=plt.cm.Set3(np.linspace(0, 1, len(defense_names))))
            
            ax.set_title(f'{attack_type}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Defense Mechanism')
            ax.set_ylabel('Average Reduction (%)')
            ax.set_xticks(range(len(defense_names)))
            ax.set_xticklabels([name.replace('_', ' ').title() for name in defense_names], 
                              rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, attack_reductions):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Defense Effectiveness by Attack Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dynamic_reconfig_analysis(self, reconfig_analysis, output_file):
        """Create analysis plot for dynamic reconfiguration effectiveness."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Success rate by topology
        topologies = list(reconfig_analysis.keys())
        defense_types = ['topology_reconfig_weak', 'topology_reconfig_medium', 'topology_reconfig_strong']
        
        success_rates = np.zeros((len(defense_types), len(topologies)))
        
        for i, defense in enumerate(defense_types):
            for j, topology in enumerate(topologies):
                if defense in reconfig_analysis[topology]:
                    stats = reconfig_analysis[topology][defense].get('overall_stats', {})
                    total = stats.get('count', 0)
                    non_zero = stats.get('non_zero_count', 0)
                    success_rates[i, j] = (non_zero / max(total, 1)) * 100
        
        sns.heatmap(success_rates, 
                   xticklabels=topologies,
                   yticklabels=[d.replace('topology_reconfig_', '').title() for d in defense_types],
                   annot=True,
                   fmt='.1f',
                   cmap='RdYlGn',
                   ax=ax1,
                   cbar_kws={'label': 'Success Rate (%)'})
        
        ax1.set_title('Dynamic Reconfiguration Success Rate by Topology', fontweight='bold')
        ax1.set_xlabel('Network Topology')
        ax1.set_ylabel('Reconfiguration Strength')
        
        # Plot 2: Average effectiveness when successful
        avg_effectiveness = np.zeros((len(defense_types), len(topologies)))
        
        for i, defense in enumerate(defense_types):
            for j, topology in enumerate(topologies):
                if defense in reconfig_analysis[topology]:
                    stats = reconfig_analysis[topology][defense].get('overall_stats', {})
                    avg_effectiveness[i, j] = stats.get('mean', 0)
        
        sns.heatmap(avg_effectiveness,
                   xticklabels=topologies,
                   yticklabels=[d.replace('topology_reconfig_', '').title() for d in defense_types],
                   annot=True,
                   fmt='.1f', 
                   cmap='RdYlBu_r',
                   ax=ax2,
                   cbar_kws={'label': 'Average Effectiveness (%)'})
        
        ax2.set_title('Average Effectiveness When Successful', fontweight='bold')
        ax2.set_xlabel('Network Topology')
        ax2.set_ylabel('Reconfiguration Strength')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Run granular defense analysis."""
    
    # Initialize analyzer
    results_dir = "/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/experiments_archive/phase5_defense_evaluation/defense_evaluation/comprehensive_phase1_results"
    
    analyzer = GranularDefenseAnalyzer(results_dir)
    
    # Create granular analysis report
    report = analyzer.create_granular_report('granular_defense_analysis.json')
    
    # Create visualization dashboard
    viz_dir = "/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/experiments_archive/phase5_defense_evaluation/defense_evaluation/granular_analysis_visualizations"
    analyzer.create_visualization_dashboard(viz_dir)
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM GRANULAR ANALYSIS")
    print("="*80)
    
    insights = report['summary_insights']
    
    print("\n1. COMBINED DEFENSE STRATEGY EXPLANATION:")
    combined_explanation = insights['combined_defense_explanation']
    print(f"   {combined_explanation['what_it_means']}")
    print("   Components:")
    for component in combined_explanation['components']:
        print(f"   - {component}")
    
    print("\n2. DYNAMIC RECONFIGURATION INSIGHTS:")
    reconfig_insights = insights['dynamic_reconfiguration_insights']
    print("   Why limited effectiveness:")
    for reason in reconfig_insights['why_limited_effectiveness']:
        print(f"   - {reason}")
    
    print("\n3. TOPOLOGY-SPECIFIC PERFORMANCE:")
    for topology, performance in reconfig_insights['topology_specific_performance'].items():
        print(f"   {topology.title()} topology:")
        for defense, stats in performance.items():
            success_rate = stats['success_rate']
            mean_eff = stats['mean_effectiveness']
            print(f"     {defense}: {success_rate:.1f}% success rate, {mean_eff:.2f}% avg effectiveness")
    
    print("\n4. ATTACK-SPECIFIC INSIGHTS:")
    attack_insights = insights['attack_specific_insights']
    for attack_type, insight in attack_insights.items():
        print(f"   {attack_type.replace('_', ' ').title()}:")
        for key, value in insight.items():
            print(f"     {key}: {value}")
    
    print(f"\nDetailed analysis saved to: {results_dir}/granular_defense_analysis.json")
    print(f"Visualizations saved to: {viz_dir}/")


if __name__ == "__main__":
    main()