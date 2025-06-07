#!/usr/bin/env python3
"""
Comprehensive Paper Experiments - Large-scale systematic evaluation of topology-based attacks
for federated learning paper with up to 25 nodes across MNIST and skin lesion datasets.
"""

import os
import sys
import json
import subprocess
import argparse
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Add murmura to path
sys.path.append('.')
from murmura.attacks.topology_attacks import run_topology_attacks


class PaperExperimentRunner:
    """Comprehensive experiment runner for topology attack paper."""
    
    def __init__(self, base_output_dir: str = "./paper_experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_output_dir / "visualizations").mkdir(exist_ok=True)
        (self.base_output_dir / "results").mkdir(exist_ok=True)
        (self.base_output_dir / "logs").mkdir(exist_ok=True)
        (self.base_output_dir / "analysis").mkdir(exist_ok=True)
        
        self.results = []
        self.experiment_id = 0
        
        # Setup logging
        self.setup_logging()
        
        # Define compatibility matrix based on actual example script choices
        self.compatibility_matrix = {
            # Centralized FL (FedAvg/TrimmedMean): star, ring, complete, line
            "federated": {
                "compatible_topologies": ["star", "ring", "complete", "line"],
                "strategies": ["fedavg", "trimmed_mean"]
            },
            # Decentralized FL (GossipAvg): ring, complete, line (NO star support)
            "decentralized": {
                "compatible_topologies": ["ring", "complete", "line"],
                "strategies": ["gossip_avg"]
            }
        }
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.base_output_dir / "logs" / f"paper_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_valid_configurations(self) -> List[Dict[str, Any]]:
        """Generate all valid experimental configurations."""
        
        configurations = []
        
        # Experimental parameters
        datasets = ["mnist", "skin_lesion"]
        attack_strategies = ["sensitive_groups", "topology_correlated", "imbalanced_sensitive"]
        node_counts = [5, 10, 15, 20, 25]  # Scale up to 25 nodes
        dp_settings = [
            {"enabled": False, "epsilon": None, "name": "no_dp"},
            {"enabled": True, "epsilon": 16.0, "name": "weak_dp"},
            {"enabled": True, "epsilon": 8.0, "name": "medium_dp"},
            {"enabled": True, "epsilon": 4.0, "name": "strong_dp"},
            {"enabled": True, "epsilon": 1.0, "name": "very_strong_dp"}
        ]
        
        config_id = 1
        
        for dataset in datasets:
            for attack_strategy in attack_strategies:
                for node_count in node_counts:
                    for dp_setting in dp_settings:
                        
                        # Test both federated and decentralized approaches
                        for fl_type in ["federated", "decentralized"]:
                            valid_topologies = self.compatibility_matrix[fl_type]["compatible_topologies"]
                            
                            for topology in valid_topologies:
                                # Skip very large networks for some topologies to manage compute
                                if topology in ["complete"] and node_count > 15:
                                    continue  # Complete graph gets expensive with many nodes
                                
                                config = {
                                    "config_id": config_id,
                                    "dataset": dataset,
                                    "attack_strategy": attack_strategy,
                                    "fl_type": fl_type,  # federated or decentralized
                                    "topology": topology,
                                    "node_count": node_count,
                                    "dp_setting": dp_setting,
                                    "expected_runtime": self._estimate_runtime(dataset, node_count, fl_type)
                                }
                                
                                configurations.append(config)
                                config_id += 1
        
        self.logger.info(f"Generated {len(configurations)} valid experimental configurations")
        return configurations
    
    def _estimate_runtime(self, dataset: str, node_count: int, fl_type: str) -> int:
        """Estimate runtime in seconds for resource planning."""
        
        base_time = {
            "mnist": {"federated": 60, "decentralized": 90},  # Base time for 5 nodes
            "skin_lesion": {"federated": 180, "decentralized": 240}  # Skin lesion takes longer
        }
        
        # Scale with node count (roughly linear)
        scaling_factor = node_count / 5.0
        estimated_time = base_time[dataset][fl_type] * scaling_factor
        
        return int(estimated_time)
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        
        self.experiment_id += 1
        experiment_name = f"exp_{self.experiment_id:04d}_{config['dataset']}_{config['fl_type']}_{config['topology']}_{config['node_count']}n_{config['dp_setting']['name']}"
        
        self.logger.info(f"🎯 Starting experiment {self.experiment_id}: {experiment_name}")
        self.logger.info(f"   Config: {config['attack_strategy']} attack, {config['node_count']} nodes, {config['dp_setting']['name']}")
        
        start_time = time.time()
        
        try:
            # Select appropriate example script
            script_map = {
                ("mnist", "federated"): "dp_mnist_example.py",
                ("mnist", "decentralized"): "dp_decentralized_mnist_example.py", 
                ("skin_lesion", "federated"): "dp_skin_lesion_example.py",
                ("skin_lesion", "decentralized"): "dp_decentralized_skin_lesion_example.py"
            }
            
            script = script_map[(config['dataset'], config['fl_type'])]
            
            # Build command
            cmd = self._build_command(script, config, experiment_name)
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config['expected_runtime'] * 3,  # 3x safety margin
                cwd="."
            )
            
            if result.returncode != 0:
                error_msg = f"Training failed: {result.stderr[-500:]}"  # Last 500 chars
                self.logger.error(f"   ❌ {error_msg}")
                return self._create_failed_result(config, experiment_name, error_msg, start_time)
            
            self.logger.info(f"   ✅ Training completed in {time.time() - start_time:.1f}s")
            
            # Run attacks
            viz_dir = self.base_output_dir / "visualizations" / experiment_name
            if viz_dir.exists():
                attack_results = run_topology_attacks(str(viz_dir))
                evaluation = self.evaluate_attack_comprehensive(attack_results, config)
                
                self.logger.info(f"   📊 Attack success: {'✅' if evaluation['attack_success'] else '❌'} "
                               f"(confidence: {evaluation['confidence_score']:.3f})")
                
                return self._create_success_result(config, experiment_name, attack_results, evaluation, start_time)
            else:
                error_msg = f"Visualization directory not found: {viz_dir}"
                return self._create_failed_result(config, experiment_name, error_msg, start_time)
                
        except subprocess.TimeoutExpired:
            error_msg = f"Experiment timed out after {config['expected_runtime'] * 3}s"
            self.logger.warning(f"   ⏰ {error_msg}")
            return self._create_failed_result(config, experiment_name, error_msg, start_time)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"   ❌ {error_msg}")
            return self._create_failed_result(config, experiment_name, error_msg, start_time)
    
    def _build_command(self, script: str, config: Dict[str, Any], experiment_name: str) -> List[str]:
        """Build command for running experiment."""
        
        cmd = [
            "python", f"murmura/examples/{script}",
            "--partition_strategy", config['attack_strategy'],
            "--topology", config['topology'],
            "--num_actors", str(config['node_count']),
            "--rounds", "3",  # Keep training fast for large-scale experiments
            "--epochs", "1",
            "--vis_dir", str(self.base_output_dir / "visualizations"),
        ]
        
        # Add DP settings
        if config['dp_setting']['enabled']:
            cmd.extend(["--enable_dp", "--target_epsilon", str(config['dp_setting']['epsilon'])])
        
        # Add FL-type specific settings
        if config['fl_type'] == "federated":
            cmd.extend(["--aggregation_strategy", "fedavg"])
        else:  # decentralized
            cmd.extend(["--aggregation_strategy", "gossip_avg"])
        
        # Dataset-specific adjustments
        if config['dataset'] == "skin_lesion":
            cmd.extend(["--image_size", "64"])  # Smaller images for faster training
        
        return cmd
    
    def _create_success_result(self, config: Dict[str, Any], experiment_name: str, 
                             attack_results: Dict[str, Any], evaluation: Dict[str, Any],
                             start_time: float) -> Dict[str, Any]:
        """Create result dictionary for successful experiment."""
        
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "config": config,
            "status": "success",
            "runtime_seconds": time.time() - start_time,
            "attack_results": attack_results,
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_failed_result(self, config: Dict[str, Any], experiment_name: str,
                            error_msg: str, start_time: float) -> Dict[str, Any]:
        """Create result dictionary for failed experiment."""
        
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "config": config,
            "status": "failed",
            "error": error_msg,
            "runtime_seconds": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def evaluate_attack_comprehensive(self, attack_results: Dict[str, Any], 
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive attack evaluation for paper analysis."""
        
        evaluation = {
            "attack_success": False,
            "confidence_score": 0.0,
            "attack_type_scores": {},
            "best_attack_type": None,
            "topology_specific_metrics": {},
            "dp_effectiveness_metrics": {},
            "scalability_metrics": {}
        }
        
        # Analyze each attack type
        attack_types = ["Communication Pattern Attack", "Parameter Magnitude Attack", "Topology Structure Attack"]
        
        for result in attack_results.get('attack_results', []):
            attack_name = result.get('attack_name', '')
            success_metric = result.get('attack_success_metric', 0.0)
            
            if attack_name in attack_types:
                evaluation['attack_type_scores'][attack_name] = success_metric
                
                if success_metric > evaluation['confidence_score']:
                    evaluation['confidence_score'] = success_metric
                    evaluation['best_attack_type'] = attack_name
        
        # Overall success threshold
        evaluation['attack_success'] = evaluation['confidence_score'] > 0.3
        
        # Add strategy-specific analysis
        evaluation.update(self._analyze_attack_strategy(attack_results, config))
        
        # Add topology-specific metrics
        evaluation['topology_specific_metrics'] = self._analyze_topology_effects(attack_results, config)
        
        # Add DP effectiveness metrics
        evaluation['dp_effectiveness_metrics'] = self._analyze_dp_effectiveness(attack_results, config)
        
        # Add scalability metrics
        evaluation['scalability_metrics'] = self._analyze_scalability_effects(attack_results, config)
        
        return evaluation
    
    def _analyze_attack_strategy(self, attack_results: Dict[str, Any], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attack success specific to the attack strategy."""
        
        strategy = config['attack_strategy']
        analysis = {"strategy_specific_findings": []}
        
        if strategy == "sensitive_groups":
            # Analyze group detection success
            for result in attack_results.get('attack_results', []):
                if result.get('attack_name') == 'Communication Pattern Attack':
                    if 'node_clusters' in result:
                        clusters = result['node_clusters']
                        n_clusters = len(set(clusters.values()))
                        analysis['strategy_specific_findings'].append(
                            f"Detected {n_clusters} distinct groups via communication patterns"
                        )
                        analysis['group_detection_success'] = n_clusters > 1
        
        elif strategy == "topology_correlated":
            # Analyze correlation detection
            for result in attack_results.get('attack_results', []):
                if result.get('attack_name') == 'Topology Structure Attack':
                    if 'correlations' in result:
                        correlations = result['correlations']
                        max_correlation = max([abs(v) for v in correlations.values()], default=0.0)
                        analysis['max_correlation'] = max_correlation
                        analysis['strategy_specific_findings'].append(
                            f"Maximum topology correlation: {max_correlation:.3f}"
                        )
        
        elif strategy == "imbalanced_sensitive":
            # Analyze imbalance detection
            for result in attack_results.get('attack_results', []):
                if result.get('attack_name') == 'Parameter Magnitude Attack':
                    success_metric = result.get('attack_success_metric', 0.0)
                    analysis['imbalance_detection_score'] = success_metric
                    analysis['strategy_specific_findings'].append(
                        f"Imbalance detection score: {success_metric:.3f}"
                    )
        
        return analysis
    
    def _analyze_topology_effects(self, attack_results: Dict[str, Any], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how topology affects attack success."""
        
        topology = config['topology']
        node_count = config['node_count']
        
        metrics = {
            "topology": topology,
            "node_count": node_count,
            "topology_vulnerability_score": 0.0,
            "communication_complexity": self._calculate_communication_complexity(topology, node_count)
        }
        
        # Calculate topology-specific vulnerability
        overall_eval = attack_results.get('evaluation', {})
        max_signal = overall_eval.get('attack_indicators', {}).get('max_signal', 0.0)
        metrics['topology_vulnerability_score'] = max_signal
        
        return metrics
    
    def _calculate_communication_complexity(self, topology: str, node_count: int) -> Dict[str, int]:
        """Calculate communication complexity for different topologies."""
        
        if topology == "star":
            # Hub communicates with all others
            return {"total_edges": node_count - 1, "max_degree": node_count - 1, "avg_degree": 2.0}
        elif topology == "ring":
            # Each node connects to 2 neighbors
            return {"total_edges": node_count, "max_degree": 2, "avg_degree": 2.0}
        elif topology == "complete":
            # All nodes connect to all others
            total_edges = node_count * (node_count - 1) // 2
            return {"total_edges": total_edges, "max_degree": node_count - 1, "avg_degree": node_count - 1}
        elif topology == "line":
            # Linear chain
            return {"total_edges": node_count - 1, "max_degree": 2, "avg_degree": 2.0}
        else:
            return {"total_edges": 0, "max_degree": 0, "avg_degree": 0.0}
    
    def _analyze_dp_effectiveness(self, attack_results: Dict[str, Any], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze differential privacy effectiveness."""
        
        dp_setting = config['dp_setting']
        
        metrics = {
            "dp_enabled": dp_setting['enabled'],
            "epsilon": dp_setting.get('epsilon'),
            "privacy_level": dp_setting['name']
        }
        
        if dp_setting['enabled']:
            # DP is enabled - measure how well it protects
            overall_eval = attack_results.get('evaluation', {})
            attack_success = overall_eval.get('overall_success', False)
            max_signal = overall_eval.get('attack_indicators', {}).get('max_signal', 0.0)
            
            metrics.update({
                "dp_protection_effective": not attack_success,
                "signal_strength_with_dp": max_signal,
                "privacy_budget_utilization": self._estimate_privacy_budget_usage(config)
            })
        
        return metrics
    
    def _estimate_privacy_budget_usage(self, config: Dict[str, Any]) -> float:
        """Estimate privacy budget utilization (simplified)."""
        # This is a simplified estimation - in real scenarios you'd get this from the DP mechanism
        rounds = 3  # We're using 3 rounds
        epsilon = config['dp_setting'].get('epsilon', 0)
        if epsilon:
            # Rough estimation of budget usage
            return min(1.0, rounds * 0.3)  # Assume ~30% budget per round
        return 0.0
    
    def _analyze_scalability_effects(self, attack_results: Dict[str, Any], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how network size affects attack success."""
        
        node_count = config['node_count']
        
        metrics = {
            "network_size": node_count,
            "size_category": self._categorize_network_size(node_count),
            "attack_scalability_score": 0.0
        }
        
        # Analyze how attack effectiveness scales with network size
        overall_eval = attack_results.get('evaluation', {})
        max_signal = overall_eval.get('attack_indicators', {}).get('max_signal', 0.0)
        
        # Normalize by network size (larger networks might naturally have lower signal)
        # This is a heuristic - larger networks should be harder to attack
        size_normalization = 1.0 + (node_count - 5) * 0.02  # Slight penalty for larger networks
        normalized_score = max_signal * size_normalization
        
        metrics['attack_scalability_score'] = normalized_score
        
        return metrics
    
    def _categorize_network_size(self, node_count: int) -> str:
        """Categorize network size for analysis."""
        if node_count <= 5:
            return "small"
        elif node_count <= 10:
            return "medium" 
        elif node_count <= 15:
            return "large"
        else:
            return "very_large"
    
    def run_comprehensive_experiments(self, 
                                    max_parallel: int = 2,
                                    sample_configs: Optional[int] = None) -> None:
        """Run comprehensive experiments for the paper."""
        
        # Generate all configurations
        all_configs = self.get_valid_configurations()
        
        if sample_configs:
            # For testing, sample a subset
            import random
            random.shuffle(all_configs)
            all_configs = all_configs[:sample_configs]
        
        total_experiments = len(all_configs)
        
        self.logger.info(f"🚀 Starting {total_experiments} comprehensive experiments")
        self.logger.info(f"   Max parallel: {max_parallel}")
        self.logger.info(f"   Output directory: {self.base_output_dir}")
        
        # Estimate total runtime
        total_estimated_time = sum(config['expected_runtime'] for config in all_configs)
        self.logger.info(f"   Estimated total time: {total_estimated_time / 3600:.1f} hours")
        
        print("=" * 100)
        
        start_time = time.time()
        
        # Run experiments with controlled parallelism
        if max_parallel == 1:
            # Sequential execution
            for i, config in enumerate(all_configs, 1):
                self.logger.info(f"[{i}/{total_experiments}] Starting experiment...")
                result = self.run_single_experiment(config)
                self.results.append(result)
                
                # Log brief result summary
                self._log_experiment_result_brief(result, i, total_experiments)
                
                # Save intermediate results
                if i % 10 == 0:
                    self.save_intermediate_results()
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_parallel) as executor:
                # Submit all experiments
                future_to_config = {
                    executor.submit(self.run_single_experiment, config): (i, config) 
                    for i, config in enumerate(all_configs, 1)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_config):
                    i, config = future_to_config[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        
                        # Log brief result summary
                        self._log_experiment_result_brief(result, len(self.results), total_experiments)
                        
                        # Save intermediate results
                        if len(self.results) % 10 == 0:
                            self.save_intermediate_results()
                            
                    except Exception as e:
                        self.logger.error(f"Experiment {i} failed with exception: {e}")
                        error_result = self._create_failed_result(config, f"exp_{i:04d}", str(e), time.time())
                        self.results.append(error_result)
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ All experiments completed in {total_time / 3600:.1f} hours")
        
        # Final analysis
        self.save_final_results()
        self.generate_comprehensive_analysis()
        self.generate_paper_ready_outputs()
    
    def _log_experiment_result_brief(self, result: Dict[str, Any], current_num: int, total_num: int):
        """Log a brief summary of experiment result."""
        config = result['config']
        status = result['status']
        
        # Create brief experiment description
        exp_desc = f"{config['dataset']}/{config['fl_type']}/{config['topology']}/{config['node_count']}n"
        dp_desc = f"DP(ε={config['dp_setting'].get('epsilon', 'off')})" if config['dp_setting']['enabled'] else "no-DP"
        attack_desc = config['attack_strategy'].replace('_', '-')
        
        if status == 'success':
            evaluation = result['evaluation']
            attack_success = evaluation['attack_success']
            confidence = evaluation['confidence_score']
            runtime = result['runtime_seconds']
            
            success_icon = "🎯" if attack_success else "🛡️"
            self.logger.info(f"[{current_num}/{total_num}] {success_icon} {exp_desc} | {dp_desc} | {attack_desc} | "
                           f"Attack: {'SUCCESS' if attack_success else 'FAILED'} (conf:{confidence:.3f}) | "
                           f"Runtime: {runtime:.1f}s")
        else:
            error_brief = result.get('error', 'Unknown error')[:100]  # First 100 chars of error
            runtime = result['runtime_seconds']
            self.logger.info(f"[{current_num}/{total_num}] ❌ {exp_desc} | {dp_desc} | {attack_desc} | "
                           f"EXPERIMENT FAILED | Error: {error_brief} | Runtime: {runtime:.1f}s")
    
    def save_intermediate_results(self):
        """Save intermediate results."""
        results_file = self.base_output_dir / "results" / "intermediate_results.json"
        
        def convert_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return str(obj)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=convert_types)
    
    def save_final_results(self):
        """Save final comprehensive results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.base_output_dir / "results" / f"final_results_{timestamp}.json"
        
        def convert_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return str(obj)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=convert_types)
        
        self.logger.info(f"💾 Final results saved to: {results_file}")
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis for the paper."""
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        total_experiments = len(self.results)
        
        self.logger.info(f"\n" + "=" * 100)
        self.logger.info("📊 COMPREHENSIVE PAPER ANALYSIS")
        self.logger.info("=" * 100)
        
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Successful experiments: {len(successful_results)}")
        self.logger.info(f"Success rate: {len(successful_results)/total_experiments:.1%}")
        
        if not successful_results:
            self.logger.warning("No successful experiments to analyze")
            return
        
        # Create comprehensive DataFrame for analysis
        analysis_data = []
        
        for result in successful_results:
            config = result['config']
            evaluation = result['evaluation']
            
            row = {
                # Experimental parameters
                'experiment_id': result['experiment_id'],
                'dataset': config['dataset'],
                'attack_strategy': config['attack_strategy'],
                'fl_type': config['fl_type'],
                'topology': config['topology'],
                'node_count': config['node_count'],
                'dp_enabled': config['dp_setting']['enabled'],
                'dp_epsilon': config['dp_setting'].get('epsilon'),
                'dp_privacy_level': config['dp_setting']['name'],
                
                # Attack results
                'attack_success': evaluation['attack_success'],
                'confidence_score': evaluation['confidence_score'],
                'best_attack_type': evaluation['best_attack_type'],
                
                # Attack type specific scores
                'comm_pattern_score': evaluation['attack_type_scores'].get('Communication Pattern Attack', 0.0),
                'param_magnitude_score': evaluation['attack_type_scores'].get('Parameter Magnitude Attack', 0.0),
                'topology_structure_score': evaluation['attack_type_scores'].get('Topology Structure Attack', 0.0),
                
                # Topology metrics
                'topology_vulnerability': evaluation['topology_specific_metrics'].get('topology_vulnerability_score', 0.0),
                'communication_complexity': evaluation['topology_specific_metrics'].get('communication_complexity', {}).get('total_edges', 0),
                
                # DP effectiveness
                'dp_protection_effective': evaluation['dp_effectiveness_metrics'].get('dp_protection_effective', True),
                'signal_strength_with_dp': evaluation['dp_effectiveness_metrics'].get('signal_strength_with_dp', 0.0),
                
                # Scalability
                'size_category': evaluation['scalability_metrics'].get('size_category', 'unknown'),
                'attack_scalability_score': evaluation['scalability_metrics'].get('attack_scalability_score', 0.0),
                
                # Performance
                'runtime_seconds': result['runtime_seconds']
            }
            
            analysis_data.append(row)
        
        # Save comprehensive analysis DataFrame
        df = pd.DataFrame(analysis_data)
        analysis_file = self.base_output_dir / "analysis" / "comprehensive_analysis.csv"
        df.to_csv(analysis_file, index=False)
        
        self.logger.info(f"📊 Analysis data saved to: {analysis_file}")
        
        # Generate key insights
        self._generate_key_insights(df)
    
    def _generate_key_insights(self, df: pd.DataFrame):
        """Generate key insights for the paper."""
        
        self.logger.info(f"\n🔑 KEY INSIGHTS FOR PAPER:")
        
        # 1. Overall attack success rates
        overall_success_rate = df['attack_success'].mean()
        self.logger.info(f"1. Overall attack success rate: {overall_success_rate:.1%}")
        
        # 2. DP effectiveness analysis
        if 'dp_enabled' in df.columns:
            dp_comparison = df.groupby('dp_enabled')['attack_success'].agg(['mean', 'count'])
            self.logger.info(f"2. DP Effectiveness:")
            for dp_enabled, stats in dp_comparison.iterrows():
                dp_str = "With DP" if dp_enabled else "Without DP"
                self.logger.info(f"   {dp_str}: {stats['mean']:.1%} success rate ({stats['count']} experiments)")
            
            # DP strength analysis
            if 'dp_epsilon' in df.columns:
                dp_strength = df[df['dp_enabled']].groupby('dp_epsilon')['attack_success'].mean().sort_index(ascending=False)
                self.logger.info(f"   DP by epsilon value:")
                for epsilon, success_rate in dp_strength.items():
                    self.logger.info(f"     ε={epsilon}: {success_rate:.1%} attack success")
        
        # 3. Topology vulnerability ranking
        topo_vuln = df.groupby('topology')['attack_success'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        self.logger.info(f"3. Topology vulnerability ranking:")
        for topology, stats in topo_vuln.iterrows():
            self.logger.info(f"   {topology}: {stats['mean']:.1%} success rate ({stats['count']} experiments)")
        
        # 4. Scalability effects
        scale_effects = df.groupby('size_category')['attack_success'].mean()
        self.logger.info(f"4. Scalability effects:")
        for size_cat, success_rate in scale_effects.items():
            self.logger.info(f"   {size_cat} networks: {success_rate:.1%} attack success")
        
        # 5. Attack strategy effectiveness
        strategy_effectiveness = df.groupby('attack_strategy')['attack_success'].agg(['mean', 'count'])
        self.logger.info(f"5. Attack strategy effectiveness:")
        for strategy, stats in strategy_effectiveness.iterrows():
            self.logger.info(f"   {strategy}: {stats['mean']:.1%} success rate ({stats['count']} experiments)")
        
        # 6. Dataset-specific findings
        dataset_findings = df.groupby('dataset')['attack_success'].agg(['mean', 'count'])
        self.logger.info(f"6. Dataset-specific findings:")
        for dataset, stats in dataset_findings.iterrows():
            self.logger.info(f"   {dataset}: {stats['mean']:.1%} success rate ({stats['count']} experiments)")
        
        # 7. FL type comparison
        fl_type_comparison = df.groupby('fl_type')['attack_success'].agg(['mean', 'count'])
        self.logger.info(f"7. FL type comparison:")
        for fl_type, stats in fl_type_comparison.iterrows():
            self.logger.info(f"   {fl_type}: {stats['mean']:.1%} success rate ({stats['count']} experiments)")
    
    def generate_paper_ready_outputs(self):
        """Generate publication-ready outputs."""
        
        self.logger.info(f"\n📄 Generating paper-ready outputs...")
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        if not successful_results:
            self.logger.warning("No successful results for paper outputs")
            return
        
        # Create summary tables for paper
        self._create_paper_summary_tables()
        
        # Generate LaTeX tables
        self._generate_latex_tables()
        
        # Create experimental setup summary
        self._create_experimental_setup_summary()
        
        self.logger.info(f"📄 Paper-ready outputs saved in: {self.base_output_dir / 'analysis'}")
    
    def _create_paper_summary_tables(self):
        """Create summary tables for the paper."""
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        # Table 1: Attack success by configuration
        summary_data = []
        
        for result in successful_results:
            config = result['config']
            evaluation = result['evaluation']
            
            summary_data.append({
                'Dataset': config['dataset'].upper(),
                'FL Type': config['fl_type'].title(),
                'Topology': config['topology'].title(),
                'Nodes': config['node_count'],
                'DP': 'Yes' if config['dp_setting']['enabled'] else 'No',
                'Epsilon': config['dp_setting'].get('epsilon', 'N/A'),
                'Attack Strategy': config['attack_strategy'].replace('_', ' ').title(),
                'Attack Success': 'Yes' if evaluation['attack_success'] else 'No',
                'Confidence': f"{evaluation['confidence_score']:.3f}",
                'Best Attack Type': evaluation['best_attack_type']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.base_output_dir / "analysis" / "paper_summary_table.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Table 2: Aggregate statistics
        aggregate_stats = self._create_aggregate_statistics_table(summary_df)
        agg_file = self.base_output_dir / "analysis" / "aggregate_statistics.csv"
        aggregate_stats.to_csv(agg_file, index=False)
    
    def _create_aggregate_statistics_table(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate statistics table."""
        
        # Convert Attack Success to numeric
        summary_df['Attack Success Numeric'] = summary_df['Attack Success'].map({'Yes': 1, 'No': 0})
        
        # Group by key factors
        stats = []
        
        # By topology
        topo_stats = summary_df.groupby('Topology').agg({
            'Attack Success Numeric': ['count', 'sum', 'mean'],
            'Confidence': lambda x: pd.to_numeric(x, errors='coerce').mean()
        }).round(3)
        
        for topology in topo_stats.index:
            count = topo_stats.loc[topology, ('Attack Success Numeric', 'count')]
            success = topo_stats.loc[topology, ('Attack Success Numeric', 'sum')]
            rate = topo_stats.loc[topology, ('Attack Success Numeric', 'mean')]
            conf = topo_stats.loc[topology, ('Confidence', '<lambda>')]
            
            stats.append({
                'Category': 'Topology',
                'Subcategory': topology,
                'Total Experiments': count,
                'Successful Attacks': success,
                'Success Rate': f"{rate:.1%}",
                'Avg Confidence': f"{conf:.3f}"
            })
        
        # By DP setting
        dp_stats = summary_df.groupby('DP').agg({
            'Attack Success Numeric': ['count', 'sum', 'mean'],
            'Confidence': lambda x: pd.to_numeric(x, errors='coerce').mean()
        }).round(3)
        
        for dp in dp_stats.index:
            count = dp_stats.loc[dp, ('Attack Success Numeric', 'count')]
            success = dp_stats.loc[dp, ('Attack Success Numeric', 'sum')]
            rate = dp_stats.loc[dp, ('Attack Success Numeric', 'mean')]
            conf = dp_stats.loc[dp, ('Confidence', '<lambda>')]
            
            stats.append({
                'Category': 'Differential Privacy',
                'Subcategory': dp,
                'Total Experiments': count,
                'Successful Attacks': success,
                'Success Rate': f"{rate:.1%}",
                'Avg Confidence': f"{conf:.3f}"
            })
        
        return pd.DataFrame(stats)
    
    def _generate_latex_tables(self):
        """Generate LaTeX formatted tables for the paper."""
        
        # This would generate LaTeX table code - simplified for now
        latex_content = """
% LaTeX tables for topology attack paper
% Generated automatically from experimental results

\\begin{table}[h]
\\centering
\\caption{Attack Success Rates by Configuration}
\\label{tab:attack_success}
\\begin{tabular}{|l|l|l|c|c|c|}
\\hline
Topology & FL Type & DP & Nodes & Success Rate & Avg Confidence \\\\
\\hline
% Data would be inserted here from the experimental results
\\end{tabular}
\\end{table}
"""
        
        latex_file = self.base_output_dir / "analysis" / "latex_tables.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_content)
    
    def _create_experimental_setup_summary(self):
        """Create experimental setup summary for the paper."""
        
        setup_summary = {
            "experimental_design": {
                "total_configurations": len(self.get_valid_configurations()),
                "datasets": ["MNIST", "Skin Lesion"],
                "node_counts": [5, 10, 15, 20, 25],
                "topologies": {
                    "federated": ["star", "complete"],
                    "decentralized": ["ring", "complete", "star", "line"]
                },
                "attack_strategies": ["sensitive_groups", "topology_correlated", "imbalanced_sensitive"],
                "dp_settings": ["no_dp", "weak_dp", "medium_dp", "strong_dp", "very_strong_dp"]
            },
            "experimental_parameters": {
                "training_rounds": 3,
                "local_epochs": 1,
                "attack_types": ["Communication Pattern", "Parameter Magnitude", "Topology Structure"],
                "success_threshold": 0.3
            },
            "infrastructure": {
                "max_parallel_experiments": 2,
                "timeout_per_experiment": "3x estimated runtime",
                "storage_location": str(self.base_output_dir)
            }
        }
        
        setup_file = self.base_output_dir / "analysis" / "experimental_setup.json"
        with open(setup_file, 'w') as f:
            json.dump(setup_summary, f, indent=2)


def main():
    """Main function for comprehensive paper experiments."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Topology Attack Experiments for Paper"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paper_experiments",
        help="Output directory for all experimental data"
    )
    
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=2,
        help="Maximum parallel experiments (be careful with resource usage)"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample only N configurations for testing (use all if not specified)"
    )
    
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run a quick test with minimal configurations"
    )
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = PaperExperimentRunner(args.output_dir)
    
    if args.quick_test:
        print("🏃‍♂️ Running quick test with 10 sample configurations...")
        sample_size = 10
    else:
        sample_size = args.sample_size
    
    # Run comprehensive experiments
    runner.run_comprehensive_experiments(
        max_parallel=args.max_parallel,
        sample_configs=sample_size
    )
    
    print(f"\n✅ Comprehensive paper experiments completed!")
    print(f"📁 All data available in: {args.output_dir}")
    print(f"📊 Analysis files in: {args.output_dir}/analysis")
    print(f"📈 Visualization data in: {args.output_dir}/visualizations")


if __name__ == "__main__":
    main()