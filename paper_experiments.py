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
    
    def __init__(self, base_output_dir: str = "./paper_experiments", phase: int = 1):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.phase = phase
        
        # Create phase-specific subdirectories
        (self.base_output_dir / f"visualizations_phase{phase}").mkdir(exist_ok=True)
        (self.base_output_dir / f"results_phase{phase}").mkdir(exist_ok=True)
        (self.base_output_dir / "logs").mkdir(exist_ok=True)
        (self.base_output_dir / "analysis").mkdir(exist_ok=True)
        
        self.results = []
        self.experiment_id = 0
        
        # Setup logging
        self.setup_logging()
        
        # Define compatibility matrix based on actual framework constraints
        self.compatibility_matrix = {
            # Centralized FL (FedAvg/TrimmedMean): ONLY star and complete (global parameter collection)
            "federated": {
                "compatible_topologies": ["star", "complete"],
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
    
    def get_valid_configurations(self, dataset_filter: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate all valid experimental configurations.
        
        Args:
            dataset_filter: If specified, only generate configs for this dataset ('mnist' or 'ham10000')
            filters: Dictionary of filters to apply (datasets, fl_types, topologies, node_counts, dp_settings, attack_strategies)
        """
        
        configurations = []
        
        # Experimental parameters with dataset-optimal node counts
        all_datasets_and_node_counts = [
            ("mnist", [5, 10, 15, 20, 30]),      # MNIST: 10 classes, optimal at 10 nodes
            ("ham10000", [5, 7, 10, 15, 20, 30])          # HAM10000: 7 classes, optimal at 7 nodes; using multiples of 10 for cluster efficiency (except 7)
        ]
        
        # Apply filters if provided
        if filters:
            # Dataset filtering
            if filters.get('datasets'):
                all_datasets_and_node_counts = [(d, n) for d, n in all_datasets_and_node_counts if d in filters['datasets']]
                self.logger.info(f"Filtering experiments to datasets: {filters['datasets']}")
            
            # Node count filtering
            if filters.get('node_counts'):
                all_datasets_and_node_counts = [(d, [nc for nc in n if nc in filters['node_counts']]) 
                                               for d, n in all_datasets_and_node_counts]
                self.logger.info(f"Filtering experiments to node counts: {filters['node_counts']}")
        elif dataset_filter:
            # Legacy dataset_filter support
            if dataset_filter not in ["mnist", "ham10000"]:
                raise ValueError(f"Invalid dataset filter: {dataset_filter}. Must be 'mnist' or 'ham10000'")
            all_datasets_and_node_counts = [(d, n) for d, n in all_datasets_and_node_counts if d == dataset_filter]
            self.logger.info(f"Filtering experiments to dataset: {dataset_filter}")
        
        datasets_and_node_counts = all_datasets_and_node_counts
        
        # Attack strategies
        attack_strategies = ["sensitive_groups", "topology_correlated", "imbalanced_sensitive"]
        if filters and filters.get('attack_strategies'):
            attack_strategies = [a for a in attack_strategies if a in filters['attack_strategies']]
            self.logger.info(f"Filtering experiments to attack strategies: {filters['attack_strategies']}")
        
        # DP settings
        dp_settings = [
            {"enabled": False, "epsilon": None, "name": "no_dp"},
            {"enabled": True, "epsilon": 16.0, "name": "weak_dp"},
            {"enabled": True, "epsilon": 8.0, "name": "medium_dp"},
            {"enabled": True, "epsilon": 4.0, "name": "strong_dp"},
            {"enabled": True, "epsilon": 1.0, "name": "very_strong_dp"}
        ]
        if filters and filters.get('dp_settings'):
            dp_settings = [dp for dp in dp_settings if dp['name'] in filters['dp_settings']]
            self.logger.info(f"Filtering experiments to DP settings: {filters['dp_settings']}")
        
        config_id = 1
        
        for dataset, node_counts in datasets_and_node_counts:
            for attack_strategy in attack_strategies:
                # For topology_correlated, only use optimal node count
                if attack_strategy == "topology_correlated":
                    # Optimal: 10 nodes for MNIST (10 classes), 7 nodes for HAM10000 (7 classes)
                    optimal_nodes = 10 if dataset == "mnist" else 7
                    node_counts_to_use = [optimal_nodes]
                else:
                    # For other strategies, use all varying node counts
                    node_counts_to_use = node_counts
                
                for node_count in node_counts_to_use:
                    for dp_setting in dp_settings:
                        
                        # Test both federated and decentralized approaches
                        fl_types = ["federated", "decentralized"]
                        if filters and filters.get('fl_types'):
                            fl_types = [fl for fl in fl_types if fl in filters['fl_types']]
                        
                        for fl_type in fl_types:
                            valid_topologies = self.compatibility_matrix[fl_type]["compatible_topologies"]
                            
                            # Apply topology filter
                            if filters and filters.get('topologies'):
                                valid_topologies = [t for t in valid_topologies if t in filters['topologies']]
                            
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
    
    def get_phase2_sampling_configurations(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate phase 2 configurations focused on client and data sampling effects.
        
        This is a targeted set of experiments to address reviewer concerns about 
        sampling effects in realistic federated learning deployments.
        """
        
        configurations = []
        
        # Phase 2 Focus: Test sampling effects on most vulnerable configurations
        # from phase 1 results_phase1 to maximize insight per experiment
        
        # Sampling rates to test (realistic FL deployment scenarios)
        sampling_scenarios = [
            {"name": "moderate_sampling", "client_rate": 0.5, "data_rate": 0.8},
            {"name": "strong_sampling", "client_rate": 0.3, "data_rate": 0.6},
            {"name": "very_strong_sampling", "client_rate": 0.2, "data_rate": 0.5}
        ]
        
        # Focus on key datasets (MNIST for baseline, HAM10000 for real-world validation)
        focus_datasets = ["mnist", "ham10000"]
        if filters and filters.get('datasets'):
            focus_datasets = [d for d in focus_datasets if d in filters['datasets']]
        
        # Focus on most vulnerable attack strategies from phase 1
        focus_attack_strategies = ["topology_correlated", "sensitive_groups"]
        if filters and filters.get('attack_strategies'):
            focus_attack_strategies = [a for a in focus_attack_strategies if a in filters['attack_strategies']]
        
        # Focus on optimal/representative node counts
        focus_node_counts = {
            "mnist": [10, 15],      # 10 = optimal (matches classes), 15 = larger scale
            "ham10000": [7, 10]     # 7 = optimal (matches classes), 10 = common scale
        }
        if filters and filters.get('node_counts'):
            for dataset in focus_node_counts:
                focus_node_counts[dataset] = [n for n in focus_node_counts[dataset] if n in filters['node_counts']]
        
        # Focus on key topology/FL-type combinations that showed vulnerability
        focus_combinations = [
            ("federated", "star"),      # Classic FL - most vulnerable to centralized attacks
            ("federated", "complete"),  # Fully connected FL
            ("decentralized", "ring"),  # Structured decentralized - vulnerable to topology attacks
            ("decentralized", "complete") # Fully connected decentralized
        ]
        if filters:
            if filters.get('fl_types') or filters.get('topologies'):
                focus_combinations = [
                    (fl, topo) for fl, topo in focus_combinations
                    if (not filters.get('fl_types') or fl in filters['fl_types']) and
                       (not filters.get('topologies') or topo in filters['topologies'])
                ]
        
        # Focus on key DP settings that matter for sampling analysis
        focus_dp_settings = [
            {"enabled": False, "epsilon": None, "name": "no_dp"},
            {"enabled": True, "epsilon": 8.0, "name": "medium_dp"},  # Most representative DP setting
            {"enabled": True, "epsilon": 4.0, "name": "strong_dp"}   # Strong DP to test amplification
        ]
        if filters and filters.get('dp_settings'):
            focus_dp_settings = [dp for dp in focus_dp_settings if dp['name'] in filters['dp_settings']]
        
        config_id = 2001  # Start phase 2 IDs at 2001 to distinguish from phase 1
        
        self.logger.info("🎯 Generating Phase 2 (Sampling) configurations...")
        self.logger.info(f"   Sampling scenarios: {[s['name'] for s in sampling_scenarios]}")
        self.logger.info(f"   Focus datasets: {focus_datasets}")
        self.logger.info(f"   Focus attack strategies: {focus_attack_strategies}")
        self.logger.info(f"   Focus combinations: {focus_combinations}")
        
        for dataset in focus_datasets:
            for attack_strategy in focus_attack_strategies:
                for node_count in focus_node_counts[dataset]:
                    for fl_type, topology in focus_combinations:
                        # Check compatibility
                        if topology not in self.compatibility_matrix[fl_type]["compatible_topologies"]:
                            continue
                        
                        for dp_setting in focus_dp_settings:
                            for sampling_scenario in sampling_scenarios:
                                config = {
                                    "config_id": config_id,
                                    "phase": 2,  # Mark as phase 2
                                    "sampling_scenario": sampling_scenario["name"],
                                    "client_sampling_rate": sampling_scenario["client_rate"],
                                    "data_sampling_rate": sampling_scenario["data_rate"],
                                    "dataset": dataset,
                                    "attack_strategy": attack_strategy,
                                    "fl_type": fl_type,
                                    "topology": topology,
                                    "node_count": node_count,
                                    "dp_setting": dp_setting,
                                    "expected_runtime": self._estimate_runtime(dataset, node_count, fl_type)
                                }
                                
                                configurations.append(config)
                                config_id += 1
        
        self.logger.info(f"Generated {len(configurations)} Phase 2 sampling configurations")
        return configurations
    
    def _estimate_runtime(self, dataset: str, node_count: int, fl_type: str) -> int:
        """Estimate runtime in seconds for resource planning."""
        
        base_time = {
            "mnist": {"federated": 60, "decentralized": 90},  # Base time for 5 nodes
            "ham10000": {"federated": 120, "decentralized": 180}  # HAM10000 is lightweight medical imaging dataset
        }
        
        # Scale with node count (roughly linear)
        scaling_factor = node_count / 5.0
        estimated_time = base_time[dataset][fl_type] * scaling_factor
        
        return int(estimated_time)
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        
        self.experiment_id += 1
        
        # Create experiment name with phase and sampling info
        base_name = f"exp_{self.experiment_id:04d}_{config['dataset']}_{config['fl_type']}_{config['topology']}_{config['node_count']}n_{config['dp_setting']['name']}"
        
        if config.get('phase') == 2:
            # Phase 2 experiments include sampling info in name
            sampling_name = config.get('sampling_scenario', 'sampling')
            experiment_name = f"{base_name}_{sampling_name}"
        else:
            experiment_name = base_name
        
        self.logger.info(f"🎯 Starting experiment {self.experiment_id}: {experiment_name}")
        
        # Log config details
        config_details = f"   Config: {config['attack_strategy']} attack, {config['node_count']} nodes, {config['dp_setting']['name']}"
        if config.get('phase') == 2:
            config_details += f", sampling: C={config.get('client_sampling_rate', 1.0):.1f} D={config.get('data_sampling_rate', 1.0):.1f}"
        self.logger.info(config_details)
        
        start_time = time.time()
        
        try:
            # Select appropriate example script
            script_map = {
                ("mnist", "federated"): "dp_mnist_example.py",
                ("mnist", "decentralized"): "dp_decentralized_mnist_example.py", 
                ("ham10000", "federated"): "dp_ham10000_example.py",
                ("ham10000", "decentralized"): "dp_decentralized_ham10000_example.py"
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
            
            # Check for success/failure more intelligently
            stdout_output = result.stdout.strip() if result.stdout else ""
            stderr_output = result.stderr.strip() if result.stderr else ""
            
            # Look for specific success indicators in output
            success_indicators = [
                "All training data exported to",
                "Summary plot saved to",
                "Model saved to",
                "Learning process shut down successfully"
            ]
            
            # Look for actual error indicators
            error_indicators = [
                "Traceback (most recent call last):",
                "ValueError:",
                "RuntimeError:",
                "Exception:",
                "Error:",
                "FAILED"
            ]
            
            has_success = any(indicator in stdout_output for indicator in success_indicators)
            has_error = any(indicator in stdout_output or indicator in stderr_output for indicator in error_indicators)
            
            if result.returncode != 0 or (has_error and not has_success):
                # Get more complete error information
                all_output = stdout_output + "\n" + stderr_output
                
                # Find the actual error in the output
                if "Traceback" in all_output:
                    # Extract traceback
                    lines = all_output.split('\n')
                    traceback_start = -1
                    for i, line in enumerate(lines):
                        if "Traceback (most recent call last):" in line:
                            traceback_start = i
                            break
                    
                    if traceback_start >= 0:
                        traceback_lines = lines[traceback_start:traceback_start+20]  # Get traceback + context
                        error_msg = f"Training failed with traceback: {' '.join(traceback_lines)}"
                    else:
                        error_msg = f"Training failed: {all_output[-1500:]}"
                else:
                    error_msg = f"Training failed (return code {result.returncode}): {all_output[-1500:]}"
                
                self.logger.error(f"   ❌ {error_msg}")
                return self._create_failed_result(config, experiment_name, error_msg, start_time)
            
            self.logger.info(f"   ✅ Training completed in {time.time() - start_time:.1f}s")
            
            # Run attacks
            viz_dir = self.base_output_dir / f"visualizations_phase{self.phase}" / experiment_name
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
            "--vis_dir", str(self.base_output_dir / f"visualizations_phase{self.phase}"),
            "--create_summary",  # Enable visualization generation
            "--experiment_name", experiment_name,  # Use custom experiment name
        ]
        
        # Add sampling settings (phase 2 experiments)
        if 'client_sampling_rate' in config:
            cmd.extend(["--client_sampling_rate", str(config['client_sampling_rate'])])
        if 'data_sampling_rate' in config:
            cmd.extend(["--data_sampling_rate", str(config['data_sampling_rate'])])
        
        # Add DP settings
        if config['dp_setting']['enabled']:
            cmd.extend(["--enable_dp", "--target_epsilon", str(config['dp_setting']['epsilon'])])
            # Enable privacy amplification for sampling experiments
            if 'client_sampling_rate' in config or 'data_sampling_rate' in config:
                cmd.extend(["--enable_subsampling_amplification"])
        
        # Add FL-type specific settings
        if config['fl_type'] == "federated":
            cmd.extend(["--aggregation_strategy", "fedavg"])
        else:  # decentralized
            cmd.extend(["--aggregation_strategy", "gossip_avg"])
        
        # Dataset-specific adjustments
        if config['dataset'] == "ham10000":
            cmd.extend(["--image_size", "128"])  # HAM10000 uses 128x128 images
        
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
                                    sample_configs: Optional[int] = None,
                                    dataset_filter: Optional[str] = None,
                                    resume_from: Optional[int] = None,
                                    filters: Optional[Dict[str, Any]] = None,
                                    phase: int = 1) -> None:
        """Run comprehensive experiments for the paper."""
        
        # Generate configurations based on phase
        if phase == 2:
            self.logger.info("🔬 Running Phase 2: Sampling Effects Experiments")
            all_configs = self.get_phase2_sampling_configurations(filters)
        else:
            self.logger.info("🔬 Running Phase 1: Comprehensive Baseline Experiments")
            all_configs = self.get_valid_configurations(dataset_filter, filters)
        
        # Handle resume functionality
        if resume_from is not None:
            if resume_from < 1 or resume_from > len(all_configs):
                raise ValueError(f"Resume point {resume_from} is out of range (1-{len(all_configs)})")
            
            # Skip already completed experiments
            all_configs = all_configs[resume_from - 1:]
            self.experiment_id = resume_from - 1  # Set starting experiment ID
            
            self.logger.info(f"🔄 Resuming from experiment {resume_from}")
            self.logger.info(f"   Skipping first {resume_from - 1} experiments")
            
            # Load previous results_phase1 if they exist
            try:
                results_file = self.base_output_dir / f"results_phase{self.phase}" / "intermediate_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        previous_results = json.load(f)
                        self.results = previous_results
                        self.logger.info(f"   Loaded {len(previous_results)} previous results_phase1")
            except Exception as e:
                self.logger.warning(f"Could not load previous results_phase1: {e}")
        
        if sample_configs and resume_from is None:
            # For testing, sample a subset (only if not resuming)
            import random
            random.shuffle(all_configs)
            all_configs = all_configs[:sample_configs]
        
        total_experiments = len(all_configs)
        original_total = len(self.get_valid_configurations(dataset_filter))
        
        if resume_from is not None:
            self.logger.info(f"🚀 Resuming {total_experiments} remaining experiments (from {resume_from}/{original_total})")
        else:
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
                # Calculate actual experiment number (accounting for resume)
                actual_experiment_num = (resume_from or 1) + i - 1
                self.logger.info(f"[{actual_experiment_num}/{original_total}] Starting experiment...")
                result = self.run_single_experiment(config)
                self.results.append(result)
                
                # Log brief result summary
                self._log_experiment_result_brief(result, actual_experiment_num, original_total)
                
                # Save intermediate results_phase1 after each experiment
                self.save_intermediate_results()
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_parallel) as executor:
                # Submit all experiments
                future_to_config = {
                    executor.submit(self.run_single_experiment, config): (i, config) 
                    for i, config in enumerate(all_configs, 1)
                }
                
                # Collect results_phase1 as they complete
                for future in as_completed(future_to_config):
                    i, config = future_to_config[future]
                    # Calculate actual experiment number (accounting for resume)
                    actual_experiment_num = (resume_from or 1) + i - 1
                    try:
                        result = future.result()
                        self.results.append(result)
                        
                        # Log brief result summary
                        self._log_experiment_result_brief(result, actual_experiment_num, original_total)
                        
                        # Save intermediate results_phase1 after each experiment
                        self.save_intermediate_results()
                            
                    except Exception as e:
                        self.logger.error(f"Experiment {actual_experiment_num} failed with exception: {e}")
                        error_result = self._create_failed_result(config, f"exp_{actual_experiment_num:04d}", str(e), time.time())
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
    
    def _convert_for_json(self, obj):
        """Recursively convert objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {str(k): self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item') and hasattr(obj, 'size'):
            try:
                if obj.size == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            except (ValueError, AttributeError):
                return str(obj)
        elif hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
            try:
                return obj.tolist()
            except (ValueError, AttributeError):
                return str(obj)
        return obj

    def save_intermediate_results(self):
        """Save intermediate results_phase1."""
        results_file = self.base_output_dir / f"results_phase{self.phase}" / "intermediate_results.json"
        
        # Convert all results_phase1 to JSON-serializable format
        serializable_results = self._convert_for_json(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def save_final_results(self):
        """Save final comprehensive results_phase1."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.base_output_dir / f"results_phase{self.phase}" / f"final_results_{timestamp}.json"
        
        # Convert all results_phase1 to JSON-serializable format
        serializable_results = self._convert_for_json(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"💾 Final results_phase1 saved to: {results_file}")
    
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
                
                # Attack results_phase1
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
            self.logger.warning("No successful results_phase1 for paper outputs")
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
% Generated automatically from experimental results_phase1

\\begin{table}[h]
\\centering
\\caption{Attack Success Rates by Configuration}
\\label{tab:attack_success}
\\begin{tabular}{|l|l|l|c|c|c|}
\\hline
Topology & FL Type & DP & Nodes & Success Rate & Avg Confidence \\\\
\\hline
% Data would be inserted here from the experimental results_phase1
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
                "datasets": ["MNIST", "HAM10000"],
                "node_counts": {"MNIST": [5, 10, 15, 20, 25], "HAM10000": [5, 7, 10, 20]},
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
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "ham10000"],
        default=None,
        help="Run experiments only for specified dataset (mnist or ham10000)"
    )
    
    parser.add_argument(
        "--resume_from",
        type=int,
        default=None,
        help="Resume experiments from a specific experiment number (1-based)"
    )
    
    # New filtering arguments for rerunning specific experiments
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["mnist", "ham10000"],
        default=None,
        help="Run experiments only for specified datasets"
    )
    
    parser.add_argument(
        "--fl-types",
        type=str,
        nargs="+",
        choices=["federated", "decentralized"],
        default=None,
        help="Run experiments only for specified FL types"
    )
    
    parser.add_argument(
        "--topologies",
        type=str,
        nargs="+",
        choices=["star", "complete", "ring", "line"],
        default=None,
        help="Run experiments only for specified topologies"
    )
    
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=None,
        help="Run experiments only for specified node counts"
    )
    
    parser.add_argument(
        "--dp-settings",
        type=str,
        nargs="+",
        choices=["no_dp", "weak_dp", "medium_dp", "strong_dp", "very_strong_dp"],
        default=None,
        help="Run experiments only for specified DP settings"
    )
    
    parser.add_argument(
        "--attack-strategies",
        type=str,
        nargs="+",
        choices=["sensitive_groups", "imbalanced_sensitive", "topology_correlated"],
        default=None,
        help="Run experiments only for specified attack strategies"
    )
    
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Experimental phase: 1=baseline comprehensive experiments, 2=sampling effects experiments"
    )
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = PaperExperimentRunner(args.output_dir, args.phase)
    
    if args.quick_test:
        print("🏃‍♂️ Running quick test with 10 sample configurations...")
        sample_size = 10
    else:
        sample_size = args.sample_size
    
    # Build filters from command line arguments
    filters = {}
    if args.datasets:
        filters['datasets'] = args.datasets
    if args.fl_types:
        filters['fl_types'] = [ft.replace('-', '_') for ft in args.fl_types]
    if args.topologies:
        filters['topologies'] = args.topologies
    if args.node_counts:
        filters['node_counts'] = args.node_counts
    if args.dp_settings:
        filters['dp_settings'] = [dp.replace('-', '_') for dp in args.dp_settings]
    if args.attack_strategies:
        filters['attack_strategies'] = [a.replace('-', '_') for a in args.attack_strategies]
    
    # Run comprehensive experiments
    runner.run_comprehensive_experiments(
        max_parallel=args.max_parallel,
        sample_configs=sample_size,
        dataset_filter=args.dataset,
        resume_from=args.resume_from,
        filters=filters if filters else None,
        phase=args.phase
    )
    
    phase_name = "Phase 1 (Baseline)" if args.phase == 1 else "Phase 2 (Sampling Effects)"
    print(f"\n✅ {phase_name} paper experiments completed!")
    print(f"📁 All data available in: {args.output_dir}")
    print(f"📊 Analysis files in: {args.output_dir}/analysis")
    print(f"📈 Visualization data in: {args.output_dir}/visualizations_phase{args.phase}")


if __name__ == "__main__":
    main()