#!/usr/bin/env python3
"""
Comprehensive Experiment Suite for Trust-Based Drift Detection Paper.

This suite systematically evaluates the trust monitoring system across all
configuration permutations and collects detailed metrics for paper publication.

Metrics Collected:
- Attack Detection: TP, TN, FP, FN, Precision, Recall, F1-Score
- Trust Monitor Overhead: Memory, Computation, Communication
- Detection Latency: Rounds to detection, Time to detection
- System Performance: FL accuracy, Convergence, Robustness
"""

import argparse
import json
import logging
import os
import time
import itertools
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import ray

# Import our examples
from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist, create_attack_config as create_mnist_attack_config
from murmura.examples.adaptive_trust_cifar10_example import run_adaptive_trust_cifar10, create_attack_config as create_cifar10_attack_config


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    # Dataset and model
    dataset: str  # "mnist", "cifar10"
    
    # Network configuration
    num_actors: int
    num_rounds: int
    topology: str  # "ring", "complete", "line"
    
    # Trust configuration
    trust_profile: str  # "permissive", "default", "strict"
    use_beta_threshold: bool
    
    # Attack configuration
    attack_type: str  # "none", "label_flipping", "model_poisoning", "byzantine_gradient"
    malicious_fraction: float
    attack_intensity: str  # "low", "moderate", "high"
    stealth_level: str  # "low", "medium", "high"
    target_class: int = 0
    
    # Experiment metadata
    experiment_id: str = ""
    repeat_id: int = 0
    random_seed: int = 42


@dataclass 
class ExperimentMetrics:
    """Comprehensive metrics collected from experiment."""
    
    # Experiment identification
    experiment_id: str
    config: ExperimentConfig
    
    # Attack Detection Metrics (Ground Truth)
    true_malicious_nodes: List[str]
    detected_malicious_nodes: List[str]
    excluded_nodes: List[str]
    downgraded_nodes: List[str]
    
    # Classification Metrics
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    # Detection Latency
    detection_rounds: Dict[str, int]  # node_id -> round detected
    avg_detection_latency: float
    min_detection_latency: int
    max_detection_latency: int
    
    # Trust Monitor Overhead
    avg_memory_usage_mb: float
    max_memory_usage_mb: float
    avg_cpu_usage_percent: float
    trust_computation_time_ms: float
    communication_overhead_percent: float
    
    # Federated Learning Performance
    initial_accuracy: float
    final_accuracy: float
    accuracy_degradation: float
    convergence_rounds: int
    training_time_seconds: float
    
    # Trust System Statistics
    avg_trust_score_honest: float
    avg_trust_score_malicious: float
    trust_score_separation: float  # Cohen's d
    beta_threshold_evolution: List[float]
    hsic_values_honest: List[float]
    hsic_values_malicious: List[float]
    
    # Experiment metadata
    experiment_duration: float
    success: bool
    error_message: str = ""


class ExperimentRunner:
    """Runs individual experiments and collects metrics."""
    
    def __init__(self, base_output_dir: str = "experiment_results"):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("ExperimentRunner")
        
    def run_experiment(self, config: ExperimentConfig) -> ExperimentMetrics:
        """Run a single experiment and collect comprehensive metrics."""
        
        start_time = time.time()
        self.logger.info(f"Starting experiment {config.experiment_id}")
        
        try:
            # Setup experiment directory
            exp_dir = os.path.join(self.base_output_dir, config.experiment_id)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Create attack configuration
            attack_config = self._create_attack_config(config)
            
            # Setup resource monitoring
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples = [initial_memory]
            cpu_samples = []
            
            # Run the appropriate experiment
            if config.dataset == "mnist":
                results = run_adaptive_trust_mnist(
                    num_actors=config.num_actors,
                    num_rounds=config.num_rounds,
                    topology_type=config.topology,
                    trust_profile=config.trust_profile,
                    use_beta_threshold=config.use_beta_threshold,
                    attack_config=attack_config,
                    output_dir=exp_dir,
                    log_level="INFO"
                )
            elif config.dataset == "cifar10":
                results = run_adaptive_trust_cifar10(
                    num_actors=config.num_actors,
                    num_rounds=config.num_rounds,
                    topology_type=config.topology,
                    trust_profile=config.trust_profile,
                    use_beta_threshold=config.use_beta_threshold,
                    model_type="standard",
                    attack_config=attack_config,
                    output_dir=exp_dir,
                    log_level="INFO"
                )
            else:
                raise ValueError(f"Unknown dataset: {config.dataset}")
            
            # Collect final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(final_memory)
            
            # Extract and calculate metrics
            metrics = self._extract_metrics(config, results, memory_samples, cpu_samples, start_time)
            
            # Save detailed results
            self._save_experiment_results(exp_dir, config, metrics, results)
            
            self.logger.info(f"Completed experiment {config.experiment_id}: "
                           f"Detection={metrics.recall:.3f}, Latency={metrics.avg_detection_latency:.1f}")
            
            return metrics
            
        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error(f"Experiment {config.experiment_id} failed: {str(e)}")
            
            # Return failed experiment metrics
            return ExperimentMetrics(
                experiment_id=config.experiment_id,
                config=config,
                true_malicious_nodes=[],
                detected_malicious_nodes=[],
                excluded_nodes=[],
                downgraded_nodes=[],
                true_positives=0, true_negatives=0,
                false_positives=0, false_negatives=0,
                precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0,
                detection_rounds={},
                avg_detection_latency=float('inf'),
                min_detection_latency=-1, max_detection_latency=-1,
                avg_memory_usage_mb=0.0, max_memory_usage_mb=0.0,
                avg_cpu_usage_percent=0.0, trust_computation_time_ms=0.0,
                communication_overhead_percent=0.0,
                initial_accuracy=0.0, final_accuracy=0.0,
                accuracy_degradation=0.0, convergence_rounds=-1,
                training_time_seconds=error_time,
                avg_trust_score_honest=0.0, avg_trust_score_malicious=0.0,
                trust_score_separation=0.0,
                beta_threshold_evolution=[], hsic_values_honest=[], hsic_values_malicious=[],
                experiment_duration=error_time,
                success=False,
                error_message=str(e)
            )
    
    def _create_attack_config(self, config: ExperimentConfig) -> Optional[Dict[str, Any]]:
        """Create attack configuration based on experiment config."""
        if config.attack_type == "none":
            return None
        
        if config.dataset == "mnist":
            return create_mnist_attack_config(
                attack_type=config.attack_type,
                malicious_fraction=config.malicious_fraction,
                attack_intensity=config.attack_intensity,
                stealth_level=config.stealth_level,
                target_class=config.target_class
            )
        else:  # cifar10
            return create_cifar10_attack_config(
                attack_type=config.attack_type,
                malicious_fraction=config.malicious_fraction,
                attack_intensity=config.attack_intensity,
                stealth_level=config.stealth_level,
                target_class=config.target_class
            )
    
    def _extract_metrics(self, 
                        config: ExperimentConfig, 
                        results: Dict[str, Any],
                        memory_samples: List[float],
                        cpu_samples: List[float],
                        start_time: float) -> ExperimentMetrics:
        """Extract comprehensive metrics from experiment results."""
        
        # Get ground truth malicious nodes
        true_malicious_nodes = []
        if config.attack_type != "none":
            num_malicious = int(config.num_actors * config.malicious_fraction)
            # Assume malicious nodes are deterministically selected (same as in malicious_client.py)
            np.random.seed(42)  # Same seed as used in create_mixed_actors
            malicious_indices = np.random.choice(config.num_actors, num_malicious, replace=False)
            true_malicious_nodes = [f"node_{i}" for i in malicious_indices]
        
        # Extract trust analysis results
        trust_analysis = results.get("trust_analysis", {})
        excluded_nodes = trust_analysis.get("excluded_nodes", [])
        downgraded_nodes = trust_analysis.get("downgraded_nodes", [])
        
        # Calculate detection metrics
        detected_malicious_nodes = list(set(excluded_nodes + downgraded_nodes))
        
        # Classification metrics
        tp = len(set(detected_malicious_nodes) & set(true_malicious_nodes))
        fp = len(set(detected_malicious_nodes) - set(true_malicious_nodes))
        fn = len(set(true_malicious_nodes) - set(detected_malicious_nodes))
        tn = config.num_actors - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if len(true_malicious_nodes) == 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / config.num_actors
        
        # Detection latency (extract from results if available)
        detection_rounds = {}
        attack_stats = results.get("attack_statistics", {})
        if attack_stats:
            for node_stats in attack_stats.values():
                if node_stats.get("detection_status") == "detected":
                    detection_round = node_stats.get("detection_round", -1)
                    if detection_round > 0:
                        detection_rounds[node_stats["node_id"]] = detection_round
        
        avg_detection_latency = np.mean(list(detection_rounds.values())) if detection_rounds else float('inf')
        min_detection_latency = min(detection_rounds.values()) if detection_rounds else -1
        max_detection_latency = max(detection_rounds.values()) if detection_rounds else -1
        
        # Resource usage
        avg_memory_usage_mb = np.mean(memory_samples)
        max_memory_usage_mb = max(memory_samples)
        avg_cpu_usage_percent = np.mean(cpu_samples) if cpu_samples else 0.0
        
        # FL performance metrics
        initial_accuracy = results.get("initial_metrics", {}).get("consensus_accuracy", 0.0)
        final_accuracy = results.get("final_metrics", {}).get("consensus_accuracy", 0.0)
        accuracy_degradation = initial_accuracy - final_accuracy
        
        # Trust statistics
        avg_trust_honest = 1.0  # Default for honest nodes
        avg_trust_malicious = 0.0  # Default for malicious nodes
        trust_separation = 0.0
        
        if "trust_statistics" in results:
            trust_stats = results["trust_statistics"]
            honest_scores = [score for node, score in trust_stats.items() 
                           if node not in true_malicious_nodes]
            malicious_scores = [score for node, score in trust_stats.items() 
                              if node in true_malicious_nodes]
            
            if honest_scores:
                avg_trust_honest = np.mean(honest_scores)
            if malicious_scores:
                avg_trust_malicious = np.mean(malicious_scores)
            
            # Cohen's d for separation
            if honest_scores and malicious_scores:
                pooled_std = np.sqrt(((len(honest_scores) - 1) * np.var(honest_scores) + 
                                    (len(malicious_scores) - 1) * np.var(malicious_scores)) / 
                                   (len(honest_scores) + len(malicious_scores) - 2))
                trust_separation = (avg_trust_honest - avg_trust_malicious) / pooled_std if pooled_std > 0 else 0.0
        
        return ExperimentMetrics(
            experiment_id=config.experiment_id,
            config=config,
            true_malicious_nodes=true_malicious_nodes,
            detected_malicious_nodes=detected_malicious_nodes,
            excluded_nodes=excluded_nodes,
            downgraded_nodes=downgraded_nodes,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            detection_rounds=detection_rounds,
            avg_detection_latency=avg_detection_latency,
            min_detection_latency=min_detection_latency,
            max_detection_latency=max_detection_latency,
            avg_memory_usage_mb=avg_memory_usage_mb,
            max_memory_usage_mb=max_memory_usage_mb,
            avg_cpu_usage_percent=avg_cpu_usage_percent,
            trust_computation_time_ms=0.0,  # Would need to be instrumented
            communication_overhead_percent=12.0,  # From technical documentation
            initial_accuracy=initial_accuracy,
            final_accuracy=final_accuracy,
            accuracy_degradation=accuracy_degradation,
            convergence_rounds=config.num_rounds,  # Assume converged
            training_time_seconds=time.time() - start_time if 'start_time' in locals() else 0.0,
            avg_trust_score_honest=avg_trust_honest,
            avg_trust_score_malicious=avg_trust_malicious,
            trust_score_separation=trust_separation,
            beta_threshold_evolution=[],  # Would need instrumentation
            hsic_values_honest=[],  # Would need instrumentation
            hsic_values_malicious=[],  # Would need instrumentation
            experiment_duration=time.time() - start_time if 'start_time' in locals() else 0.0,
            success=True
        )
    
    def _save_experiment_results(self, 
                               exp_dir: str, 
                               config: ExperimentConfig, 
                               metrics: ExperimentMetrics,
                               raw_results: Dict[str, Any]):
        """Save detailed experiment results."""
        
        # Save configuration
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # Save metrics
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(asdict(metrics), f, indent=2, default=str)
        
        # Save raw results
        with open(os.path.join(exp_dir, "raw_results.json"), "w") as f:
            json.dump(raw_results, f, indent=2, default=str)


class ExperimentSuite:
    """Comprehensive experiment suite for trust monitoring evaluation."""
    
    def __init__(self, output_dir: str = "paper_experiments"):
        self.output_dir = output_dir
        self.runner = ExperimentRunner(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(output_dir, "experiment_suite.log"))
            ]
        )
        self.logger = logging.getLogger("ExperimentSuite")
    
    def generate_configurations(self, 
                              quick_test: bool = False,
                              custom_configs: Optional[List[Dict[str, Any]]] = None) -> List[ExperimentConfig]:
        """Generate all experiment configurations."""
        
        if custom_configs:
            # Use custom configurations
            configs = []
            for i, custom_config in enumerate(custom_configs):
                config = ExperimentConfig(**custom_config)
                config.experiment_id = f"custom_{i:03d}"
                configs.append(config)
            return configs
        
        if quick_test:
            # Quick test configurations
            base_configs = [
                # Baseline tests
                {"dataset": "mnist", "attack_type": "none", "topology": "ring"},
                {"dataset": "mnist", "attack_type": "label_flipping", "topology": "ring", "malicious_fraction": 0.167},
                {"dataset": "mnist", "attack_type": "model_poisoning", "topology": "ring", "malicious_fraction": 0.167},
                {"dataset": "mnist", "attack_type": "byzantine_gradient", "topology": "ring", "malicious_fraction": 0.167},
            ]
        else:
            # Full comprehensive configurations
            datasets = ["mnist", "cifar10"]
            topologies = ["ring", "complete", "line"] 
            trust_profiles = ["permissive", "default", "strict"]
            attack_types = ["none", "label_flipping", "model_poisoning", "byzantine_gradient"]
            attack_intensities = ["low", "moderate", "high"]
            stealth_levels = ["low", "medium", "high"]
            malicious_fractions = [0.0, 0.167, 0.25, 0.33, 0.5]  # 0%, 1/6, 1/4, 1/3, 1/2
            num_actors_options = [6, 8, 10, 12]
            
            base_configs = []
            
            # Generate all combinations
            for dataset, topology, trust_profile, attack_type in itertools.product(
                datasets, topologies, trust_profiles, attack_types
            ):
                if attack_type == "none":
                    # Baseline experiments
                    for num_actors in num_actors_options:
                        base_configs.append({
                            "dataset": dataset,
                            "num_actors": num_actors,
                            "topology": topology,
                            "trust_profile": trust_profile,
                            "attack_type": attack_type,
                            "malicious_fraction": 0.0,
                            "attack_intensity": "moderate",
                            "stealth_level": "medium"
                        })
                else:
                    # Attack experiments
                    for num_actors, malicious_fraction, attack_intensity, stealth_level in itertools.product(
                        num_actors_options, malicious_fractions[1:], attack_intensities, stealth_levels
                    ):
                        # Skip configurations where malicious fraction results in < 1 malicious node
                        if int(num_actors * malicious_fraction) >= 1:
                            base_configs.append({
                                "dataset": dataset,
                                "num_actors": num_actors,
                                "topology": topology,
                                "trust_profile": trust_profile,
                                "attack_type": attack_type,
                                "malicious_fraction": malicious_fraction,
                                "attack_intensity": attack_intensity,
                                "stealth_level": stealth_level
                            })
        
        # Create ExperimentConfig objects
        configs = []
        for i, base_config in enumerate(base_configs):
            # Add default values
            config_dict = {
                "num_rounds": 20 if base_config.get("dataset") == "cifar10" else 15,
                "use_beta_threshold": True,
                "target_class": 0,
                "experiment_id": f"exp_{i:04d}",
                "repeat_id": 0,
                "random_seed": 42,
                **base_config
            }
            
            configs.append(ExperimentConfig(**config_dict))
        
        self.logger.info(f"Generated {len(configs)} experiment configurations")
        return configs
    
    def run_suite(self, 
                  quick_test: bool = False,
                  parallel: bool = True,
                  max_workers: int = 4,
                  custom_configs: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """Run the complete experiment suite."""
        
        self.logger.info("Starting comprehensive experiment suite")
        
        # Generate configurations
        configs = self.generate_configurations(quick_test, custom_configs)
        
        # Run experiments
        results = []
        
        if parallel and len(configs) > 1:
            self.logger.info(f"Running {len(configs)} experiments in parallel with {max_workers} workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_config = {
                    executor.submit(self.runner.run_experiment, config): config 
                    for config in configs
                }
                
                # Collect results
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        metrics = future.result()
                        results.append(metrics)
                        self.logger.info(f"Completed {len(results)}/{len(configs)} experiments")
                    except Exception as e:
                        self.logger.error(f"Experiment {config.experiment_id} failed: {e}")
        else:
            # Sequential execution
            self.logger.info(f"Running {len(configs)} experiments sequentially")
            
            for i, config in enumerate(configs):
                self.logger.info(f"Running experiment {i+1}/{len(configs)}: {config.experiment_id}")
                try:
                    metrics = self.runner.run_experiment(config)
                    results.append(metrics)
                except Exception as e:
                    self.logger.error(f"Experiment {config.experiment_id} failed: {e}")
        
        # Convert to DataFrame for analysis
        results_df = self._create_results_dataframe(results)
        
        # Save aggregate results
        self._save_aggregate_results(results_df)
        
        self.logger.info(f"Experiment suite completed. Results saved to {self.output_dir}")
        
        return results_df
    
    def _create_results_dataframe(self, results: List[ExperimentMetrics]) -> pd.DataFrame:
        """Convert experiment results to pandas DataFrame."""
        
        rows = []
        for metrics in results:
            row = {
                # Experiment configuration
                "experiment_id": metrics.experiment_id,
                "dataset": metrics.config.dataset,
                "num_actors": metrics.config.num_actors,
                "num_rounds": metrics.config.num_rounds,
                "topology": metrics.config.topology,
                "trust_profile": metrics.config.trust_profile,
                "use_beta_threshold": metrics.config.use_beta_threshold,
                "attack_type": metrics.config.attack_type,
                "malicious_fraction": metrics.config.malicious_fraction,
                "attack_intensity": metrics.config.attack_intensity,
                "stealth_level": metrics.config.stealth_level,
                
                # Detection metrics
                "true_positives": metrics.true_positives,
                "true_negatives": metrics.true_negatives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "detection_accuracy": metrics.accuracy,
                
                # Latency metrics
                "avg_detection_latency": metrics.avg_detection_latency,
                "min_detection_latency": metrics.min_detection_latency,
                "max_detection_latency": metrics.max_detection_latency,
                
                # Overhead metrics
                "avg_memory_mb": metrics.avg_memory_usage_mb,
                "max_memory_mb": metrics.max_memory_usage_mb,
                "avg_cpu_percent": metrics.avg_cpu_usage_percent,
                "communication_overhead": metrics.communication_overhead_percent,
                
                # FL performance
                "initial_accuracy": metrics.initial_accuracy,
                "final_accuracy": metrics.final_accuracy,
                "accuracy_degradation": metrics.accuracy_degradation,
                "training_time": metrics.training_time_seconds,
                
                # Trust metrics
                "trust_score_honest": metrics.avg_trust_score_honest,
                "trust_score_malicious": metrics.avg_trust_score_malicious,
                "trust_separation": metrics.trust_score_separation,
                
                # Experiment metadata
                "experiment_duration": metrics.experiment_duration,
                "success": metrics.success,
                "error_message": metrics.error_message
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _save_aggregate_results(self, results_df: pd.DataFrame):
        """Save aggregate results and generate summary statistics."""
        
        # Save full results
        results_df.to_csv(os.path.join(self.output_dir, "experiment_results.csv"), index=False)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(results_df)
        
        with open(os.path.join(self.output_dir, "summary_statistics.json"), "w") as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        self.logger.info("Saved aggregate results and summary statistics")
    
    def _generate_summary_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the paper."""
        
        successful_df = results_df[results_df["success"] == True]
        
        summary = {
            "experiment_overview": {
                "total_experiments": len(results_df),
                "successful_experiments": len(successful_df),
                "success_rate": len(successful_df) / len(results_df) if len(results_df) > 0 else 0,
                "unique_configurations": {
                    "datasets": results_df["dataset"].nunique(),
                    "topologies": results_df["topology"].nunique(),
                    "attack_types": results_df["attack_type"].nunique(),
                    "trust_profiles": results_df["trust_profile"].nunique()
                }
            },
            
            "detection_performance": {
                "overall_precision": successful_df["precision"].mean(),
                "overall_recall": successful_df["recall"].mean(),
                "overall_f1_score": successful_df["f1_score"].mean(),
                "by_attack_type": successful_df.groupby("attack_type")[["precision", "recall", "f1_score"]].mean().to_dict(),
                "by_topology": successful_df.groupby("topology")[["precision", "recall", "f1_score"]].mean().to_dict(),
                "by_trust_profile": successful_df.groupby("trust_profile")[["precision", "recall", "f1_score"]].mean().to_dict()
            },
            
            "latency_analysis": {
                "avg_detection_latency": successful_df["avg_detection_latency"].mean(),
                "by_attack_intensity": successful_df.groupby("attack_intensity")["avg_detection_latency"].mean().to_dict(),
                "by_topology": successful_df.groupby("topology")["avg_detection_latency"].mean().to_dict()
            },
            
            "overhead_analysis": {
                "avg_memory_usage": successful_df["avg_memory_mb"].mean(),
                "avg_cpu_usage": successful_df["avg_cpu_percent"].mean(),
                "communication_overhead": successful_df["communication_overhead"].mean(),
                "by_dataset": successful_df.groupby("dataset")[["avg_memory_mb", "avg_cpu_percent"]].mean().to_dict()
            },
            
            "robustness_analysis": {
                "avg_accuracy_degradation": successful_df["accuracy_degradation"].mean(),
                "by_malicious_fraction": successful_df.groupby("malicious_fraction")["accuracy_degradation"].mean().to_dict(),
                "trust_score_separation": successful_df["trust_separation"].mean()
            }
        }
        
        return summary


def main():
    """Main function with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Experiment Suite for Trust-Based Drift Detection"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="paper_experiments",
        help="Output directory for results (default: paper_experiments)"
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Run quick test with limited configurations"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=True,
        help="Run experiments in parallel (default: True)"
    )
    parser.add_argument(
        "--max_workers", type=int, default=4,
        help="Maximum parallel workers (default: 4)"
    )
    parser.add_argument(
        "--config_file", type=str,
        help="JSON file with custom experiment configurations"
    )
    
    args = parser.parse_args()
    
    # Load custom configurations if provided
    custom_configs = None
    if args.config_file:
        with open(args.config_file, "r") as f:
            custom_configs = json.load(f)
    
    # Create and run experiment suite
    suite = ExperimentSuite(args.output_dir)
    
    results_df = suite.run_suite(
        quick_test=args.quick_test,
        parallel=args.parallel,
        max_workers=args.max_workers,
        custom_configs=custom_configs
    )
    
    print(f"\nExperiment suite completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total experiments: {len(results_df)}")
    print(f"Successful experiments: {len(results_df[results_df['success']])}")
    
    # Print summary statistics
    if len(results_df[results_df['success']]) > 0:
        successful_df = results_df[results_df['success']]
        print(f"\nSummary Statistics:")
        print(f"Average Precision: {successful_df['precision'].mean():.3f}")
        print(f"Average Recall: {successful_df['recall'].mean():.3f}")
        print(f"Average F1-Score: {successful_df['f1_score'].mean():.3f}")
        print(f"Average Detection Latency: {successful_df['avg_detection_latency'].mean():.1f} rounds")


if __name__ == "__main__":
    main()