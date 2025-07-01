#!/usr/bin/env python3
"""
Paper Experiment Suite for Trust-Based Drift Detection.

This suite runs systematic experiments using our existing MNIST and CIFAR-10 
scripts and collects comprehensive metrics for paper publication.
"""

import argparse
import json
import logging
import os
import subprocess
import time
import itertools
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiment_metrics_collector import MetricsCollector, save_metrics, analyze_metrics_batch


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    # Core parameters
    dataset: str  # "mnist", "cifar10"
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
    random_seed: int = 42


class PaperExperimentSuite:
    """Comprehensive experiment suite for trust monitoring paper."""
    
    def __init__(self, output_dir: str = "paper_experiments"):
        self.output_dir = output_dir
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
        self.logger = logging.getLogger("PaperExperimentSuite")
        
        # Track experiment results
        self.experiment_results = []
    
    def generate_experiment_configurations(self, 
                                         quick_test: bool = False,
                                         focused_study: str = None) -> List[ExperimentConfig]:
        """Generate systematic experiment configurations."""
        
        configs = []
        
        if quick_test:
            # Quick validation tests
            base_configs = [
                # Baseline (no attack)
                {"dataset": "mnist", "attack_type": "none", "malicious_fraction": 0.0},
                # Basic attacks
                {"dataset": "mnist", "attack_type": "label_flipping", "malicious_fraction": 0.167},
                {"dataset": "mnist", "attack_type": "model_poisoning", "malicious_fraction": 0.167},
                {"dataset": "mnist", "attack_type": "byzantine_gradient", "malicious_fraction": 0.167},
                # Different topology
                {"dataset": "mnist", "attack_type": "label_flipping", "malicious_fraction": 0.167, "topology": "complete"},
            ]
        
        elif focused_study == "topology_comparison":
            # Focus on topology effects
            topologies = ["ring", "complete", "line"]
            attack_types = ["label_flipping", "model_poisoning", "byzantine_gradient"]
            
            base_configs = []
            for topology, attack_type in itertools.product(topologies, attack_types):
                base_configs.extend([
                    # Different malicious fractions
                    {"dataset": "mnist", "topology": topology, "attack_type": attack_type, "malicious_fraction": 0.167},
                    {"dataset": "mnist", "topology": topology, "attack_type": attack_type, "malicious_fraction": 0.25},
                    {"dataset": "mnist", "topology": topology, "attack_type": attack_type, "malicious_fraction": 0.33},
                ])
        
        elif focused_study == "attack_intensity":
            # Focus on attack intensity effects
            intensities = ["low", "moderate", "high"]
            stealth_levels = ["low", "medium", "high"]
            
            base_configs = []
            for intensity, stealth in itertools.product(intensities, stealth_levels):
                base_configs.extend([
                    {"dataset": "mnist", "attack_type": "label_flipping", "attack_intensity": intensity, "stealth_level": stealth, "malicious_fraction": 0.25},
                    {"dataset": "mnist", "attack_type": "model_poisoning", "attack_intensity": intensity, "stealth_level": stealth, "malicious_fraction": 0.25},
                    {"dataset": "mnist", "attack_type": "byzantine_gradient", "attack_intensity": intensity, "stealth_level": stealth, "malicious_fraction": 0.25},
                ])
        
        elif focused_study == "scalability":
            # Focus on scalability with different network sizes
            num_actors_options = [6, 8, 10, 12, 16]
            
            base_configs = []
            for num_actors in num_actors_options:
                base_configs.extend([
                    # Baseline
                    {"dataset": "mnist", "num_actors": num_actors, "attack_type": "none", "malicious_fraction": 0.0},
                    # Fixed malicious fraction
                    {"dataset": "mnist", "num_actors": num_actors, "attack_type": "label_flipping", "malicious_fraction": 0.25},
                    {"dataset": "mnist", "num_actors": num_actors, "attack_type": "model_poisoning", "malicious_fraction": 0.25},
                    # Test different topologies
                    {"dataset": "mnist", "num_actors": num_actors, "attack_type": "label_flipping", "malicious_fraction": 0.25, "topology": "complete"},
                ])
        
        elif focused_study == "cross_dataset":
            # Compare MNIST vs CIFAR-10
            datasets = ["mnist", "cifar10"]
            attack_types = ["label_flipping", "model_poisoning", "byzantine_gradient"]
            
            base_configs = []
            for dataset, attack_type in itertools.product(datasets, attack_types):
                base_configs.extend([
                    # Baseline
                    {"dataset": dataset, "attack_type": "none", "malicious_fraction": 0.0},
                    # Standard attack
                    {"dataset": dataset, "attack_type": attack_type, "malicious_fraction": 0.25},
                    # High intensity
                    {"dataset": dataset, "attack_type": attack_type, "malicious_fraction": 0.33, "attack_intensity": "high"},
                ])
        
        else:
            # Comprehensive study (all combinations)
            datasets = ["mnist", "cifar10"]
            topologies = ["ring", "complete", "line"]
            trust_profiles = ["default", "strict"]  # Skip permissive for paper focus
            attack_types = ["none", "label_flipping", "model_poisoning", "byzantine_gradient"]
            intensities = ["low", "moderate", "high"]
            malicious_fractions = [0.0, 0.167, 0.25, 0.33, 0.5]
            num_actors_options = [6, 8, 10]
            
            base_configs = []
            
            for dataset, topology, trust_profile, num_actors in itertools.product(
                datasets, topologies, trust_profiles, num_actors_options
            ):
                # Baseline experiments
                base_configs.append({
                    "dataset": dataset, "topology": topology, "trust_profile": trust_profile,
                    "num_actors": num_actors, "attack_type": "none", "malicious_fraction": 0.0
                })
                
                # Attack experiments
                for attack_type, malicious_fraction, intensity in itertools.product(
                    attack_types[1:], malicious_fractions[1:], intensities
                ):
                    # Skip configurations with < 1 malicious node
                    if int(num_actors * malicious_fraction) >= 1:
                        base_configs.append({
                            "dataset": dataset, "topology": topology, "trust_profile": trust_profile,
                            "num_actors": num_actors, "attack_type": attack_type,
                            "malicious_fraction": malicious_fraction, "attack_intensity": intensity
                        })
        
        # Convert to ExperimentConfig objects with defaults
        for i, base_config in enumerate(base_configs):
            config_dict = {
                # Defaults
                "dataset": "mnist",
                "num_actors": 6,
                "num_rounds": 15,  # Will be adjusted for CIFAR-10
                "topology": "ring",
                "trust_profile": "default",
                "use_beta_threshold": True,
                "attack_type": "none",
                "malicious_fraction": 0.0,
                "attack_intensity": "moderate",
                "stealth_level": "medium",
                "target_class": 0,
                "experiment_id": f"exp_{i:04d}",
                "random_seed": 42,
                # Override with specific config
                **base_config
            }
            
            # Adjust rounds for CIFAR-10
            if config_dict["dataset"] == "cifar10":
                config_dict["num_rounds"] = 20
            
            configs.append(ExperimentConfig(**config_dict))
        
        self.logger.info(f"Generated {len(configs)} experiment configurations")
        return configs
    
    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment using existing scripts."""
        
        start_time = time.time()
        exp_dir = os.path.join(self.output_dir, config.experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        try:
            # Determine script to run
            if config.dataset == "mnist":
                script_path = "murmura/examples/adaptive_trust_mnist_example.py"
            elif config.dataset == "cifar10":
                script_path = "murmura/examples/adaptive_trust_cifar10_example.py"
            else:
                raise ValueError(f"Unknown dataset: {config.dataset}")
            
            # Build command arguments
            cmd_args = [
                "--num_actors", str(config.num_actors),
                "--num_rounds", str(config.num_rounds),
                "--topology", config.topology,
                "--trust_profile", config.trust_profile,
                "--attack_type", config.attack_type,
                "--malicious_fraction", str(config.malicious_fraction),
                "--attack_intensity", config.attack_intensity,
                "--stealth_level", config.stealth_level,
                "--target_class", str(config.target_class),
                "--output_dir", exp_dir,
                "--log_level", "WARNING"  # Reduce log noise
            ]
            
            if not config.use_beta_threshold:
                cmd_args.append("--disable_beta")
            
            # Run experiment
            cmd = ["python", script_path] + cmd_args
            self.logger.info(f"Running: {config.experiment_id} ({config.dataset}, {config.attack_type})")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode != 0:
                raise RuntimeError(f"Experiment failed: {result.stderr}")
            
            # Load and analyze results
            results = self._load_experiment_results(exp_dir)
            
            # Extract metrics using our collector
            experiment_config_dict = asdict(config)
            if results:  # Only add attack_config if we have results
                experiment_config_dict["attack_config"] = {
                    "attack_type": config.attack_type,
                    "malicious_fraction": config.malicious_fraction,
                    "attack_intensity": config.attack_intensity,
                    "stealth_level": config.stealth_level
                }
            
            # Create a simplified metrics collector for post-processing
            metrics_collector = MetricsCollector(experiment_config_dict)
            comprehensive_metrics = metrics_collector.finalize_metrics(results)
            
            # Save comprehensive metrics
            save_metrics(comprehensive_metrics, os.path.join(exp_dir, "comprehensive_metrics.json"))
            
            # Extract key metrics for summary
            trust_metrics = comprehensive_metrics["trust_metrics"]
            performance_metrics = comprehensive_metrics["performance_metrics"]
            
            summary_metrics = {
                "experiment_id": config.experiment_id,
                "config": asdict(config),
                "success": True,
                "duration": time.time() - start_time,
                
                # Key metrics for paper
                "precision": trust_metrics["precision"],
                "recall": trust_metrics["recall"],
                "f1_score": trust_metrics["f1_score"],
                "detection_accuracy": trust_metrics["accuracy"],
                "avg_detection_latency": trust_metrics["avg_detection_latency"],
                "false_positive_rate": trust_metrics["false_positives"] / max(1, trust_metrics["false_positives"] + trust_metrics["true_negatives"]),
                
                "initial_accuracy": performance_metrics["initial_accuracy"],
                "final_accuracy": performance_metrics["final_accuracy"],
                "accuracy_degradation": performance_metrics["accuracy_degradation"],
                
                "avg_memory_mb": performance_metrics["avg_memory_mb"],
                "peak_memory_mb": performance_metrics["peak_memory_mb"],
                "avg_cpu_percent": performance_metrics["avg_cpu_percent"],
                "trust_overhead_mb": performance_metrics["trust_memory_overhead_mb"],
                "communication_overhead_percent": performance_metrics["communication_overhead_percent"],
                
                "trust_score_separation": trust_metrics["trust_score_separation"],
                "avg_trust_honest": trust_metrics["avg_trust_honest"],
                "avg_trust_malicious": trust_metrics["avg_trust_malicious"],
            }
            
            self.logger.info(f"Completed {config.experiment_id}: P={trust_metrics['precision']:.3f}, "
                           f"R={trust_metrics['recall']:.3f}, Latency={trust_metrics['avg_detection_latency']:.1f}")
            
            return summary_metrics
            
        except Exception as e:
            self.logger.error(f"Experiment {config.experiment_id} failed: {str(e)}")
            return {
                "experiment_id": config.experiment_id,
                "config": asdict(config),
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "detection_accuracy": 0.0, "avg_detection_latency": float('inf'),
                "false_positive_rate": 0.0,
                "initial_accuracy": 0.0, "final_accuracy": 0.0, "accuracy_degradation": 0.0,
                "avg_memory_mb": 0.0, "peak_memory_mb": 0.0, "avg_cpu_percent": 0.0,
                "trust_overhead_mb": 0.0, "communication_overhead_percent": 0.0,
                "trust_score_separation": 0.0, "avg_trust_honest": 0.0, "avg_trust_malicious": 0.0
            }
    
    def _load_experiment_results(self, exp_dir: str) -> Dict[str, Any]:
        """Load experiment results from output directory."""
        # Look for results files
        result_files = [f for f in os.listdir(exp_dir) 
                       if f.startswith("adaptive_trust_results_") and f.endswith(".json")]
        
        if not result_files:
            return {}
        
        # Load the most recent results
        latest_file = sorted(result_files)[-1]
        try:
            with open(os.path.join(exp_dir, latest_file), 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load results from {latest_file}: {e}")
            return {}
    
    def run_experiment_suite(self, 
                           configs: List[ExperimentConfig],
                           parallel: bool = True,
                           max_workers: int = 4) -> pd.DataFrame:
        """Run the complete experiment suite."""
        
        self.logger.info(f"Starting experiment suite with {len(configs)} experiments")
        
        results = []
        
        if parallel and len(configs) > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {executor.submit(self.run_single_experiment, config): config 
                                  for config in configs}
                
                for future in as_completed(future_to_config):
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.info(f"Progress: {len(results)}/{len(configs)} completed")
                    except Exception as e:
                        config = future_to_config[future]
                        self.logger.error(f"Failed to get result for {config.experiment_id}: {e}")
        else:
            # Sequential execution
            for i, config in enumerate(configs):
                self.logger.info(f"Running experiment {i+1}/{len(configs)}: {config.experiment_id}")
                result = self.run_single_experiment(config)
                results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv(os.path.join(self.output_dir, "experiment_results.csv"), index=False)
        
        # Generate summary statistics
        self._generate_paper_statistics(df)
        
        self.logger.info(f"Experiment suite completed. Results saved to {self.output_dir}")
        return df
    
    def _generate_paper_statistics(self, df: pd.DataFrame):
        """Generate summary statistics for paper."""
        
        successful_df = df[df["success"] == True]
        
        if len(successful_df) == 0:
            self.logger.warning("No successful experiments to analyze")
            return
        
        # Overall performance
        overall_stats = {
            "total_experiments": len(df),
            "successful_experiments": len(successful_df),
            "success_rate": len(successful_df) / len(df),
            
            "detection_performance": {
                "avg_precision": successful_df["precision"].mean(),
                "std_precision": successful_df["precision"].std(),
                "avg_recall": successful_df["recall"].mean(),
                "std_recall": successful_df["recall"].std(),
                "avg_f1_score": successful_df["f1_score"].mean(),
                "std_f1_score": successful_df["f1_score"].std(),
                "avg_detection_latency": successful_df[successful_df["avg_detection_latency"] != float('inf')]["avg_detection_latency"].mean(),
            },
            
            "overhead_analysis": {
                "avg_memory_mb": successful_df["avg_memory_mb"].mean(),
                "avg_trust_overhead_mb": successful_df["trust_overhead_mb"].mean(),
                "avg_cpu_percent": successful_df["avg_cpu_percent"].mean(),
                "communication_overhead": successful_df["communication_overhead_percent"].mean(),
            },
            
            "robustness_analysis": {
                "avg_accuracy_degradation": successful_df["accuracy_degradation"].mean(),
                "avg_trust_separation": successful_df["trust_score_separation"].mean(),
            }
        }
        
        # By attack type
        attack_analysis = {}
        for attack_type in successful_df["config"].apply(lambda x: x["attack_type"]).unique():
            attack_df = successful_df[successful_df["config"].apply(lambda x: x["attack_type"]) == attack_type]
            attack_analysis[attack_type] = {
                "count": len(attack_df),
                "avg_precision": attack_df["precision"].mean(),
                "avg_recall": attack_df["recall"].mean(),
                "avg_f1_score": attack_df["f1_score"].mean(),
                "avg_detection_latency": attack_df[attack_df["avg_detection_latency"] != float('inf')]["avg_detection_latency"].mean() if len(attack_df[attack_df["avg_detection_latency"] != float('inf')]) > 0 else None,
            }
        
        # By topology
        topology_analysis = {}
        for topology in successful_df["config"].apply(lambda x: x["topology"]).unique():
            topo_df = successful_df[successful_df["config"].apply(lambda x: x["topology"]) == topology]
            topology_analysis[topology] = {
                "count": len(topo_df),
                "avg_precision": topo_df["precision"].mean(),
                "avg_recall": topo_df["recall"].mean(),
                "avg_memory_mb": topo_df["avg_memory_mb"].mean(),
                "avg_detection_latency": topo_df[topo_df["avg_detection_latency"] != float('inf')]["avg_detection_latency"].mean() if len(topo_df[topo_df["avg_detection_latency"] != float('inf')]) > 0 else None,
            }
        
        paper_stats = {
            "overall": overall_stats,
            "by_attack_type": attack_analysis,
            "by_topology": topology_analysis,
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save paper statistics
        with open(os.path.join(self.output_dir, "paper_statistics.json"), "w") as f:
            json.dump(paper_stats, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("PAPER EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        print(f"Total Experiments: {overall_stats['total_experiments']}")
        print(f"Successful: {overall_stats['successful_experiments']} ({overall_stats['success_rate']:.1%})")
        print(f"\nDetection Performance:")
        print(f"  Average Precision: {overall_stats['detection_performance']['avg_precision']:.3f} ± {overall_stats['detection_performance']['std_precision']:.3f}")
        print(f"  Average Recall: {overall_stats['detection_performance']['avg_recall']:.3f} ± {overall_stats['detection_performance']['std_recall']:.3f}")
        print(f"  Average F1-Score: {overall_stats['detection_performance']['avg_f1_score']:.3f} ± {overall_stats['detection_performance']['std_f1_score']:.3f}")
        if overall_stats['detection_performance']['avg_detection_latency']:
            print(f"  Average Detection Latency: {overall_stats['detection_performance']['avg_detection_latency']:.1f} rounds")
        print(f"\nOverhead Analysis:")
        print(f"  Average Memory Usage: {overall_stats['overhead_analysis']['avg_memory_mb']:.1f} MB")
        print(f"  Trust Monitor Overhead: {overall_stats['overhead_analysis']['avg_trust_overhead_mb']:.1f} MB")
        print(f"  Average CPU Usage: {overall_stats['overhead_analysis']['avg_cpu_percent']:.1f}%")
        print(f"  Communication Overhead: {overall_stats['overhead_analysis']['communication_overhead']:.1f}%")
        print("="*60)


def main():
    """Main function with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="Paper Experiment Suite for Trust-Based Drift Detection"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="paper_experiments",
        help="Output directory for results (default: paper_experiments)"
    )
    parser.add_argument(
        "--study_type", type=str, default="comprehensive",
        choices=["quick_test", "topology_comparison", "attack_intensity", "scalability", "cross_dataset", "comprehensive"],
        help="Type of study to run (default: comprehensive)"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=True,
        help="Run experiments in parallel (default: True)"
    )
    parser.add_argument(
        "--max_workers", type=int, default=4,
        help="Maximum parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Create experiment suite
    suite = PaperExperimentSuite(args.output_dir)
    
    # Generate configurations based on study type
    if args.study_type == "quick_test":
        configs = suite.generate_experiment_configurations(quick_test=True)
    else:
        configs = suite.generate_experiment_configurations(focused_study=args.study_type if args.study_type != "comprehensive" else None)
    
    # Run experiments
    results_df = suite.run_experiment_suite(
        configs=configs,
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    
    print(f"\nExperiment suite completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"CSV file: {os.path.join(args.output_dir, 'experiment_results.csv')}")
    print(f"Statistics: {os.path.join(args.output_dir, 'paper_statistics.json')}")


if __name__ == "__main__":
    main()