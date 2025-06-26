#!/usr/bin/env python3
"""
Trust Monitoring Baseline Experiments

This script runs comprehensive baseline experiments across MNIST and CIFAR-10
to establish performance benchmarks for the adaptive trust monitoring system
before implementing attack scenarios.

The experiments test various configurations to understand:
1. Trust system overhead and performance impact
2. False positive rates across different configurations
3. Convergence behavior with different trust profiles
4. Scalability across network topologies and model complexities
5. Beta thresholding effectiveness vs manual thresholds
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

# Add the murmura package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Results storage
RESULTS_DIR = "trust_baseline_experiments"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class ExperimentRunner:
    """Manages and executes trust monitoring baseline experiments."""
    
    def __init__(self, base_output_dir: str = RESULTS_DIR):
        self.base_output_dir = Path(base_output_dir) / f"baseline_{TIMESTAMP}"
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_summary = {
            "experiment_start": datetime.now().isoformat(),
            "configurations": [],
            "results": [],
            "summary_stats": {},
        }
        
        # Experiment configurations
        self.mnist_configs = self._define_mnist_configs()
        self.cifar10_configs = self._define_cifar10_configs()
        
        print(f"🔬 Trust Baseline Experiments")
        print(f"📁 Results directory: {self.base_output_dir}")
        print(f"📊 Total experiments planned: {len(self.mnist_configs) + len(self.cifar10_configs)}")
    
    def _define_mnist_configs(self) -> List[Dict[str, Any]]:
        """Define MNIST experiment configurations."""
        configs = [
            # Basic topology comparisons
            {
                "name": "mnist_ring_baseline",
                "dataset": "mnist",
                "num_actors": 6,
                "num_rounds": 15,
                "topology": "ring",
                "trust_profile": "default",
                "use_beta": True,
                "description": "MNIST ring topology baseline with Beta thresholding"
            },
            {
                "name": "mnist_complete_baseline", 
                "dataset": "mnist",
                "num_actors": 6,
                "num_rounds": 15,
                "topology": "complete",
                "trust_profile": "default",
                "use_beta": True,
                "description": "MNIST complete topology baseline"
            },
            {
                "name": "mnist_line_baseline",
                "dataset": "mnist", 
                "num_actors": 6,
                "num_rounds": 15,
                "topology": "line",
                "trust_profile": "default",
                "use_beta": True,
                "description": "MNIST line topology baseline"
            },
            
            # Trust profile comparisons
            {
                "name": "mnist_permissive_trust",
                "dataset": "mnist",
                "num_actors": 6,
                "num_rounds": 15,
                "topology": "ring",
                "trust_profile": "permissive",
                "use_beta": True,
                "description": "MNIST with permissive trust profile"
            },
            {
                "name": "mnist_strict_trust",
                "dataset": "mnist",
                "num_actors": 6,
                "num_rounds": 15,
                "topology": "ring", 
                "trust_profile": "strict",
                "use_beta": True,
                "description": "MNIST with strict trust profile"
            },
            
            # Beta vs Manual thresholding
            {
                "name": "mnist_manual_threshold",
                "dataset": "mnist",
                "num_actors": 6,
                "num_rounds": 15,
                "topology": "ring",
                "trust_profile": "default",
                "use_beta": False,
                "description": "MNIST with manual thresholding (no Beta adaptation)"
            },
            
            # Scalability tests
            {
                "name": "mnist_scale_small",
                "dataset": "mnist",
                "num_actors": 4,
                "num_rounds": 12,
                "topology": "ring",
                "trust_profile": "default",
                "use_beta": True,
                "description": "MNIST small scale (4 actors)"
            },
            {
                "name": "mnist_scale_large",
                "dataset": "mnist",
                "num_actors": 10,
                "num_rounds": 18,
                "topology": "ring",
                "trust_profile": "default", 
                "use_beta": True,
                "description": "MNIST large scale (10 actors)"
            },
            
            # Extended training
            {
                "name": "mnist_extended_training",
                "dataset": "mnist",
                "num_actors": 6,
                "num_rounds": 25,
                "topology": "ring",
                "trust_profile": "default",
                "use_beta": True,
                "description": "MNIST extended training (25 rounds)"
            },
        ]
        return configs
    
    def _define_cifar10_configs(self) -> List[Dict[str, Any]]:
        """Define CIFAR-10 experiment configurations."""
        configs = [
            # Basic model comparisons
            {
                "name": "cifar10_simple_baseline",
                "dataset": "cifar10",
                "num_actors": 6,
                "num_rounds": 20,
                "topology": "ring",
                "trust_profile": "default",
                "model_type": "simple",
                "use_beta": True,
                "description": "CIFAR-10 simple model baseline"
            },
            {
                "name": "cifar10_standard_baseline",
                "dataset": "cifar10",
                "num_actors": 6,
                "num_rounds": 20,
                "topology": "ring",
                "trust_profile": "default",
                "model_type": "standard",
                "use_beta": True,
                "description": "CIFAR-10 standard model baseline"
            },
            {
                "name": "cifar10_resnet_baseline",
                "dataset": "cifar10",
                "num_actors": 6,
                "num_rounds": 25,
                "topology": "ring",
                "trust_profile": "default",
                "model_type": "resnet",
                "use_beta": True,
                "description": "CIFAR-10 ResNet model baseline"
            },
            
            # Topology comparisons (standard model)
            {
                "name": "cifar10_complete_topology",
                "dataset": "cifar10",
                "num_actors": 6,
                "num_rounds": 20,
                "topology": "complete",
                "trust_profile": "default",
                "model_type": "standard",
                "use_beta": True,
                "description": "CIFAR-10 complete topology"
            },
            
            # Trust profile with complex model
            {
                "name": "cifar10_permissive_resnet",
                "dataset": "cifar10",
                "num_actors": 6,
                "num_rounds": 25,
                "topology": "ring",
                "trust_profile": "permissive",
                "model_type": "resnet",
                "use_beta": True,
                "description": "CIFAR-10 ResNet with permissive trust"
            },
            {
                "name": "cifar10_strict_standard",
                "dataset": "cifar10",
                "num_actors": 6,
                "num_rounds": 20,
                "topology": "ring",
                "trust_profile": "strict",
                "model_type": "standard", 
                "use_beta": True,
                "description": "CIFAR-10 standard model with strict trust"
            },
            
            # Beta vs Manual for complex models
            {
                "name": "cifar10_manual_threshold",
                "dataset": "cifar10",
                "num_actors": 6,
                "num_rounds": 20,
                "topology": "ring",
                "trust_profile": "default",
                "model_type": "standard",
                "use_beta": False,
                "description": "CIFAR-10 with manual thresholding"
            },
            
            # Scalability with complex models
            {
                "name": "cifar10_scale_large",
                "dataset": "cifar10",
                "num_actors": 8,
                "num_rounds": 22,
                "topology": "ring",
                "trust_profile": "default",
                "model_type": "standard",
                "use_beta": True,
                "description": "CIFAR-10 large scale (8 actors)"
            },
        ]
        return configs
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        print(f"\n🚀 Running experiment: {config['name']}")
        print(f"📝 Description: {config['description']}")
        
        start_time = time.time()
        experiment_dir = self.base_output_dir / config['name']
        experiment_dir.mkdir(exist_ok=True)
        
        try:
            # Build command based on dataset
            if config['dataset'] == 'mnist':
                cmd = self._build_mnist_command(config, experiment_dir)
            else:  # cifar10
                cmd = self._build_cifar10_command(config, experiment_dir)
            
            print(f"💻 Command: {' '.join(cmd)}")
            
            # Run experiment
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=Path(__file__).parent.parent
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                # Try to find the results JSON file
                results_files = list(experiment_dir.glob("*results*.json"))
                experiment_results = None
                
                if results_files:
                    with open(results_files[0], 'r') as f:
                        experiment_results = json.load(f)
                
                success_result = {
                    "config": config,
                    "status": "success",
                    "execution_time": execution_time,
                    "stdout": result.stdout[-2000:],  # Last 2000 chars
                    "stderr": result.stderr[-1000:] if result.stderr else "",
                    "results_file": str(results_files[0]) if results_files else None,
                    "experiment_results": experiment_results,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Extract key metrics
                if experiment_results:
                    fl_results = experiment_results.get("fl_results", {})
                    trust_analysis = experiment_results.get("trust_analysis", {})
                    
                    success_result.update({
                        "final_accuracy": fl_results.get("final_metrics", {}).get("accuracy", 0),
                        "accuracy_improvement": fl_results.get("accuracy_improvement", 0),
                        "total_excluded": trust_analysis.get("total_excluded", 0),
                        "total_downgraded": trust_analysis.get("total_downgraded", 0),
                        "avg_trust_score": trust_analysis.get("avg_trust_score", 1.0),
                        "false_positive_rate": trust_analysis.get("false_positive_rate", 0),
                    })
                
                print(f"✅ Success! Execution time: {execution_time:.2f}s")
                if experiment_results:
                    print(f"📈 Final accuracy: {success_result.get('final_accuracy', 0):.4f}")
                    print(f"🛡️ False positive rate: {success_result.get('false_positive_rate', 0):.3f}")
                
                return success_result
                
            else:
                error_result = {
                    "config": config,
                    "status": "failed",
                    "execution_time": execution_time,
                    "return_code": result.returncode,
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:],
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"❌ Failed! Return code: {result.returncode}")
                print(f"🔍 Error: {result.stderr[-500:]}")
                
                return error_result
                
        except subprocess.TimeoutExpired:
            timeout_result = {
                "config": config,
                "status": "timeout",
                "execution_time": 3600,
                "timestamp": datetime.now().isoformat()
            }
            print(f"⏰ Timeout after 1 hour")
            return timeout_result
            
        except Exception as e:
            exception_result = {
                "config": config, 
                "status": "exception",
                "execution_time": time.time() - start_time,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            print(f"💥 Exception: {str(e)}")
            return exception_result
    
    def _build_mnist_command(self, config: Dict[str, Any], output_dir: Path) -> List[str]:
        """Build command for MNIST experiment."""
        cmd = [
            "python", "murmura/examples/adaptive_trust_mnist_example.py",
            "--num_actors", str(config["num_actors"]),
            "--num_rounds", str(config["num_rounds"]),
            "--topology", config["topology"],
            "--trust_profile", config["trust_profile"],
            "--output_dir", str(output_dir),
            "--log_level", "INFO"
        ]
        
        if not config.get("use_beta", True):
            cmd.append("--disable_beta")
        
        return cmd
    
    def _build_cifar10_command(self, config: Dict[str, Any], output_dir: Path) -> List[str]:
        """Build command for CIFAR-10 experiment."""
        cmd = [
            "python", "murmura/examples/adaptive_trust_cifar10_example.py",
            "--num_actors", str(config["num_actors"]),
            "--num_rounds", str(config["num_rounds"]),
            "--topology", config["topology"],
            "--trust_profile", config["trust_profile"],
            "--model_type", config.get("model_type", "standard"),
            "--output_dir", str(output_dir),
            "--log_level", "INFO"
        ]
        
        if not config.get("use_beta", True):
            cmd.append("--disable_beta")
        
        return cmd
    
    def run_all_experiments(self, 
                          include_mnist: bool = True, 
                          include_cifar10: bool = True,
                          specific_experiments: Optional[List[str]] = None) -> None:
        """Run all baseline experiments."""
        
        all_configs = []
        if include_mnist:
            all_configs.extend(self.mnist_configs)
        if include_cifar10:
            all_configs.extend(self.cifar10_configs)
        
        # Filter to specific experiments if requested
        if specific_experiments:
            all_configs = [c for c in all_configs if c["name"] in specific_experiments]
        
        print(f"\n🎯 Running {len(all_configs)} baseline experiments...")
        
        # Store configurations
        self.results_summary["configurations"] = all_configs
        
        successful_experiments = 0
        failed_experiments = 0
        
        for i, config in enumerate(all_configs, 1):
            print(f"\n{'='*80}")
            print(f"📊 Experiment {i}/{len(all_configs)}: {config['name']}")
            print(f"{'='*80}")
            
            result = self.run_single_experiment(config)
            self.results_summary["results"].append(result)
            
            if result["status"] == "success":
                successful_experiments += 1
            else:
                failed_experiments += 1
            
            # Save intermediate results
            self._save_results()
            
            print(f"\n📈 Progress: {successful_experiments} successful, {failed_experiments} failed")
        
        # Generate final summary
        self._generate_summary()
        print(f"\n🏁 All experiments completed!")
        print(f"📊 Results summary: {successful_experiments} successful, {failed_experiments} failed")
        print(f"📁 Results saved to: {self.base_output_dir}")
    
    def _save_results(self) -> None:
        """Save current results to file."""
        results_file = self.base_output_dir / "experiment_results.json"
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        serializable_results = convert_paths(self.results_summary)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _generate_summary(self) -> None:
        """Generate experiment summary statistics."""
        results = self.results_summary["results"]
        successful_results = [r for r in results if r["status"] == "success"]
        
        if not successful_results:
            print("⚠️ No successful experiments to summarize")
            return
        
        # Overall statistics
        summary_stats = {
            "total_experiments": len(results),
            "successful_experiments": len(successful_results),
            "failed_experiments": len(results) - len(successful_results),
            "average_execution_time": sum(r["execution_time"] for r in successful_results) / len(successful_results),
        }
        
        # Trust monitoring effectiveness
        trust_stats = {
            "average_false_positive_rate": sum(r.get("false_positive_rate", 0) for r in successful_results) / len(successful_results),
            "zero_false_positive_experiments": len([r for r in successful_results if r.get("false_positive_rate", 0) == 0]),
            "average_trust_score": sum(r.get("avg_trust_score", 1.0) for r in successful_results) / len(successful_results),
            "total_exclusions": sum(r.get("total_excluded", 0) for r in successful_results),
            "total_downgrades": sum(r.get("total_downgraded", 0) for r in successful_results),
        }
        
        # Performance statistics
        performance_stats = {
            "average_final_accuracy": sum(r.get("final_accuracy", 0) for r in successful_results) / len(successful_results),
            "average_accuracy_improvement": sum(r.get("accuracy_improvement", 0) for r in successful_results) / len(successful_results),
        }
        
        # Dataset-specific stats
        mnist_results = [r for r in successful_results if r["config"]["dataset"] == "mnist"]
        cifar10_results = [r for r in successful_results if r["config"]["dataset"] == "cifar10"]
        
        dataset_stats = {
            "mnist_experiments": len(mnist_results),
            "cifar10_experiments": len(cifar10_results),
        }
        
        if mnist_results:
            dataset_stats.update({
                "mnist_avg_accuracy": sum(r.get("final_accuracy", 0) for r in mnist_results) / len(mnist_results),
                "mnist_avg_false_positives": sum(r.get("false_positive_rate", 0) for r in mnist_results) / len(mnist_results),
            })
        
        if cifar10_results:
            dataset_stats.update({
                "cifar10_avg_accuracy": sum(r.get("final_accuracy", 0) for r in cifar10_results) / len(cifar10_results),
                "cifar10_avg_false_positives": sum(r.get("false_positive_rate", 0) for r in cifar10_results) / len(cifar10_results),
            })
        
        # Compile summary
        self.results_summary["summary_stats"] = {
            **summary_stats,
            "trust_monitoring": trust_stats,
            "performance": performance_stats,
            "datasets": dataset_stats,
            "experiment_end": datetime.now().isoformat()
        }
        
        # Save final results
        self._save_results()
        
        # Print summary
        print(f"\n📊 EXPERIMENT SUMMARY")
        print(f"{'='*50}")
        print(f"Total Experiments: {summary_stats['total_experiments']}")
        print(f"Successful: {summary_stats['successful_experiments']}")
        print(f"Failed: {summary_stats['failed_experiments']}")
        print(f"Average Execution Time: {summary_stats['average_execution_time']:.2f}s")
        print(f"\n🛡️ TRUST MONITORING EFFECTIVENESS")
        print(f"Average False Positive Rate: {trust_stats['average_false_positive_rate']:.4f}")
        print(f"Zero False Positive Experiments: {trust_stats['zero_false_positive_experiments']}/{len(successful_results)}")
        print(f"Average Trust Score: {trust_stats['average_trust_score']:.4f}")
        print(f"Total Exclusions Across All Experiments: {trust_stats['total_exclusions']}")
        print(f"Total Downgrades Across All Experiments: {trust_stats['total_downgrades']}")
        print(f"\n📈 FEDERATED LEARNING PERFORMANCE")
        print(f"Average Final Accuracy: {performance_stats['average_final_accuracy']:.4f}")
        print(f"Average Accuracy Improvement: {performance_stats['average_accuracy_improvement']:.4f}")
        
        if mnist_results and cifar10_results:
            print(f"\n📊 DATASET COMPARISON")
            print(f"MNIST Average Accuracy: {dataset_stats['mnist_avg_accuracy']:.4f}")
            print(f"CIFAR-10 Average Accuracy: {dataset_stats['cifar10_avg_accuracy']:.4f}")
            print(f"MNIST False Positive Rate: {dataset_stats['mnist_avg_false_positives']:.4f}")
            print(f"CIFAR-10 False Positive Rate: {dataset_stats['cifar10_avg_false_positives']:.4f}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive trust monitoring baseline experiments"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default=RESULTS_DIR,
        help=f"Base output directory (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--skip_mnist", action="store_true",
        help="Skip MNIST experiments"
    )
    parser.add_argument(
        "--skip_cifar10", action="store_true", 
        help="Skip CIFAR-10 experiments"
    )
    parser.add_argument(
        "--experiments", nargs="+", 
        help="Run only specific experiments (by name)"
    )
    parser.add_argument(
        "--list_experiments", action="store_true",
        help="List all available experiments and exit"
    )
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(args.output_dir)
    
    if args.list_experiments:
        print("\n📋 Available MNIST Experiments:")
        for config in runner.mnist_configs:
            print(f"  • {config['name']}: {config['description']}")
        
        print("\n📋 Available CIFAR-10 Experiments:")
        for config in runner.cifar10_configs:
            print(f"  • {config['name']}: {config['description']}")
        
        return
    
    # Run experiments
    runner.run_all_experiments(
        include_mnist=not args.skip_mnist,
        include_cifar10=not args.skip_cifar10,
        specific_experiments=args.experiments
    )


if __name__ == "__main__":
    main()