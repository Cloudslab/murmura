#!/usr/bin/env python3
"""
Automated Murmura Experimentation Script
========================================

This script implements the comprehensive experimentation plan for comparing
centralized, federated, and decentralized learning paradigms across diverse domains.

Hardware Configuration:
- 5x AWS G5.2XLARGE instances
- 1 GPU, 8 vCPUs, 32 GiB RAM per instance
- Total: 5 GPUs, 40 vCPUs, 160 GiB RAM

Experiment Design:
- Core Three-Way Comparison (Priority 1)
- Topology-Privacy Interaction Analysis (Priority 2)
- Scalability Analysis (Priority 3)
- Robustness Stress Tests (Priority 4)
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


class ExperimentConfig:
    """Configuration for the experiment runner."""

    def __init__(self):
        # Hardware constraints
        self.total_gpus = 5
        self.total_cpus = 40
        self.total_memory_gb = 160

        # Experiment parameters - optimized for hardware
        # Reduced from original plan to fit hardware constraints while maintaining quality
        self.paradigms = ["centralized", "federated", "decentralized"]
        self.topologies = ["star", "ring", "complete"]  # Removed "random" for time
        self.data_heterogeneity = ["iid", "moderate_noniid", "extreme_noniid"]
        self.privacy_levels = ["none", "moderate_dp", "strong_dp"]
        self.network_scales = [5, 10, 15]  # Reduced from [5, 10, 20] for faster execution

        # Training parameters - reduced for faster execution
        self.training_rounds = 5  # Reduced from 10 for faster execution
        self.local_epochs = 2
        self.batch_size = 32
        self.learning_rate = 0.001

        # Datasets
        self.datasets = ["mnist", "ham10000"]

        # Alpha values for Dirichlet distribution
        self.alpha_mapping = {
            "iid": 100.0,  # High alpha = more IID
            "moderate_noniid": 0.5,
            "extreme_noniid": 0.1
        }

        # Privacy parameters
        self.privacy_mapping = {
            "none": None,
            "moderate_dp": {"epsilon": 1.0, "delta": 1e-5},
            "strong_dp": {"epsilon": 0.1, "delta": 1e-6}
        }

        # Timeout settings (in seconds)
        self.experiment_timeout = 1800  # 30 minutes per experiment
        self.total_timeout = 86400  # 24 hours total


class ExperimentRunner:
    """Main experiment runner class."""

    def __init__(self, config: ExperimentConfig, output_dir: str = "./experiment_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Results storage
        self.results = []
        self.failed_experiments = []

        # Timing
        self.start_time = None
        self.experiment_count = 0
        self.total_experiments = 0

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Ray cluster info
        self.ray_address = os.environ.get("RAY_ADDRESS")
        if not self.ray_address:
            self.logger.warning("RAY_ADDRESS not set. Will attempt to auto-detect cluster.")

    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.output_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("ExperimentRunner")
        self.logger.info(f"Experiment runner initialized. Logging to {log_file}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Saving results and shutting down...")
        self.save_results()
        sys.exit(0)

    def generate_experiment_sets(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations based on the plan."""
        experiments = []

        # Set A: Core Three-Way Comparison (PRIORITY 1)
        self.logger.info("Generating Set A: Core Three-Way Comparison experiments")
        set_a = self.generate_set_a()
        experiments.extend(set_a)

        # Set B: Topology-Privacy Interaction (PRIORITY 2) - Reduced scope
        self.logger.info("Generating Set B: Topology-Privacy Interaction experiments")
        set_b = self.generate_set_b()
        experiments.extend(set_b)

        # Set C: Scalability Analysis (PRIORITY 3) - Reduced scope
        self.logger.info("Generating Set C: Scalability Analysis experiments")
        set_c = self.generate_set_c()
        experiments.extend(set_c)

        # Shuffle experiments to distribute load
        np.random.shuffle(experiments)

        self.logger.info(f"Generated {len(experiments)} total experiments")
        return experiments

    def generate_set_a(self) -> List[Dict[str, Any]]:
        """Generate Set A: Core Three-Way Comparison experiments."""
        experiments = []

        for dataset in self.config.datasets:
            for paradigm in self.config.paradigms:
                for topology in self.config.topologies:
                    for heterogeneity in self.config.data_heterogeneity:
                        for privacy in self.config.privacy_levels:
                            for scale in self.config.network_scales:
                                # Skip invalid combinations
                                if not self.is_valid_combination(paradigm, topology, privacy):
                                    continue

                                exp = {
                                    "set": "A",
                                    "dataset": dataset,
                                    "paradigm": paradigm,
                                    "topology": topology,
                                    "heterogeneity": heterogeneity,
                                    "privacy": privacy,
                                    "scale": scale,
                                    "priority": 1
                                }
                                experiments.append(exp)

        self.logger.info(f"Set A: Generated {len(experiments)} experiments")
        return experiments

    def generate_set_b(self) -> List[Dict[str, Any]]:
        """Generate Set B: Topology-Privacy Interaction experiments (reduced scope)."""
        experiments = []

        # Focus on federated learning for topology-privacy interactions
        for dataset in self.config.datasets:
            for topology in self.config.topologies:
                for privacy in self.config.privacy_levels:
                    if privacy == "none":  # Skip non-private experiments for this set
                        continue

                    exp = {
                        "set": "B",
                        "dataset": dataset,
                        "paradigm": "federated",  # Focus on federated for cleaner results
                        "topology": topology,
                        "heterogeneity": "moderate_noniid",  # Fixed heterogeneity
                        "privacy": privacy,
                        "scale": 10,  # Fixed scale
                        "priority": 2
                    }
                    experiments.append(exp)

        self.logger.info(f"Set B: Generated {len(experiments)} experiments")
        return experiments

    def generate_set_c(self) -> List[Dict[str, Any]]:
        """Generate Set C: Scalability Analysis experiments (reduced scope)."""
        experiments = []

        # Test scalability with ring topology and moderate non-IID
        for dataset in self.config.datasets:
            for paradigm in self.config.paradigms:
                for scale in [5, 10, 15, 20]:  # Extended scale range
                    exp = {
                        "set": "C",
                        "dataset": dataset,
                        "paradigm": paradigm,
                        "topology": "ring",  # Fixed topology
                        "heterogeneity": "moderate_noniid",  # Fixed heterogeneity
                        "privacy": "none",  # No privacy for cleaner scalability results
                        "scale": scale,
                        "priority": 3
                    }
                    experiments.append(exp)

        self.logger.info(f"Set C: Generated {len(experiments)} experiments")
        return experiments

    def is_valid_combination(self, paradigm: str, topology: str, privacy: str) -> bool:
        """Check if paradigm-topology-privacy combination is valid."""
        # Decentralized learning constraints
        if paradigm == "decentralized":
            # Decentralized only works with certain topologies
            if topology not in ["ring", "complete"]:
                return False
            # Decentralized with privacy requires local DP
            if privacy != "none":
                return True  # We'll handle local DP in the experiment

        # Centralized learning constraints
        if paradigm == "centralized":
            # Centralized typically uses star topology
            if topology != "star":
                return False

        return True

    def run_experiments(self, experiments: List[Dict[str, Any]], max_parallel: int = 2):
        """Run all experiments with proper resource management."""
        self.total_experiments = len(experiments)
        self.start_time = time.time()

        self.logger.info(f"Starting {self.total_experiments} experiments with max {max_parallel} parallel")

        # Group experiments by priority
        priority_groups = {}
        for exp in experiments:
            priority = exp.get("priority", 1)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(exp)

        # Run experiments by priority
        for priority in sorted(priority_groups.keys()):
            exp_group = priority_groups[priority]
            self.logger.info(f"Running Priority {priority} experiments ({len(exp_group)} experiments)")

            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                future_to_exp = {
                    executor.submit(self.run_single_experiment, exp): exp
                    for exp in exp_group
                }

                for future in as_completed(future_to_exp, timeout=self.config.total_timeout):
                    exp = future_to_exp[future]
                    try:
                        result = future.result(timeout=self.config.experiment_timeout)
                        if result:
                            self.results.append(result)
                            self.logger.info(f"Completed experiment {self.experiment_count + 1}/{self.total_experiments}")
                        else:
                            self.failed_experiments.append(exp)
                            self.logger.error(f"Experiment failed: {exp}")
                    except Exception as e:
                        self.logger.error(f"Experiment {exp} failed with error: {e}")
                        self.failed_experiments.append(exp)

                    self.experiment_count += 1

                    # Save progress every 10 experiments
                    if self.experiment_count % 10 == 0:
                        self.save_partial_results()

        self.logger.info(f"All experiments completed. {len(self.results)} successful, {len(self.failed_experiments)} failed")

    def run_single_experiment(self, exp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run a single experiment configuration."""
        exp_id = self.generate_experiment_id(exp)
        self.logger.info(f"Starting experiment: {exp_id}")

        try:
            # Build command
            cmd = self.build_experiment_command(exp)

            # Run experiment with timeout
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.experiment_timeout,
                check=False  # Don't raise exception on non-zero exit
            )

            execution_time = time.time() - start_time

            # Parse results
            if result.returncode == 0:
                metrics = self.parse_experiment_output(result.stdout)
                if metrics:
                    metrics.update(exp)
                    metrics["experiment_id"] = exp_id
                    metrics["execution_time"] = execution_time
                    metrics["timestamp"] = datetime.now().isoformat()

                    self.logger.info(f"Experiment {exp_id} completed successfully in {execution_time:.2f}s")
                    return metrics
                else:
                    self.logger.error(f"Experiment {exp_id} failed to parse output")
                    return None
            else:
                self.logger.error(f"Experiment {exp_id} failed with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"Experiment {exp_id} timed out after {self.config.experiment_timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"Experiment {exp_id} failed with exception: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def build_experiment_command(self, exp: Dict[str, Any]) -> List[str]:
        """Build the command to run a single experiment."""
        dataset = exp["dataset"]
        paradigm = exp["paradigm"]

        # Determine which script to use
        if dataset == "mnist":
            if paradigm == "decentralized":
                if exp["privacy"] != "none":
                    script = "murmura/examples/dp_decentralized_mnist_example.py"
                else:
                    script = "murmura/examples/decentralized_mnist_example.py"
            else:  # centralized or federated
                if exp["privacy"] != "none":
                    script = "murmura/examples/dp_mnist_example.py"
                else:
                    script = "murmura/examples/mnist_example.py"
        else:  # ham10000
            if paradigm == "decentralized":
                if exp["privacy"] != "none":
                    script = "murmura/examples/dp_decentralized_skin_lesion_example.py"
                else:
                    script = "murmura/examples/decentralized_skin_lesion_example.py"
            else:  # centralized or federated
                if exp["privacy"] != "none":
                    script = "murmura/examples/dp_skin_lesion_example.py"
                else:
                    script = "murmura/examples/skin_lesion_example.py"

        # Build command
        cmd = ["python", script]

        # Add common parameters
        cmd.extend([
            "--num_actors", str(exp["scale"]),
            "--rounds", str(self.config.training_rounds),
            "--epochs", str(self.config.local_epochs),
            "--batch_size", str(self.config.batch_size),
            "--lr", str(self.config.learning_rate),
            "--topology", exp["topology"],
            "--partition_strategy", "dirichlet" if exp["heterogeneity"] != "iid" else "iid",
            "--log_level", "WARNING",  # Reduce log verbosity
        ])

        # Add heterogeneity parameter
        if exp["heterogeneity"] != "iid":
            alpha = self.config.alpha_mapping[exp["heterogeneity"]]
            cmd.extend(["--alpha", str(alpha)])

        # Add privacy parameters
        if exp["privacy"] != "none":
            privacy_params = self.config.privacy_mapping[exp["privacy"]]
            cmd.extend([
                "--epsilon", str(privacy_params["epsilon"]),
                "--delta", str(privacy_params["delta"])
            ])

        # Add Ray cluster parameters
        if self.ray_address:
            cmd.extend(["--ray_address", self.ray_address])

        cmd.extend([
            "--auto_detect_cluster",
            "--placement_strategy", "spread",
        ])

        # Resource allocation based on experiment scale
        cpus_per_actor = max(0.5, self.config.total_cpus / (exp["scale"] * 1.5))
        gpus_per_actor = self.config.total_gpus / exp["scale"] if exp["scale"] <= self.config.total_gpus else 0

        cmd.extend([
            "--cpus_per_actor", str(cpus_per_actor),
        ])

        if gpus_per_actor > 0:
            cmd.extend(["--gpus_per_actor", str(gpus_per_actor)])

        return cmd

    def generate_experiment_id(self, exp: Dict[str, Any]) -> str:
        """Generate a unique experiment ID."""
        components = [
            exp["set"],
            exp["dataset"],
            exp["paradigm"],
            exp["topology"],
            exp["heterogeneity"],
            exp["privacy"],
            str(exp["scale"])
        ]
        return "_".join(components)

    def parse_experiment_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse experiment output to extract metrics."""
        metrics = {}

        try:
            lines = output.strip().split('\n')

            for line in lines:
                # Look for key metrics in the output
                if "Initial accuracy:" in line or "Initial Test Accuracy:" in line:
                    try:
                        # Extract percentage value
                        parts = line.split(':')[-1].strip().replace('%', '')
                        metrics["initial_accuracy"] = float(parts) / 100.0
                    except (ValueError, IndexError):
                        pass

                elif "Final accuracy:" in line or "Final Test Accuracy:" in line:
                    try:
                        parts = line.split(':')[-1].strip().replace('%', '')
                        metrics["final_accuracy"] = float(parts) / 100.0
                    except (ValueError, IndexError):
                        pass

                elif "Accuracy improvement:" in line or "Accuracy Improvement:" in line:
                    try:
                        parts = line.split(':')[-1].strip().replace('%', '')
                        metrics["accuracy_improvement"] = float(parts) / 100.0
                    except (ValueError, IndexError):
                        pass

                elif "Privacy Spent:" in line:
                    try:
                        # Extract epsilon and delta values
                        parts = line.split("Privacy Spent:")[-1].strip()
                        if "ε=" in parts and "δ=" in parts:
                            eps_part = parts.split("ε=")[1].split(",")[0].strip()
                            delta_part = parts.split("δ=")[1].split(",")[0].strip()
                            metrics["privacy_epsilon_spent"] = float(eps_part.split("/")[0])
                            metrics["privacy_delta_spent"] = float(delta_part.split("/")[0])
                    except (ValueError, IndexError):
                        pass

                elif "Training completed and resources cleaned up" in line:
                    metrics["completed_successfully"] = True

            # Validate that we have essential metrics
            if "final_accuracy" in metrics and "initial_accuracy" in metrics:
                return metrics
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error parsing experiment output: {e}")
            return None

    def save_partial_results(self):
        """Save partial results during execution."""
        if self.results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            partial_file = self.output_dir / f"partial_results_{timestamp}.csv"
            df = pd.DataFrame(self.results)
            df.to_csv(partial_file, index=False)
            self.logger.info(f"Saved partial results to {partial_file}")

    def save_results(self):
        """Save final results to CSV and JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save successful results
        if self.results:
            # CSV format
            df = pd.DataFrame(self.results)
            csv_file = self.output_dir / f"experiment_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)

            # Excel format with multiple sheets
            excel_file = self.output_dir / f"experiment_results_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All_Results', index=False)

                # Create summary sheets
                if len(df) > 0:
                    # Summary by paradigm
                    paradigm_summary = df.groupby(['paradigm', 'dataset']).agg({
                        'final_accuracy': ['mean', 'std', 'count'],
                        'accuracy_improvement': ['mean', 'std'],
                        'execution_time': ['mean', 'std']
                    }).round(4)
                    paradigm_summary.to_excel(writer, sheet_name='Paradigm_Summary')

                    # Summary by topology
                    topology_summary = df.groupby(['topology', 'dataset']).agg({
                        'final_accuracy': ['mean', 'std', 'count'],
                        'accuracy_improvement': ['mean', 'std'],
                        'execution_time': ['mean', 'std']
                    }).round(4)
                    topology_summary.to_excel(writer, sheet_name='Topology_Summary')

                    # Summary by privacy
                    privacy_summary = df.groupby(['privacy', 'dataset']).agg({
                        'final_accuracy': ['mean', 'std', 'count'],
                        'accuracy_improvement': ['mean', 'std'],
                        'execution_time': ['mean', 'std']
                    }).round(4)
                    privacy_summary.to_excel(writer, sheet_name='Privacy_Summary')

            self.logger.info(f"Saved results to {csv_file} and {excel_file}")

        # Save failed experiments
        if self.failed_experiments:
            failed_file = self.output_dir / f"failed_experiments_{timestamp}.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_experiments, f, indent=2)
            self.logger.info(f"Saved failed experiments to {failed_file}")

        # Save summary statistics
        summary = {
            "total_experiments": self.total_experiments,
            "successful_experiments": len(self.results),
            "failed_experiments": len(self.failed_experiments),
            "success_rate": len(self.results) / self.total_experiments if self.total_experiments > 0 else 0,
            "total_execution_time": time.time() - self.start_time if self.start_time else 0,
            "timestamp": timestamp
        }

        summary_file = self.output_dir / f"experiment_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Experiment summary: {summary}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Automated Murmura Experimentation")
    parser.add_argument("--output_dir", default="./experiment_results",
                        help="Output directory for results")
    parser.add_argument("--max_parallel", type=int, default=2,
                        help="Maximum parallel experiments")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate experiment plan without execution")
    parser.add_argument("--priority", type=int, choices=[1, 2, 3], default=None,
                        help="Run only experiments of specific priority")

    args = parser.parse_args()

    # Initialize configuration and runner
    config = ExperimentConfig()
    runner = ExperimentRunner(config, args.output_dir)

    # Generate experiment plan
    runner.logger.info("Generating experiment plan...")
    experiments = runner.generate_experiment_sets()

    # Filter by priority if specified
    if args.priority:
        experiments = [exp for exp in experiments if exp.get("priority") == args.priority]
        runner.logger.info(f"Filtered to {len(experiments)} experiments with priority {args.priority}")

    if args.dry_run:
        # Save experiment plan and exit
        plan_file = Path(args.output_dir) / "experiment_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(experiments, f, indent=2)
        runner.logger.info(f"Experiment plan saved to {plan_file}")
        return

    # Run experiments
    try:
        runner.run_experiments(experiments, args.max_parallel)
    finally:
        runner.save_results()


if __name__ == "__main__":
    main()
