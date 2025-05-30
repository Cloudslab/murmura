#!/usr/bin/env python3
"""
Automated Experiment Runner for Murmura Framework
Executes a comprehensive suite of experiments comparing traditional, federated, and decentralized learning.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class ExperimentRunner:
    """Manages and executes federated learning experiments."""

    def __init__(self, base_dir: str = "./experiments", ray_address: str = None):
        self.base_dir = Path(base_dir)
        self.ray_address = ray_address
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.base_dir / f"results_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Results storage
        self.all_results = []

    def setup_logging(self):
        """Configure logging for the experiment runner."""
        log_file = self.results_dir / "experiment_runner.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("ExperimentRunner")

    def build_command(self, script: str, args: Dict[str, any]) -> List[str]:
        """Build command line arguments for running an experiment."""
        cmd = ["python", script]

        for key, value in args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        return cmd

    def run_experiment(self, name: str, script: str, args: Dict[str, any],
                       experiment_group: str = "") -> Dict:
        """Execute a single experiment and collect results."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Running experiment: {name}")
        self.logger.info(f"Script: {script}")
        self.logger.info(f"Arguments: {json.dumps(args, indent=2)}")

        # Create experiment directory
        exp_dir = self.results_dir / experiment_group / name.replace(" ", "_")
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Add common arguments
        args["vis_dir"] = str(exp_dir / "visualizations")
        args["save_path"] = str(exp_dir / "model.pt")
        args["log_level"] = "INFO"
        args["create_summary"] = True
        args["placement_strategy"] = "spread"

        if self.ray_address:
            args["ray_address"] = self.ray_address

        # Build and run command
        cmd = self.build_command(script, args)
        self.logger.info(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            # Run the experiment
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            execution_time = time.time() - start_time

            # Parse results from output
            metrics = self.parse_results(result.stdout)

            # Store results
            experiment_result = {
                "name": name,
                "group": experiment_group,
                "script": script,
                "args": args,
                "execution_time": execution_time,
                "metrics": metrics,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }

            # Save individual experiment results
            with open(exp_dir / "results.json", "w") as f:
                json.dump(experiment_result, f, indent=2)

            self.logger.info(f"Experiment completed successfully in {execution_time:.2f}s")
            self.logger.info(f"Results: {metrics}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Experiment failed: {e}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")

            experiment_result = {
                "name": name,
                "group": experiment_group,
                "script": script,
                "args": args,
                "execution_time": time.time() - start_time,
                "status": "failed",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "timestamp": datetime.now().isoformat()
            }

        self.all_results.append(experiment_result)
        return experiment_result

    def parse_results(self, output: str) -> Dict:
        """Parse experiment results from stdout."""
        metrics = {}

        # Parse key metrics from output
        for line in output.split('\n'):
            if "Initial accuracy:" in line:
                try:
                    metrics["initial_accuracy"] = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "Final accuracy:" in line:
                try:
                    metrics["final_accuracy"] = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "Accuracy improvement:" in line:
                try:
                    metrics["accuracy_improvement"] = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "Training completed with" in line and "rounds" in line:
                try:
                    parts = line.split()
                    rounds_idx = parts.index("rounds") - 1
                    metrics["total_rounds"] = int(parts[rounds_idx])
                except:
                    pass

        return metrics

    def run_experiment_1(self):
        """Experiment 1: Baseline Comparison (No DP, IID data)"""
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 1: Baseline Comparison (No DP, IID data)")
        self.logger.info("="*80)

        experiments = []

        # Traditional (single node) - simulated with 1 actor
        experiments.append({
            "name": "Traditional_MNIST",
            "script": "murmura/examples/mnist_example.py",
            "args": {
                "num_actors": 1,
                "partition_strategy": "iid",
                "topology": "star",
                "rounds": 50,
                "epochs": 1,
                "aggregation_strategy": "fedavg"
            }
        })

        experiments.append({
            "name": "Traditional_SkinLesion",
            "script": "murmura/examples/skin_lesion_example.py",
            "args": {
                "num_actors": 1,
                "partition_strategy": "iid",
                "topology": "star",
                "rounds": 50,
                "epochs": 1,
                "aggregation_strategy": "fedavg"
            }
        })

        # Federated (star topology)
        experiments.append({
            "name": "Federated_MNIST_Star",
            "script": "murmura/examples/mnist_example.py",
            "args": {
                "num_actors": 10,
                "partition_strategy": "iid",
                "topology": "star",
                "rounds": 10,
                "epochs": 5,
                "aggregation_strategy": "fedavg"
            }
        })

        experiments.append({
            "name": "Federated_SkinLesion_Star",
            "script": "murmura/examples/skin_lesion_example.py",
            "args": {
                "num_actors": 10,
                "partition_strategy": "iid",
                "topology": "star",
                "rounds": 10,
                "epochs": 5,
                "aggregation_strategy": "fedavg"
            }
        })

        # Decentralized (all topologies)
        for topology in ["complete", "ring", "line"]:
            experiments.append({
                "name": f"Decentralized_MNIST_{topology}",
                "script": "murmura/examples/decentralized_mnist_example.py",
                "args": {
                    "num_actors": 10,
                    "partition_strategy": "iid",
                    "topology": topology,
                    "rounds": 10,
                    "epochs": 5,
                    "aggregation_strategy": "gossip_avg"
                }
            })

            experiments.append({
                "name": f"Decentralized_SkinLesion_{topology}",
                "script": "murmura/examples/decentralized_skin_lesion_example.py",
                "args": {
                    "num_actors": 10,
                    "partition_strategy": "iid",
                    "topology": topology,
                    "rounds": 10,
                    "epochs": 5,
                    "aggregation_strategy": "gossip_avg"
                }
            })

        # Run all experiments
        for exp in experiments:
            self.run_experiment(exp["name"], exp["script"], exp["args"], "experiment_1")

    def run_experiment_2(self):
        """Experiment 2: Topology Impact (No DP, IID data)"""
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 2: Topology Impact (No DP, IID data)")
        self.logger.info("="*80)

        experiments = []

        # Federated: Star vs Complete
        for topology in ["star", "complete"]:
            experiments.append({
                "name": f"Federated_MNIST_{topology}_20nodes",
                "script": "murmura/examples/mnist_example.py",
                "args": {
                    "num_actors": 20,
                    "partition_strategy": "iid",
                    "topology": topology,
                    "rounds": 20,
                    "epochs": 5,
                    "aggregation_strategy": "fedavg"
                }
            })

        # Decentralized: All topologies
        for topology in ["complete", "ring", "line"]:
            experiments.append({
                "name": f"Decentralized_MNIST_{topology}_20nodes",
                "script": "murmura/examples/decentralized_mnist_example.py",
                "args": {
                    "num_actors": 20,
                    "partition_strategy": "iid",
                    "topology": topology,
                    "rounds": 20,
                    "epochs": 5,
                    "aggregation_strategy": "gossip_avg"
                }
            })

        # Run all experiments
        for exp in experiments:
            self.run_experiment(exp["name"], exp["script"], exp["args"], "experiment_2")

    def run_experiment_3(self):
        """Experiment 3: Data Heterogeneity (α sweep)"""
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 3: Data Heterogeneity (α sweep)")
        self.logger.info("="*80)

        experiments = []
        alpha_values = [0.1, 0.5, 1.0, 10.0]

        for alpha in alpha_values:
            # Federated
            experiments.append({
                "name": f"Federated_MNIST_alpha_{alpha}",
                "script": "murmura/examples/mnist_example.py",
                "args": {
                    "num_actors": 20,
                    "partition_strategy": "dirichlet",
                    "alpha": alpha,
                    "topology": "complete",
                    "rounds": 20,
                    "epochs": 5,
                    "aggregation_strategy": "fedavg"
                }
            })

            # Decentralized
            experiments.append({
                "name": f"Decentralized_MNIST_alpha_{alpha}",
                "script": "murmura/examples/decentralized_mnist_example.py",
                "args": {
                    "num_actors": 20,
                    "partition_strategy": "dirichlet",
                    "alpha": alpha,
                    "topology": "complete",
                    "rounds": 20,
                    "epochs": 5,
                    "aggregation_strategy": "gossip_avg"
                }
            })

        # Run all experiments
        for exp in experiments:
            self.run_experiment(exp["name"], exp["script"], exp["args"], "experiment_3")

    def run_experiment_4(self):
        """Experiment 4: Privacy-Utility Trade-offs"""
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 4: Privacy-Utility Trade-offs")
        self.logger.info("="*80)

        experiments = []
        epsilon_values = [0.1, 1.0, 10.0]

        for epsilon in epsilon_values:
            # Federated with DP
            experiments.append({
                "name": f"Federated_DP_MNIST_epsilon_{epsilon}",
                "script": "murmura/examples/dp_mnist_example.py",
                "args": {
                    "num_actors": 20,
                    "partition_strategy": "iid",
                    "topology": "complete",
                    "rounds": 20,
                    "epochs": 5,
                    "aggregation_strategy": "fedavg",
                    "epsilon": epsilon,
                    "delta": 1e-5,
                    "clipping_norm": 1.0
                }
            })

            # Decentralized with DP
            experiments.append({
                "name": f"Decentralized_DP_MNIST_epsilon_{epsilon}",
                "script": "murmura/examples/dp_decentralized_mnist_example.py",
                "args": {
                    "num_actors": 20,
                    "partition_strategy": "iid",
                    "topology": "complete",
                    "rounds": 20,
                    "epochs": 5,
                    "aggregation_strategy": "gossip_avg",
                    "epsilon": epsilon,
                    "delta": 1e-5,
                    "clipping_norm": 1.0
                }
            })

        # Run all experiments
        for exp in experiments:
            self.run_experiment(exp["name"], exp["script"], exp["args"], "experiment_4")

    def run_experiment_5(self):
        """Experiment 5: Combined Effects - 2×2×3 design"""
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 5: Combined Effects")
        self.logger.info("="*80)

        experiments = []

        # 2x2x3: (IID/Non-IID) × (No DP/DP) × (Traditional/Federated/Decentralized)
        data_configs = [
            {"partition_strategy": "iid", "alpha": None, "label": "IID"},
            {"partition_strategy": "dirichlet", "alpha": 0.5, "label": "NonIID"}
        ]

        privacy_configs = [
            {"dp": False, "label": "NoDP"},
            {"dp": True, "epsilon": 1.0, "delta": 1e-5, "label": "DP"}
        ]

        for data_cfg in data_configs:
            for privacy_cfg in privacy_configs:
                # Traditional (simulated)
                if not privacy_cfg["dp"]:
                    experiments.append({
                        "name": f"Traditional_MNIST_{data_cfg['label']}_{privacy_cfg['label']}",
                        "script": "murmura/examples/mnist_example.py",
                        "args": {
                            "num_actors": 1,
                            "partition_strategy": data_cfg["partition_strategy"],
                            "alpha": data_cfg["alpha"],
                            "topology": "star",
                            "rounds": 50,
                            "epochs": 1,
                            "aggregation_strategy": "fedavg"
                        }
                    })

                # Federated
                script = "murmura/examples/dp_mnist_example.py" if privacy_cfg["dp"] else "murmura/examples/mnist_example.py"
                args = {
                    "num_actors": 10,
                    "partition_strategy": data_cfg["partition_strategy"],
                    "alpha": data_cfg["alpha"],
                    "topology": "star",
                    "rounds": 10,
                    "epochs": 5,
                    "aggregation_strategy": "fedavg"
                }
                if privacy_cfg["dp"]:
                    args.update({
                        "epsilon": privacy_cfg["epsilon"],
                        "delta": privacy_cfg["delta"],
                        "clipping_norm": 1.0
                    })

                experiments.append({
                    "name": f"Federated_MNIST_{data_cfg['label']}_{privacy_cfg['label']}",
                    "script": script,
                    "args": args
                })

                # Decentralized
                script = "murmura/examples/dp_decentralized_mnist_example.py" if privacy_cfg["dp"] else "murmura/examples/decentralized_mnist_example.py"
                args = {
                    "num_actors": 10,
                    "partition_strategy": data_cfg["partition_strategy"],
                    "alpha": data_cfg["alpha"],
                    "topology": "ring",
                    "rounds": 10,
                    "epochs": 5,
                    "aggregation_strategy": "gossip_avg"
                }
                if privacy_cfg["dp"]:
                    args.update({
                        "epsilon": privacy_cfg["epsilon"],
                        "delta": privacy_cfg["delta"],
                        "clipping_norm": 1.0
                    })

                experiments.append({
                    "name": f"Decentralized_MNIST_{data_cfg['label']}_{privacy_cfg['label']}",
                    "script": script,
                    "args": args
                })

        # Run all experiments
        for exp in experiments:
            self.run_experiment(exp["name"], exp["script"], exp["args"], "experiment_5")

    def run_experiment_6(self):
        """Experiment 6: Scalability"""
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 6: Scalability")
        self.logger.info("="*80)

        experiments = []
        node_counts = [10, 20, 50]

        for nodes in node_counts:
            # Federated
            experiments.append({
                "name": f"Federated_MNIST_{nodes}nodes",
                "script": "murmura/examples/mnist_example.py",
                "args": {
                    "num_actors": nodes,
                    "partition_strategy": "iid",
                    "topology": "star",
                    "rounds": 10,
                    "epochs": 2,
                    "aggregation_strategy": "fedavg"
                }
            })

            # Decentralized
            experiments.append({
                "name": f"Decentralized_MNIST_{nodes}nodes",
                "script": "murmura/examples/decentralized_mnist_example.py",
                "args": {
                    "num_actors": nodes,
                    "partition_strategy": "iid",
                    "topology": "ring",
                    "rounds": 10,
                    "epochs": 2,
                    "aggregation_strategy": "gossip_avg"
                }
            })

        # Run all experiments
        for exp in experiments:
            self.run_experiment(exp["name"], exp["script"], exp["args"], "experiment_6")

    def run_experiment_7(self):
        """Experiment 7: Dataset Generalization (Medical Dataset)"""
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 7: Dataset Generalization (Medical Dataset)")
        self.logger.info("="*80)

        experiments = []

        # Key experiments on medical dataset
        # Baseline comparison
        experiments.extend([
            {
                "name": "Federated_SkinLesion_Baseline",
                "script": "murmura/examples/skin_lesion_example.py",
                "args": {
                    "num_actors": 10,
                    "partition_strategy": "iid",
                    "topology": "star",
                    "rounds": 10,
                    "epochs": 2,
                    "aggregation_strategy": "fedavg"
                }
            },
            {
                "name": "Decentralized_SkinLesion_Baseline",
                "script": "murmura/examples/decentralized_skin_lesion_example.py",
                "args": {
                    "num_actors": 10,
                    "partition_strategy": "iid",
                    "topology": "ring",
                    "rounds": 10,
                    "epochs": 2,
                    "aggregation_strategy": "gossip_avg"
                }
            }
        ])

        # Heterogeneity
        experiments.extend([
            {
                "name": "Federated_SkinLesion_Heterogeneous",
                "script": "murmura/examples/skin_lesion_example.py",
                "args": {
                    "num_actors": 10,
                    "partition_strategy": "dirichlet",
                    "alpha": 0.3,
                    "topology": "star",
                    "rounds": 10,
                    "epochs": 2,
                    "aggregation_strategy": "fedavg"
                }
            },
            {
                "name": "Decentralized_SkinLesion_Heterogeneous",
                "script": "murmura/examples/decentralized_skin_lesion_example.py",
                "args": {
                    "num_actors": 10,
                    "partition_strategy": "dirichlet",
                    "alpha": 0.3,
                    "topology": "ring",
                    "rounds": 10,
                    "epochs": 2,
                    "aggregation_strategy": "gossip_avg"
                }
            }
        ])

        # Privacy
        experiments.extend([
            {
                "name": "Federated_DP_SkinLesion",
                "script": "murmura/examples/dp_skin_lesion_example.py",
                "args": {
                    "num_actors": 10,
                    "partition_strategy": "iid",
                    "topology": "star",
                    "rounds": 10,
                    "epochs": 2,
                    "aggregation_strategy": "fedavg",
                    "epsilon": 0.5,
                    "delta": 1e-6,
                    "clipping_norm": 1.0
                }
            },
            {
                "name": "Decentralized_DP_SkinLesion",
                "script": "murmura/examples/dp_decentralized_skin_lesion_example.py",
                "args": {
                    "num_actors": 10,
                    "partition_strategy": "iid",
                    "topology": "ring",
                    "rounds": 10,
                    "epochs": 2,
                    "aggregation_strategy": "gossip_avg",
                    "epsilon": 3.0,
                    "delta": 1e-6,
                    "clipping_norm": 1.0
                }
            }
        ])

        # Run all experiments
        for exp in experiments:
            self.run_experiment(exp["name"], exp["script"], exp["args"], "experiment_7")

    def generate_report(self):
        """Generate a comprehensive report of all experiments."""
        self.logger.info("\n" + "="*80)
        self.logger.info("Generating Experiment Report")
        self.logger.info("="*80)

        # Save all results
        results_file = self.results_dir / "all_results.json"
        with open(results_file, "w") as f:
            json.dump(self.all_results, f, indent=2)

        # Create summary DataFrame
        summary_data = []
        for result in self.all_results:
            row = {
                "experiment": result["name"],
                "group": result["group"],
                "status": result["status"],
                "execution_time": result.get("execution_time", 0)
            }

            if "metrics" in result:
                row.update(result["metrics"])

            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Save summary CSV
        csv_file = self.results_dir / "experiment_summary.csv"
        df.to_csv(csv_file, index=False)

        # Generate summary statistics
        self.logger.info("\nExperiment Summary:")
        self.logger.info(f"Total experiments: {len(self.all_results)}")
        self.logger.info(f"Successful: {len([r for r in self.all_results if r['status'] == 'success'])}")
        self.logger.info(f"Failed: {len([r for r in self.all_results if r['status'] == 'failed'])}")
        self.logger.info(f"Total execution time: {df['execution_time'].sum():.2f} seconds")

        # Group by experiment type
        if not df.empty and 'final_accuracy' in df.columns:
            self.logger.info("\nAccuracy Summary by Group:")
            accuracy_summary = df.groupby('group')['final_accuracy'].agg(['mean', 'std', 'min', 'max'])
            self.logger.info(accuracy_summary)

        self.logger.info(f"\nResults saved to: {self.results_dir}")

    def run_all_experiments(self):
        """Run all experiments in sequence."""
        self.run_experiment_1()
        self.run_experiment_2()
        self.run_experiment_3()
        self.run_experiment_4()
        self.run_experiment_5()
        self.run_experiment_6()
        self.run_experiment_7()
        self.generate_report()


def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(description="Automated Experiment Runner for Murmura Framework")

    parser.add_argument(
        "--experiments",
        nargs="+",
        type=int,
        default=None,
        help="List of experiment numbers to run (e.g., 1 2 3). If not specified, runs all."
    )

    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (e.g., ray://head-node:10001)"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="./experiments",
        help="Base directory for experiment results"
    )

    args = parser.parse_args()

    # Create experiment runner
    runner = ExperimentRunner(base_dir=args.base_dir, ray_address=args.ray_address)

    # Run specified experiments or all
    if args.experiments:
        for exp_num in args.experiments:
            method_name = f"run_experiment_{exp_num}"
            if hasattr(runner, method_name):
                getattr(runner, method_name)()
            else:
                runner.logger.error(f"Experiment {exp_num} not found")
        runner.generate_report()
    else:
        runner.run_all_experiments()


if __name__ == "__main__":
    main()
