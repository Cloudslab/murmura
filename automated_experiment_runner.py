#!/usr/bin/env python3
"""
Automated Murmura Experiment Runner for Three-Way Paradigm Comparison
Optimized for 5-node AWS G5.2XLARGE cluster (5 GPUs, 40 vCPUs, 160GB RAM total)
"""

import argparse
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    paradigm: str  # centralized, federated, decentralized
    dataset: str  # mnist, ham10000
    topology: str  # star, ring, complete, random
    heterogeneity: str  # iid, moderate_noniid, extreme_noniid
    privacy: str  # none, moderate_dp, strong_dp
    scale: int  # 5, 10, 20

    # Derived parameters
    aggregation_strategy: str = ""
    alpha: float = 1.0
    epsilon: float = 1.0
    delta: float = 1e-5

    # Execution parameters - Updated for high-quality results
    rounds: int = 50  # Increased significantly for convergence
    local_epochs: int = 3  # Balanced for communication efficiency
    batch_size: int = 32
    learning_rate: float = 0.001

    # Cluster configuration
    actors_per_node: int = 2
    cpus_per_actor: float = 1.5
    memory_per_actor: int = 3000  # MB

    def __post_init__(self):
        """Set derived parameters based on configuration"""
        # Set aggregation strategy based on paradigm
        if self.paradigm == "centralized":
            self.aggregation_strategy = "fedavg"  # Not really used for centralized
        elif self.paradigm == "federated":
            self.aggregation_strategy = "fedavg"
        elif self.paradigm == "decentralized":
            self.aggregation_strategy = "gossip_avg"

        # Set alpha based on heterogeneity
        if self.heterogeneity == "iid":
            self.alpha = 1000.0  # Effectively IID
        elif self.heterogeneity == "moderate_noniid":
            self.alpha = 0.5
        elif self.heterogeneity == "extreme_noniid":
            self.alpha = 0.1

        # Set privacy parameters
        if self.privacy == "moderate_dp":
            self.epsilon = 1.0
            self.delta = 1e-5
        elif self.privacy == "strong_dp":
            self.epsilon = 0.1
            self.delta = 1e-6

        # Adjust parameters for decentralized learning
        if self.paradigm == "decentralized":
            # Increase epsilon for Local DP (less efficient than Central DP)
            if self.privacy == "moderate_dp":
                self.epsilon = 2.0
            elif self.privacy == "strong_dp":
                self.epsilon = 0.5

        # CRITICAL: Different training paradigms need different configurations
        if self.paradigm == "centralized":
            # Traditional centralized learning: 1 actor, 1 round, many epochs
            self.scale = 1  # Override scale - only 1 actor needed
            self.rounds = 1  # No communication rounds needed

            # Set epochs based on dataset
            if self.dataset == "ham10000":
                self.local_epochs = 100  # Equivalent training to distributed case
                self.learning_rate = 0.0005
            elif self.dataset == "mnist":
                self.local_epochs = 60   # Equivalent training to distributed case
                self.learning_rate = 0.001

            # Privacy adjustments for centralized
            if self.privacy != "none":
                # Central DP is more efficient, can use more epochs
                self.local_epochs = int(self.local_epochs * 1.2)
                self.learning_rate *= 0.9

        else:
            # Federated/Decentralized learning: multiple rounds, fewer local epochs
            # Dataset-specific training adjustments
            if self.dataset == "ham10000":
                # Medical images need more training
                self.rounds = 75  # More rounds for complex medical data
                self.local_epochs = 4  # More local training
                self.learning_rate = 0.0005  # Lower LR for stability
            elif self.dataset == "mnist":
                # MNIST converges faster
                self.rounds = 40  # Sufficient for MNIST
                self.local_epochs = 3
                self.learning_rate = 0.001

            # Privacy-specific adjustments for distributed learning
            if self.privacy != "none":
                # DP requires more rounds due to noise
                self.rounds = int(self.rounds * 1.5)  # 50% more rounds for DP
                self.learning_rate *= 0.8  # Slightly lower LR for stability with noise

            # Scale-specific adjustments
            if self.scale >= 20:
                # More actors can handle more local work
                self.local_epochs += 1

        # Set resource allocation based on scale
        if self.scale <= 1:
            # Centralized learning
            self.actors_per_node = 1
            self.cpus_per_actor = 4.0  # Can use more resources
            self.memory_per_actor = 8000  # More memory for single actor
        elif self.scale <= 10:
            self.actors_per_node = 2
            self.cpus_per_actor = 1.5
            self.memory_per_actor = 3000
        elif self.scale <= 20:
            self.actors_per_node = 4
            self.cpus_per_actor = 1.8
            self.memory_per_actor = 3500
        else:
            self.actors_per_node = 6
            self.cpus_per_actor = 1.2
            self.memory_per_actor = 2500


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config: ExperimentConfig
    success: bool
    start_time: str
    end_time: str
    duration_minutes: float

    # Performance metrics
    initial_accuracy: float = 0.0
    final_accuracy: float = 0.0
    accuracy_improvement: float = 0.0
    convergence_rounds: int = 0

    # Efficiency metrics
    total_communication_mb: float = 0.0
    communication_per_round: float = 0.0
    training_time_minutes: float = 0.0

    # Privacy metrics (if applicable)
    privacy_epsilon_spent: float = 0.0
    privacy_delta_spent: float = 0.0
    privacy_budget_utilization: float = 0.0

    # Error information
    error_message: str = ""
    error_type: str = ""


class ExperimentRunner:
    """Main experiment runner class"""

    def __init__(self, base_dir: str, ray_address: str = None, max_parallel: int = 2):
        self.base_dir = Path(base_dir)
        self.ray_address = ray_address
        self.max_parallel = max_parallel
        self.results_dir = self.base_dir / "experiment_results"
        self.logs_dir = self.base_dir / "experiment_logs"

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Track experiment state
        self.completed_experiments = set()
        self.failed_experiments = set()

    def setup_logging(self):
        """Setup centralized logging"""
        log_file = self.logs_dir / f"experiment_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)

    def generate_experiment_configs(self, experiment_set: str) -> List[ExperimentConfig]:
        """Generate all experiment configurations for a given set"""
        configs = []

        if experiment_set == "core":
            # Set A: Core Three-Way Comparison (Priority 1)
            # Reduced from full factorial for practical execution
            paradigms = ["centralized", "federated", "decentralized"]
            datasets = ["mnist", "ham10000"]

            # Strategic subset to maximize insights while keeping execution time reasonable
            core_combinations = [
                # Key topology comparisons (only relevant for distributed paradigms)
                ("star", "iid", "none", 10),
                ("ring", "iid", "none", 10),
                ("complete", "iid", "none", 10),

                # Heterogeneity analysis (only relevant for distributed paradigms)
                ("star", "moderate_noniid", "none", 10),
                ("star", "extreme_noniid", "none", 10),
                ("ring", "moderate_noniid", "none", 10),

                # Privacy analysis
                ("star", "moderate_noniid", "moderate_dp", 10),
                ("star", "moderate_noniid", "strong_dp", 10),
                ("ring", "moderate_noniid", "moderate_dp", 10),

                # Scale analysis (only relevant for distributed paradigms)
                ("star", "moderate_noniid", "none", 5),
                ("star", "moderate_noniid", "none", 20),
                ("ring", "moderate_noniid", "none", 20),

                # Centralized baselines (topology and scale don't matter, but we test key scenarios)
                ("star", "iid", "none", 1),  # Basic centralized
                ("star", "iid", "moderate_dp", 1),  # Centralized with DP
                ("star", "iid", "strong_dp", 1),  # Centralized with strong DP
            ]

            for paradigm in paradigms:
                for dataset in datasets:
                    for topology, heterogeneity, privacy, scale in core_combinations:
                        # Skip invalid combinations
                        if paradigm == "centralized":
                            # For centralized, only include the centralized-specific combinations
                            if not (topology == "star" and scale == 1):
                                continue
                            # Heterogeneity doesn't matter for centralized (no partitioning)
                            if heterogeneity != "iid":
                                continue
                        elif paradigm == "federated":
                            # Skip centralized-only combinations
                            if scale == 1:
                                continue
                            # Federated can use star topology
                        elif paradigm == "decentralized":
                            # Skip centralized-only combinations
                            if scale == 1:
                                continue
                            # Decentralized cannot use star topology
                            if topology == "star":
                                continue

                        configs.append(ExperimentConfig(
                            paradigm=paradigm,
                            dataset=dataset,
                            topology=topology,
                            heterogeneity=heterogeneity,
                            privacy=privacy,
                            scale=scale
                        ))

        elif experiment_set == "topology_privacy":
            # Set B: Topology-Privacy Interaction Analysis (Priority 2)
            paradigms = ["federated", "decentralized"]  # Focus on distributed paradigms
            datasets = ["mnist", "ham10000"]
            topologies = ["star", "ring", "complete"]
            privacy_levels = ["none", "moderate_dp", "strong_dp"]

            for paradigm in paradigms:
                for dataset in datasets:
                    for topology in topologies:
                        for privacy in privacy_levels:
                            # Skip invalid combinations
                            if paradigm == "decentralized" and topology == "star":
                                continue

                            configs.append(ExperimentConfig(
                                paradigm=paradigm,
                                dataset=dataset,
                                topology=topology,
                                heterogeneity="moderate_noniid",  # Fixed
                                privacy=privacy,
                                scale=10  # Fixed
                            ))

        elif experiment_set == "scalability":
            # Set C: Scalability Analysis (Priority 3)
            paradigms = ["centralized", "federated", "decentralized"]
            datasets = ["mnist", "ham10000"]
            scales = [5, 10, 15, 20]

            for paradigm in paradigms:
                for dataset in datasets:
                    for scale in scales:
                        topology = "ring" if paradigm == "decentralized" else "star"

                        configs.append(ExperimentConfig(
                            paradigm=paradigm,
                            dataset=dataset,
                            topology=topology,
                            heterogeneity="moderate_noniid",  # Fixed
                            privacy="none",  # Fixed
                            scale=scale
                        ))

        self.logger.info(f"Generated {len(configs)} configurations for {experiment_set} experiment set")
        return configs

    def get_example_script_path(self, config: ExperimentConfig) -> str:
        """Get the appropriate example script path"""
        base_path = "murmura/examples"

        # Map dataset names to script names
        dataset_script_map = {
            "mnist": "mnist",
            "ham10000": "skin_lesion"  # HAM10000 uses skin_lesion scripts
        }

        script_dataset = dataset_script_map.get(config.dataset, config.dataset)

        if config.paradigm == "centralized":
            # Use regular federated examples for centralized (they support star topology)
            if config.privacy != "none":
                script = f"dp_{script_dataset}_example.py"
            else:
                script = f"{script_dataset}_example.py"
        elif config.paradigm == "federated":
            if config.privacy != "none":
                script = f"dp_{script_dataset}_example.py"
            else:
                script = f"{script_dataset}_example.py"
        elif config.paradigm == "decentralized":
            if config.privacy != "none":
                script = f"dp_decentralized_{script_dataset}_example.py"
            else:
                script = f"decentralized_{script_dataset}_example.py"

        return os.path.join(base_path, script)

    def build_command(self, config: ExperimentConfig, experiment_id: str) -> List[str]:
        """Build the command to run an experiment"""
        script_path = self.get_example_script_path(config)

        # Base command
        cmd = [
            sys.executable, script_path,
            "--num_actors", str(config.scale),
            "--partition_strategy", "dirichlet" if config.paradigm != "centralized" else "iid",
            "--alpha", str(config.alpha),
            "--min_partition_size", "50",
            "--rounds", str(config.rounds),
            "--epochs", str(config.local_epochs),
            "--batch_size", str(config.batch_size),
            "--lr", str(config.learning_rate),
            "--log_level", "INFO",
            "--actors_per_node", str(config.actors_per_node),
            "--cpus_per_actor", str(config.cpus_per_actor),
            "--memory_per_actor", str(config.memory_per_actor),
            "--placement_strategy", "spread",
        ]

        # Add topology and aggregation only for distributed learning
        if config.paradigm != "centralized":
            cmd.extend([
                "--topology", config.topology,
                "--aggregation_strategy", config.aggregation_strategy,
            ])
        else:
            # Centralized learning always uses star topology (though it doesn't matter much)
            cmd.extend([
                "--topology", "star",
                "--aggregation_strategy", "fedavg",
            ])

        # Add Ray cluster configuration
        if self.ray_address:
            cmd.extend(["--ray_address", self.ray_address])

        cmd.extend([
            "--ray_namespace", f"murmura_exp_{experiment_id}",
            "--auto_detect_cluster"
        ])

        # Privacy parameters
        if config.privacy != "none":
            cmd.extend([
                "--epsilon", str(config.epsilon),
                "--delta", str(config.delta),
                "--clipping_norm", "1.0",
                "--client_sampling_rate", "0.8" if config.paradigm != "decentralized" else "1.0"
            ])

        # Dataset-specific parameters
        if config.dataset == "ham10000":
            cmd.extend([
                "--dataset_name", "marmal88/skin_cancer",
                "--image_size", "128",
                "--widen_factor", "8",
                "--depth", "16",
                "--dropout", "0.3",
                "--weight_decay", "1e-4"
            ])

        # Paradigm-specific parameters
        if config.paradigm == "decentralized":
            cmd.extend(["--mixing_parameter", "0.5"])

        # Save path
        save_path = self.results_dir / f"{experiment_id}_model.pt"
        cmd.extend(["--save_path", str(save_path)])

        return cmd

    def parse_experiment_output(self, output: str, config: ExperimentConfig) -> Dict[str, Any]:
        """Parse experiment output to extract metrics"""
        metrics = {}

        lines = output.split('\n')
        for line in lines:
            line = line.strip()

            # Extract accuracy metrics
            if "Initial accuracy:" in line:
                try:
                    metrics['initial_accuracy'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif "Final accuracy:" in line:
                try:
                    metrics['final_accuracy'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif "Accuracy improvement:" in line:
                try:
                    metrics['accuracy_improvement'] = float(line.split(':')[1].strip())
                except:
                    pass

            # Extract privacy metrics (if applicable)
            elif "Privacy Spent:" in line and config.privacy != "none":
                try:
                    # Parse "ε=X.XXXX/Y.YYYY, δ=Z.ZZe-XX/W.WWe-XX"
                    parts = line.split(':')[1].strip()
                    eps_part = parts.split(',')[0].strip()
                    delta_part = parts.split(',')[1].strip()

                    eps_spent = float(eps_part.split('=')[1].split('/')[0])
                    delta_spent = float(delta_part.split('=')[1].split('/')[0])

                    metrics['privacy_epsilon_spent'] = eps_spent
                    metrics['privacy_delta_spent'] = delta_spent
                except:
                    pass

            elif "Privacy Budget Utilization:" in line and config.privacy != "none":
                try:
                    # Parse "ε=XX.X%, δ=XX.X%"
                    parts = line.split(':')[1].strip()
                    eps_util = float(parts.split(',')[0].split('=')[1].replace('%', ''))
                    metrics['privacy_budget_utilization'] = eps_util
                except:
                    pass

        return metrics

    def run_single_experiment(self, config: ExperimentConfig, experiment_id: str) -> ExperimentResult:
        """Run a single experiment"""
        start_time = datetime.now()

        self.logger.info(f"Starting experiment {experiment_id}: {config.paradigm}-{config.dataset}-{config.topology}")

        # Build command
        cmd = self.build_command(config, experiment_id)

        # Setup logging for this experiment
        log_file = self.logs_dir / f"{experiment_id}.log"

        try:
            # Run the experiment
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                    cwd=self.base_dir
                )

                output = process.stdout
                f.write(output)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60.0

            if process.returncode == 0:
                # Success - parse metrics
                metrics = self.parse_experiment_output(output, config)

                result = ExperimentResult(
                    config=config,
                    success=True,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_minutes=duration,
                    **metrics
                )

                self.logger.info(f"Experiment {experiment_id} completed successfully in {duration:.1f} minutes")

            else:
                # Failure
                error_msg = f"Process exited with code {process.returncode}"
                result = ExperimentResult(
                    config=config,
                    success=False,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_minutes=duration,
                    error_message=error_msg,
                    error_type="subprocess_error"
                )

                self.logger.error(f"Experiment {experiment_id} failed: {error_msg}")

        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60.0

            result = ExperimentResult(
                config=config,
                success=False,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_minutes=duration,
                error_message="Experiment timed out after 1 hour",
                error_type="timeout"
            )

            self.logger.error(f"Experiment {experiment_id} timed out after 1 hour")

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60.0

            result = ExperimentResult(
                config=config,
                success=False,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_minutes=duration,
                error_message=str(e),
                error_type=type(e).__name__
            )

            self.logger.error(f"Experiment {experiment_id} failed with exception: {e}")
            self.logger.error(traceback.format_exc())

        return result

    def save_results(self, results: List[ExperimentResult], output_file: str):
        """Save results to CSV/Excel file"""
        # Convert results to flat dictionaries
        data = []
        for result in results:
            row = {
                # Experiment configuration
                'experiment_id': f"{result.config.paradigm}_{result.config.dataset}_{result.config.topology}_{result.config.heterogeneity}_{result.config.privacy}_{result.config.scale}",
                'paradigm': result.config.paradigm,
                'dataset': result.config.dataset,
                'topology': result.config.topology,
                'heterogeneity': result.config.heterogeneity,
                'privacy': result.config.privacy,
                'scale': result.config.scale,
                'aggregation_strategy': result.config.aggregation_strategy,
                'alpha': result.config.alpha,
                'epsilon': result.config.epsilon,
                'delta': result.config.delta,

                # Execution metadata
                'success': result.success,
                'start_time': result.start_time,
                'end_time': result.end_time,
                'duration_minutes': result.duration_minutes,

                # Performance metrics
                'initial_accuracy': result.initial_accuracy,
                'final_accuracy': result.final_accuracy,
                'accuracy_improvement': result.accuracy_improvement,
                'convergence_rounds': result.convergence_rounds,

                # Efficiency metrics
                'total_communication_mb': result.total_communication_mb,
                'communication_per_round': result.communication_per_round,
                'training_time_minutes': result.training_time_minutes,

                # Privacy metrics
                'privacy_epsilon_spent': result.privacy_epsilon_spent,
                'privacy_delta_spent': result.privacy_delta_spent,
                'privacy_budget_utilization': result.privacy_budget_utilization,

                # Error information
                'error_message': result.error_message,
                'error_type': result.error_type,
            }
            data.append(row)

        # Save to CSV
        df = pd.DataFrame(data)

        if output_file.endswith('.xlsx'):
            df.to_excel(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)

        self.logger.info(f"Results saved to {output_file}")

        # Print summary
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results if r.success)
        failed_experiments = total_experiments - successful_experiments

        self.logger.info(f"Experiment Summary:")
        self.logger.info(f"  Total experiments: {total_experiments}")
        self.logger.info(f"  Successful: {successful_experiments}")
        self.logger.info(f"  Failed: {failed_experiments}")
        self.logger.info(f"  Success rate: {successful_experiments/total_experiments*100:.1f}%")

        if failed_experiments > 0:
            self.logger.info("Failed experiments:")
            for result in results:
                if not result.success:
                    exp_id = f"{result.config.paradigm}_{result.config.dataset}_{result.config.topology}_{result.config.privacy}_{result.config.scale}"
                    self.logger.info(f"  {exp_id}: {result.error_type} - {result.error_message}")

    def run_experiments(self, experiment_set: str, output_file: str, resume: bool = False):
        """Run all experiments in a set"""
        # Generate configurations
        configs = self.generate_experiment_configs(experiment_set)

        # Load existing results if resuming
        completed_results = []
        if resume and os.path.exists(output_file):
            try:
                if output_file.endswith('.xlsx'):
                    existing_df = pd.read_excel(output_file)
                else:
                    existing_df = pd.read_csv(output_file)

                # Convert back to results for completed experiments
                # This is simplified - you might want to implement full conversion
                completed_experiment_ids = set(existing_df['experiment_id'].tolist())
                configs = [c for c in configs if self.get_experiment_id(c) not in completed_experiment_ids]

                self.logger.info(f"Resuming: {len(completed_experiment_ids)} experiments already completed")

            except Exception as e:
                self.logger.warning(f"Could not load existing results: {e}")

        self.logger.info(f"Running {len(configs)} experiments with max {self.max_parallel} parallel")

        # Group configs by execution priority (fail fast on critical experiments)
        priority_configs = []
        regular_configs = []

        for config in configs:
            # Prioritize simple, fast experiments first to validate setup
            if (config.dataset == "mnist" and
                    config.scale <= 10 and
                    config.privacy == "none" and
                    config.heterogeneity == "iid"):
                priority_configs.append(config)
            else:
                regular_configs.append(config)

        all_configs = priority_configs + regular_configs
        results = []

        # Run experiments
        for i, config in enumerate(all_configs):
            experiment_id = self.get_experiment_id(config)

            self.logger.info(f"Progress: {i+1}/{len(all_configs)} - Running {experiment_id}")

            result = self.run_single_experiment(config, experiment_id)
            results.append(result)

            # Fail fast: if first 3 experiments fail, stop
            if i < 3 and not result.success:
                self.logger.error(f"Early failure detected. Stopping experiment suite.")
                self.logger.error(f"Fix the issue with {experiment_id} before continuing.")
                break

            # Save intermediate results every 5 experiments
            if (i + 1) % 5 == 0:
                temp_file = output_file.replace('.', f'_temp_{i+1}.')
                self.save_results(results, temp_file)

        # Save final results
        all_results = completed_results + results
        self.save_results(all_results, output_file)

        return all_results

    def get_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID"""
        return f"{config.paradigm}_{config.dataset}_{config.topology}_{config.heterogeneity}_{config.privacy}_{config.scale}"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Automated Murmura Experiment Runner")

    parser.add_argument("--experiment_set",
                        choices=["core", "topology_privacy", "scalability", "all"],
                        default="core",
                        help="Which experiment set to run")

    parser.add_argument("--base_dir",
                        type=str,
                        default="/home/ubuntu/murmura",
                        help="Base directory containing Murmura framework")

    parser.add_argument("--output_file",
                        type=str,
                        default="experiment_results.csv",
                        help="Output file for results (CSV or Excel)")

    parser.add_argument("--ray_address",
                        type=str,
                        default=None,
                        help="Ray cluster head node address")

    parser.add_argument("--max_parallel",
                        type=int,
                        default=1,
                        help="Maximum parallel experiments (be conservative with cluster resources)")

    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume from existing results file")

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(
        base_dir=args.base_dir,
        ray_address=args.ray_address,
        max_parallel=args.max_parallel
    )

    # Determine which experiment sets to run
    if args.experiment_set == "all":
        experiment_sets = ["core", "topology_privacy", "scalability"]
    else:
        experiment_sets = [args.experiment_set]

    # Run experiments
    all_results = []
    for exp_set in experiment_sets:
        output_file = args.output_file.replace('.', f'_{exp_set}.')
        runner.logger.info(f"Starting {exp_set} experiment set")

        results = runner.run_experiments(
            experiment_set=exp_set,
            output_file=output_file,
            resume=args.resume
        )
        all_results.extend(results)

    # Save combined results if running multiple sets
    if len(experiment_sets) > 1:
        runner.save_results(all_results, args.output_file)

    runner.logger.info("Experiment suite completed!")


if __name__ == "__main__":
    main()
