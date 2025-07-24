#!/usr/bin/env python3
"""
Resource-Aware Client Sampling Experiments for Differentially Private Federated Learning

This script runs comprehensive experiments to evaluate resource-aware client sampling strategies
across different topologies, privacy settings, and client configurations for the research paper:
"Resource-Aware Client Sampling Strategies for Differentially Private FL"

Author: Generated for Murmura Framework
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceAwareSamplingExperiments:
    """Comprehensive experiment suite for resource-aware client sampling research"""
    
    def __init__(self, base_output_dir: str = "resource_aware_experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.base_output_dir / f"experiments_{self.experiment_timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment configuration
        self.datasets = ["mnist", "cifar10"]
        self.topologies = ["star", "ring", "complete"]
        self.learning_paradigms = ["federated", "decentralized"]
        self.client_counts = [5, 10, 20]
        self.sampling_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.privacy_settings = [
            {"name": "no_dp", "enable_dp": False},
            {"name": "high_privacy", "enable_dp": True, "target_epsilon_per_round": 0.1, "target_delta": 1e-5},
            {"name": "medium_privacy", "enable_dp": True, "target_epsilon_per_round": 0.5, "target_delta": 1e-5},
            {"name": "low_privacy", "enable_dp": True, "target_epsilon_per_round": 1.0, "target_delta": 1e-5}
        ]
        
        # Training configuration
        self.rounds = 10
        self.epochs = 3
        self.batch_size = 64
        
        # Results tracking
        self.experiment_results = []
        self.failed_experiments = []
        
        logger.info(f"Experiment suite initialized. Results will be saved to: {self.results_dir}")
    
    def get_topology_compatibility(self, paradigm: str, topology: str) -> bool:
        """Check if topology is compatible with learning paradigm"""
        if paradigm == "decentralized" and topology == "star":
            return False  # Decentralized learning doesn't work with star topology
        return True
    
    def get_experiment_script(self, dataset: str, paradigm: str) -> str:
        """Get the appropriate example script for dataset and paradigm"""
        if paradigm == "federated":
            if dataset == "mnist":
                return "murmura/examples/dp_mnist_example.py"
            elif dataset == "cifar10":
                return "murmura/examples/dp_cifar10_example.py"
        else:  # decentralized
            if dataset == "mnist":
                return "murmura/examples/dp_decentralized_mnist_example.py"
            elif dataset == "cifar10":
                return "murmura/examples/dp_decentralized_cifar10_example.py"
        
        raise ValueError(f"No script found for dataset={dataset}, paradigm={paradigm}")
    
    def build_experiment_command(self, 
                                dataset: str, 
                                paradigm: str, 
                                topology: str,
                                num_clients: int,
                                sampling_rate: float,
                                privacy_config: Dict[str, Any],
                                output_dir: str) -> List[str]:
        """Build command line arguments for experiment"""
        
        script = self.get_experiment_script(dataset, paradigm)
        
        cmd = [
            "python", script,
            "--rounds", str(self.rounds),
            "--epochs", str(self.epochs),
            "--batch_size", str(self.batch_size),
            "--num_actors", str(num_clients),
            "--topology", topology,
            "--client_sampling_rate", str(sampling_rate),
            "--data_sampling_rate", "1.0",  # Focus on client sampling for this study
            "--vis_dir", output_dir,
            "--create_summary",
            "--experiment_name", f"{dataset}_{paradigm}_{topology}_{num_clients}clients_{sampling_rate}rate_{privacy_config['name']}"
        ]
        
        # Add privacy settings
        if privacy_config["enable_dp"]:
            cmd.extend([
                "--enable_dp",
                "--target_epsilon_per_round", str(privacy_config["target_epsilon_per_round"]),
                "--target_delta", str(privacy_config["target_delta"]),
                "--enable_subsampling_amplification"
            ])
        
        return cmd
    
    def run_single_experiment(self, 
                            dataset: str,
                            paradigm: str, 
                            topology: str,
                            num_clients: int,
                            sampling_rate: float,
                            privacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment configuration"""
        
        # Create experiment-specific output directory
        exp_name = f"{dataset}_{paradigm}_{topology}_{num_clients}clients_{sampling_rate}rate_{privacy_config['name']}"
        exp_dir = self.results_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Running experiment: {exp_name}")
        
        # Build command
        cmd = self.build_experiment_command(
            dataset, paradigm, topology, num_clients, 
            sampling_rate, privacy_config, str(exp_dir)
        )
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=Path(__file__).parent.parent  # Run from murmura root
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                status = "success"
                error_msg = None
                logger.info(f"✅ Experiment {exp_name} completed successfully in {duration:.1f}s")
            else:
                status = "failed"
                error_msg = result.stderr
                logger.error(f"❌ Experiment {exp_name} failed: {error_msg}")
            
        except subprocess.TimeoutExpired:
            status = "timeout"
            error_msg = "Experiment timed out after 1 hour"
            duration = 3600
            logger.error(f"⏰ Experiment {exp_name} timed out")
        except Exception as e:
            status = "error"
            error_msg = str(e)
            duration = time.time() - start_time
            logger.error(f"💥 Experiment {exp_name} crashed: {error_msg}")
        
        # Collect experiment metadata
        experiment_data = {
            "experiment_name": exp_name,
            "dataset": dataset,
            "paradigm": paradigm,
            "topology": topology,
            "num_clients": num_clients,
            "sampling_rate": sampling_rate,
            "privacy_config": privacy_config,
            "training_config": {
                "rounds": self.rounds,
                "epochs": self.epochs,
                "batch_size": self.batch_size
            },
            "execution": {
                "status": status,
                "duration_seconds": duration,
                "error_message": error_msg,
                "command": " ".join(cmd)
            },
            "output_directory": str(exp_dir),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save experiment metadata
        with open(exp_dir / "experiment_metadata.json", "w") as f:
            json.dump(experiment_data, f, indent=2)
        
        return experiment_data
    
    def analyze_resource_files(self, exp_dir: Path) -> Dict[str, Any]:
        """Analyze resource tracking CSV files from an experiment"""
        analysis = {}
        
        # Find the actual results directory (may be nested)
        csv_files = list(exp_dir.rglob("training_data_*.csv"))
        if not csv_files:
            return {"error": "No CSV files found"}
        
        csv_dir = csv_files[0].parent
        
        # Analyze key resource files
        resource_files = {
            "sampling": csv_dir / "training_data_sampling.csv",
            "computation_time": csv_dir / "training_data_computation_time.csv", 
            "resource_summary": csv_dir / "training_data_resource_summary.csv",
            "parameter_transfers": csv_dir / "training_data_parameter_transfers.csv"
        }
        
        for file_type, file_path in resource_files.items():
            if file_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    
                    if file_type == "resource_summary":
                        analysis[file_type] = {
                            "total_communication_bytes": df["total_communication_bytes"].sum(),
                            "avg_round_time": df["round_completion_time"].mean(),
                            "avg_efficiency_score": df["resource_efficiency_score"].mean(),
                            "avg_communication_savings": df["communication_savings_percent"].mean()
                        }
                    elif file_type == "computation_time":
                        analysis[file_type] = {
                            "total_training_time": df["time_taken"].sum(),
                            "avg_training_time_per_round": df["time_taken"].mean()
                        }
                    elif file_type == "sampling":
                        analysis[file_type] = {
                            "avg_actual_sampling_rate": df["actual_client_rate"].mean(),
                            "total_rounds": len(df)
                        }
                    else:
                        analysis[file_type] = {
                            "records": len(df),
                            "columns": list(df.columns)
                        }
                        
                except Exception as e:
                    analysis[file_type] = {"error": str(e)}
            else:
                analysis[file_type] = {"error": "File not found"}
        
        return analysis
    
    def run_experiment_subset(self, 
                            max_experiments: int = None,
                            filter_dataset: str = None,
                            filter_paradigm: str = None,
                            filter_topology: str = None) -> None:
        """Run a subset of experiments with optional filtering"""
        
        experiments_to_run = []
        
        # Generate all experiment combinations
        for dataset in self.datasets:
            if filter_dataset and dataset != filter_dataset:
                continue
                
            for paradigm in self.learning_paradigms:
                if filter_paradigm and paradigm != filter_paradigm:
                    continue
                    
                for topology in self.topologies:
                    if filter_topology and topology != filter_topology:
                        continue
                        
                    if not self.get_topology_compatibility(paradigm, topology):
                        continue
                        
                    for num_clients in self.client_counts:
                        for sampling_rate in self.sampling_rates:
                            for privacy_config in self.privacy_settings:
                                experiments_to_run.append({
                                    "dataset": dataset,
                                    "paradigm": paradigm,
                                    "topology": topology,
                                    "num_clients": num_clients,
                                    "sampling_rate": sampling_rate,
                                    "privacy_config": privacy_config
                                })
        
        # Limit number of experiments if specified
        if max_experiments:
            experiments_to_run = experiments_to_run[:max_experiments]
        
        total_experiments = len(experiments_to_run)
        logger.info(f"Running {total_experiments} experiments...")
        
        # Run experiments
        for i, exp_config in enumerate(experiments_to_run, 1):
            logger.info(f"Progress: {i}/{total_experiments}")
            
            result = self.run_single_experiment(**exp_config)
            self.experiment_results.append(result)
            
            if result["execution"]["status"] != "success":
                self.failed_experiments.append(result)
            
            # Save progress periodically
            if i % 10 == 0:
                self.save_experiment_summary()
        
        # Final summary
        self.save_experiment_summary()
        self.generate_analysis_report()
    
    def save_experiment_summary(self):
        """Save experiment results summary"""
        summary = {
            "experiment_suite": "Resource-Aware Client Sampling for Differentially Private FL",
            "timestamp": self.experiment_timestamp,
            "total_experiments": len(self.experiment_results),
            "successful_experiments": len([r for r in self.experiment_results if r["execution"]["status"] == "success"]),
            "failed_experiments": len(self.failed_experiments),
            "configuration": {
                "datasets": self.datasets,
                "topologies": self.topologies,
                "learning_paradigms": self.learning_paradigms,
                "client_counts": self.client_counts,
                "sampling_rates": self.sampling_rates,
                "privacy_settings": self.privacy_settings,
                "training_config": {
                    "rounds": self.rounds,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size
                }
            },
            "results": self.experiment_results,
            "failed_experiments": self.failed_experiments
        }
        
        # Save summary
        summary_file = self.results_dir / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment summary saved to: {summary_file}")
    
    def generate_analysis_report(self):
        """Generate analysis report from all experiments"""
        logger.info("Generating analysis report...")
        
        # Analyze successful experiments
        successful_experiments = [r for r in self.experiment_results if r["execution"]["status"] == "success"]
        
        analysis_data = []
        for exp in successful_experiments:
            exp_dir = Path(exp["output_directory"])
            resource_analysis = self.analyze_resource_files(exp_dir)
            
            analysis_record = {
                **exp,
                "resource_analysis": resource_analysis
            }
            analysis_data.append(analysis_record)
        
        # Save detailed analysis
        analysis_file = self.results_dir / "detailed_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_data, f, indent=2)
        
        # Generate summary statistics
        self.generate_summary_statistics(analysis_data)
        
        logger.info(f"Analysis report saved to: {analysis_file}")
    
    def generate_summary_statistics(self, analysis_data: List[Dict[str, Any]]):
        """Generate summary statistics across all experiments"""
        logger.info("Generating summary statistics...")
        
        try:
            import pandas as pd
            
            # Convert to DataFrame for analysis
            rows = []
            for exp in analysis_data:
                if "resource_analysis" in exp and "resource_summary" in exp["resource_analysis"]:
                    rs = exp["resource_analysis"]["resource_summary"]
                    if "error" not in rs:
                        rows.append({
                            "dataset": exp["dataset"],
                            "paradigm": exp["paradigm"], 
                            "topology": exp["topology"],
                            "num_clients": exp["num_clients"],
                            "sampling_rate": exp["sampling_rate"],
                            "privacy_setting": exp["privacy_config"]["name"],
                            "total_communication_bytes": rs.get("total_communication_bytes", 0),
                            "avg_round_time": rs.get("avg_round_time", 0),
                            "avg_efficiency_score": rs.get("avg_efficiency_score", 0),
                            "communication_savings_percent": rs.get("avg_communication_savings", 0)
                        })
            
            if rows:
                df = pd.DataFrame(rows)
                
                # Generate summary statistics
                summary_stats = {
                    "sampling_rate_analysis": df.groupby("sampling_rate").agg({
                        "avg_round_time": "mean",
                        "communication_savings_percent": "mean",
                        "avg_efficiency_score": "mean"
                    }).to_dict(),
                    
                    "topology_analysis": df.groupby("topology").agg({
                        "total_communication_bytes": "mean",
                        "avg_round_time": "mean",
                        "avg_efficiency_score": "mean"
                    }).to_dict(),
                    
                    "privacy_analysis": df.groupby("privacy_setting").agg({
                        "avg_round_time": "mean",
                        "avg_efficiency_score": "mean"
                    }).to_dict(),
                    
                    "paradigm_analysis": df.groupby("paradigm").agg({
                        "total_communication_bytes": "mean",
                        "avg_round_time": "mean"
                    }).to_dict()
                }
                
                # Save summary statistics
                stats_file = self.results_dir / "summary_statistics.json"
                with open(stats_file, "w") as f:
                    json.dump(summary_stats, f, indent=2)
                
                # Save raw data CSV for further analysis
                df.to_csv(self.results_dir / "experiment_data.csv", index=False)
                
                logger.info(f"Summary statistics saved to: {stats_file}")
                logger.info(f"Raw experiment data saved to: {self.results_dir / 'experiment_data.csv'}")
            else:
                logger.warning("No valid resource analysis data found for summary statistics")
                
        except ImportError:
            logger.warning("pandas not available for summary statistics generation")
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Resource-Aware Client Sampling Experiments")
    parser.add_argument("--output-dir", default="resource_aware_experiments", 
                       help="Base output directory for experiments")
    parser.add_argument("--max-experiments", type=int, default=None,
                       help="Maximum number of experiments to run (for testing)")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None,
                       help="Filter to specific dataset")
    parser.add_argument("--paradigm", choices=["federated", "decentralized"], default=None,
                       help="Filter to specific learning paradigm")
    parser.add_argument("--topology", choices=["star", "ring", "complete"], default=None,
                       help="Filter to specific topology")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show experiment plan without running")
    
    args = parser.parse_args()
    
    # Create experiment suite
    experiments = ResourceAwareSamplingExperiments(args.output_dir)
    
    if args.dry_run:
        # Show experiment plan
        total_configs = (
            len(experiments.datasets) * 
            len(experiments.learning_paradigms) * 
            len(experiments.topologies) * 
            len(experiments.client_counts) * 
            len(experiments.sampling_rates) * 
            len(experiments.privacy_settings)
        )
        
        # Account for topology compatibility
        compatible_configs = 0
        for dataset in experiments.datasets:
            for paradigm in experiments.learning_paradigms:
                for topology in experiments.topologies:
                    if experiments.get_topology_compatibility(paradigm, topology):
                        compatible_configs += (
                            len(experiments.client_counts) * 
                            len(experiments.sampling_rates) * 
                            len(experiments.privacy_settings)
                        )
        
        print(f"Experiment Plan:")
        print(f"  Total possible configurations: {total_configs}")
        print(f"  Compatible configurations: {compatible_configs}")
        print(f"  Datasets: {experiments.datasets}")
        print(f"  Learning paradigms: {experiments.learning_paradigms}")
        print(f"  Topologies: {experiments.topologies}")
        print(f"  Client counts: {experiments.client_counts}")
        print(f"  Sampling rates: {experiments.sampling_rates}")
        print(f"  Privacy settings: {len(experiments.privacy_settings)} configurations")
        
        if args.max_experiments:
            print(f"  Limited to: {args.max_experiments} experiments")
        
        return
    
    # Run experiments
    experiments.run_experiment_subset(
        max_experiments=args.max_experiments,
        filter_dataset=args.dataset,
        filter_paradigm=args.paradigm,
        filter_topology=args.topology
    )
    
    print(f"\n🎉 Experiment suite completed!")
    print(f"📊 Results saved to: {experiments.results_dir}")
    print(f"✅ Successful experiments: {len([r for r in experiments.experiment_results if r['execution']['status'] == 'success'])}")
    print(f"❌ Failed experiments: {len(experiments.failed_experiments)}")


if __name__ == "__main__":
    main()