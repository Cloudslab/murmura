"""
True Decentralized Trust-Aware MNIST Example

This example demonstrates:
- True decentralized learning (no global aggregation)
- Performance-based trust monitoring
- Proper evaluation of network consensus
- Detection of gradual label flipping attacks
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, Any, Optional

import numpy as np
import ray

from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner import IIDPartitioner
from murmura.models.mnist_models import MNISTModel
from murmura.model.pytorch_model import TorchModelWrapper
from murmura.orchestration.orchestration_config import (
    OrchestrationConfig,
    RayClusterConfig,
    ResourceConfig,
)
from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.orchestration.learning_process.trust_aware_true_decentralized_learning_process import (
    TrustAwareTrueDecentralizedLearningProcess
)
from murmura.trust.trust_config import (
    TrustMonitoringConfig,
    HSICConfig,
    TrustPolicyConfig,
)


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('true_decentralized_trust_mnist.log')
        ]
    )


def create_attack_config(
    attack_type: str = "none",
    malicious_fraction: float = 0.0,
    attack_intensity: str = "moderate",
    stealth_level: str = "medium"
) -> Optional[Dict[str, Any]]:
    """Create attack configuration."""
    if attack_type == "none" or malicious_fraction == 0:
        return None
    
    config = {
        "attack_type": attack_type,
        "malicious_fraction": malicious_fraction,
        "attack_intensity": attack_intensity,
        "stealth_level": stealth_level,
    }
    
    if attack_type == "gradual_label_flipping":
        config.update({
            "start_round": 1,
            "subtle_intensity": 0.05,
            "moderate_intensity": 0.15,
            "aggressive_intensity": 0.30,
            "maximum_intensity": 0.50,
        })
    
    return config


def create_trust_config(profile: str = "default") -> TrustMonitoringConfig:
    """Create trust monitoring configuration based on profile."""
    if profile == "aggressive":
        return TrustMonitoringConfig(
            enabled=True,
            trust_report_interval=2,
            hsic_config=HSICConfig(
                kernel_type="rbf",
                gamma=0.5,
                threshold=0.05,
                window_size=30,
                calibration_rounds=3,
            ),
            trust_policy_config=TrustPolicyConfig(
                warn_threshold=0.15,
                downgrade_threshold=0.25,
                exclude_threshold=0.4,
                reputation_window=10,
                min_samples_for_action=5,
            )
        )
    elif profile == "moderate":
        return TrustMonitoringConfig(
            enabled=True,
            trust_report_interval=3,
            hsic_config=HSICConfig(
                kernel_type="rbf",
                gamma=0.1,
                threshold=0.1,
                window_size=50,
                calibration_rounds=5,
            ),
            trust_policy_config=TrustPolicyConfig(
                warn_threshold=0.15,
                downgrade_threshold=0.3,
                exclude_threshold=0.5,
                reputation_window=20,
                min_samples_for_action=10,
            )
        )
    else:  # default
        return TrustMonitoringConfig(
            enabled=True,
            trust_report_interval=5,
            hsic_config=HSICConfig(
                kernel_type="rbf",
                gamma=0.1,
                threshold=0.1,
            ),
            trust_policy_config=TrustPolicyConfig(
                warn_threshold=0.15,
                downgrade_threshold=0.3,
                exclude_threshold=0.5,
            )
        )


def run_true_decentralized_experiment(
    num_actors: int = 4,
    num_rounds: int = 10,
    topology_type: str = "ring",
    trust_profile: str = "aggressive",
    attack_config: Optional[Dict[str, Any]] = None,
    output_dir: str = "true_decentralized_results",
    log_level: str = "INFO"
) -> Dict[str, Any]:
    """
    Run true decentralized trust-aware MNIST experiment.
    
    Args:
        num_actors: Number of federated learning actors
        num_rounds: Number of training rounds
        topology_type: Network topology type
        trust_profile: Trust monitoring profile
        attack_config: Attack configuration
        output_dir: Output directory for results
        log_level: Logging level
        
    Returns:
        Experiment results
    """
    setup_logging(log_level)
    logger = logging.getLogger("true_decentralized_mnist")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("TRUE DECENTRALIZED TRUST-AWARE MNIST EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Actors: {num_actors}, Rounds: {num_rounds}")
    logger.info(f"Topology: {topology_type}")
    logger.info(f"Trust Profile: {trust_profile}")
    logger.info(f"Attacks: {attack_config.get('attack_type', 'None') if attack_config else 'None'}")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Initialize Ray
    logger.info("Initializing Ray...")
    ray.init(address="local", ignore_reinit_error=True)
    
    # Log cluster resources
    resources = ray.cluster_resources()
    logger.info(f"Ray cluster resources: {resources}")
    
    # Create model
    logger.info("Creating MNIST CNN model...")
    base_model = MNISTModel()
    model = TorchModelWrapper(base_model)
    
    # Load dataset
    logger.info("Loading MNIST dataset...")
    dataset = MDataset.load(
        source=DatasetSource.HUGGING_FACE,
        dataset_name="mnist",
        split=["train", "test"]
    )
    logger.info(f"Dataset splits available: {dataset.available_splits}")
    
    # Create partitions
    logger.info("Creating IID data partitions...")
    partitioner = IIDPartitioner(num_partitions=num_actors)
    
    # Create orchestration config
    orchestration_config = OrchestrationConfig(
        num_actors=num_actors,
        rounds=num_rounds,
        epochs=2,
        batch_size=32,
        dataset_name="mnist",
        partition_strategy="iid",
        split="train",
        test_split="test",
        feature_columns=["image"],
        label_column="label",
        client_sampling_rate=1.0,
        data_sampling_rate=1.0,
        ray_cluster=RayClusterConfig(address="local"),
        resources=ResourceConfig(),
        trust_monitoring=create_trust_config(trust_profile),
    )
    
    # Create aggregation config (gossip for decentralized)
    aggregation_config = AggregationConfig(
        strategy_type=AggregationStrategyType.GOSSIP_AVG,
        mixing_parameter=0.5
    )
    
    # Create topology config
    topology_config = TopologyConfig(
        topology_type=TopologyType(topology_type)
    )
    
    # Initialize learning process
    logger.info("Initializing trust-aware true decentralized learning process...")
    learning_process = TrustAwareTrueDecentralizedLearningProcess(
        config=orchestration_config,
        dataset=dataset,
        model=model
    )
    
    # Initialize with attack config if provided
    learning_process.initialize(
        num_actors=num_actors,
        topology_config=topology_config,
        aggregation_config=aggregation_config,
        partitioner=partitioner,
        attack_config=attack_config
    )
    
    # Execute learning
    logger.info("Starting true decentralized learning...")
    results = learning_process.execute()
    
    # Calculate metrics
    elapsed_time = time.time() - start_time
    
    # Extract key metrics
    final_consensus = results["final_metrics"]["consensus_accuracy"]
    initial_consensus = results["initial_metrics"]["consensus_accuracy"]
    consensus_improvement = results["consensus_improvement"]
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT RESULTS")
    logger.info("=" * 60)
    logger.info(f"Initial Consensus Accuracy: {initial_consensus:.4f}")
    logger.info(f"Final Consensus Accuracy: {final_consensus:.4f}")
    logger.info(f"Consensus Improvement: {consensus_improvement:.4f}")
    logger.info(f"Total Training Time: {elapsed_time:.2f}s")
    
    # Log individual node performance
    logger.info("\nFinal Node Performance:")
    for node_metric in results["final_metrics"]["individual_metrics"]:
        if "error" not in node_metric:
            logger.info(
                f"  Node {node_metric['node_id']}: "
                f"Accuracy={node_metric['accuracy']:.4f}"
            )
    
    # Log trust results if enabled
    if results.get("trust_enabled", False):
        logger.info("\nTrust Monitoring Results:")
        if "trust_metrics" in results and results["trust_metrics"]:
            last_trust = results["trust_metrics"][-1]
            logger.info(f"  Performance Anomalies: {last_trust.get('performance_anomalies', 0)}")
            if last_trust.get("suspected_malicious"):
                logger.info(f"  Suspected Malicious: {last_trust['suspected_malicious']}")
        
        # Log attack detection results
        if "attack_statistics" in results:
            stats = results["attack_statistics"]
            logger.info(f"  Total Attackers: {stats.get('total_attackers', 0)}")
            logger.info(f"  Detected Attackers: {stats.get('detected_attacks', 0)}")
            logger.info(f"  Detection Rate: {stats.get('detection_rate', 0):.2%}")
    
    # Save results
    output_file = os.path.join(
        output_dir, 
        f"true_decentralized_results_{int(time.time())}.json"
    )
    
    with open(output_file, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        json_results = {
            "config": {
                "num_actors": num_actors,
                "num_rounds": num_rounds,
                "topology": topology_type,
                "trust_profile": trust_profile,
                "attack_config": attack_config,
            },
            "metrics": {
                "initial_consensus": float(initial_consensus),
                "final_consensus": float(final_consensus),
                "improvement": float(consensus_improvement),
                "training_time": elapsed_time,
            },
            "final_node_metrics": [
                {
                    "node_id": m["node_id"],
                    "accuracy": float(m["accuracy"]) if "accuracy" in m else None,
                    "error": m.get("error")
                }
                for m in results["final_metrics"]["individual_metrics"]
            ],
            "trust_results": {
                "enabled": results.get("trust_enabled", False),
                "attack_detection": results.get("attack_statistics", {})
            }
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {output_file}")
    logger.info("=" * 60)
    logger.info("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    # Shutdown learning process
    learning_process.shutdown()
    
    return results


def main():
    """Main function to run experiment from command line."""
    parser = argparse.ArgumentParser(
        description="Run True Decentralized Trust-Aware MNIST Experiment"
    )
    
    parser.add_argument(
        "--actors", type=int, default=4,
        help="Number of federated learning actors"
    )
    parser.add_argument(
        "--rounds", type=int, default=10,
        help="Number of training rounds"
    )
    parser.add_argument(
        "--topology", type=str, default="ring",
        choices=["ring", "line", "complete"],
        help="Network topology type"
    )
    parser.add_argument(
        "--trust-profile", type=str, default="aggressive",
        choices=["default", "moderate", "aggressive"],
        help="Trust monitoring profile"
    )
    parser.add_argument(
        "--attack-type", type=str, default="none",
        choices=["none", "gradual_label_flipping"],
        help="Type of attack to simulate"
    )
    parser.add_argument(
        "--malicious-fraction", type=float, default=0.25,
        help="Fraction of malicious actors"
    )
    parser.add_argument(
        "--output-dir", type=str, default="true_decentralized_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Create attack config
    attack_config = None
    if args.attack_type != "none":
        attack_config = create_attack_config(
            attack_type=args.attack_type,
            malicious_fraction=args.malicious_fraction,
            attack_intensity="moderate",
            stealth_level="medium"
        )
    
    # Run experiment
    run_true_decentralized_experiment(
        num_actors=args.actors,
        num_rounds=args.rounds,
        topology_type=args.topology,
        trust_profile=args.trust_profile,
        attack_config=attack_config,
        output_dir=args.output_dir,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()