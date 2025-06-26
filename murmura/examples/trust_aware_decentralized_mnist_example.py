"""
Trust-aware decentralized MNIST example with HSIC-based trust monitoring.

This example demonstrates how to use trust monitoring to detect and mitigate
malicious behavior in decentralized federated learning.
"""

import argparse
import os
import logging
import torch
import torch.nn as nn

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.models.mnist_models import MNISTModel
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.node.resource_config import RayClusterConfig, ResourceConfig
from murmura.orchestration.learning_process.trust_aware_decentralized_learning_process import (
    TrustAwareDecentralizedLearningProcess,
)
from murmura.network_management.topology_compatibility import (
    TopologyCompatibilityManager,
)
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.visualization.network_visualizer import NetworkVisualizer

# Import trust components
from murmura.trust.trust_config import (
    TrustMonitoringConfig,
    create_default_trust_config,
    create_strict_trust_config,
    create_relaxed_trust_config,
)

# Import DP components if needed
from murmura.privacy.dp_config import DPConfig
from murmura.privacy.dp_model_wrapper import DPTorchModelWrapper
from murmura.privacy.privacy_accountant import PrivacyAccountant

# Import attack components
from murmura.attacks.attack_config import (
    AttackConfig,
    AttackType,
    AttackTiming,
    create_progressive_label_attack,
    create_backdoor_attack,
    create_byzantine_attack,
)


def create_mnist_preprocessor():
    """
    Create MNIST-specific data preprocessor.
    """
    try:
        from murmura.data_processing.generic_preprocessor import (  # type: ignore[import-untyped]
            create_image_preprocessor,
        )

        # MNIST-specific configuration
        return create_image_preprocessor(
            grayscale=True,  # MNIST is grayscale
            normalize=True,  # Normalize to [0,1]
            target_size=None,  # Keep original 28x28
        )
    except ImportError:
        # Generic preprocessor not available
        logging.getLogger("murmura.trust_aware_mnist_example").info(
            "Using automatic data type detection"
        )
        return None


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("trust_aware_decentralized_mnist.log"),
        ],
    )


def main() -> None:
    """
    Trust-aware MNIST Decentralized Learning Example
    """
    parser = argparse.ArgumentParser(
        description="Trust-aware MNIST Decentralized Learning Example"
    )

    # Core learning arguments
    parser.add_argument(
        "--num_actors", type=int, default=10, help="Total number of virtual clients"
    )
    parser.add_argument(
        "--partition_strategy",
        choices=[
            "dirichlet",
            "iid",
            "sensitive_groups",
            "topology_correlated",
            "imbalanced_sensitive",
        ],
        default="dirichlet",
        help="Data partitioning strategy",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet concentration parameter"
    )
    parser.add_argument(
        "--min_partition_size",
        type=int,
        default=500,
        help="Minimum samples per partition",
    )
    
    # Training parameters
    parser.add_argument(
        "--rounds", type=int, default=20, help="Number of FL rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for local training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    
    # Topology configuration
    parser.add_argument(
        "--topology_type",
        choices=["ring", "complete", "grid", "random", "small_world"],
        default="ring",
        help="Network topology type",
    )
    
    # Trust monitoring arguments
    parser.add_argument(
        "--enable_trust",
        action="store_true",
        help="Enable trust monitoring",
    )
    parser.add_argument(
        "--trust_profile",
        choices=["default", "strict", "relaxed", "custom"],
        default="default",
        help="Trust monitoring profile",
    )
    parser.add_argument(
        "--hsic_window_size",
        type=int,
        default=50,
        help="HSIC sliding window size",
    )
    parser.add_argument(
        "--hsic_threshold",
        type=float,
        default=0.1,
        help="HSIC threshold for drift detection",
    )
    parser.add_argument(
        "--trust_report_interval",
        type=int,
        default=5,
        help="Rounds between trust reports",
    )
    
    # Attack simulation arguments
    parser.add_argument(
        "--attack_type",
        choices=["none", "label_flipping", "backdoor", "byzantine"],
        default="none",
        help="Type of attack to simulate",
    )
    parser.add_argument(
        "--malicious_fraction",
        type=float,
        default=0.25,
        help="Fraction of actors that are malicious",
    )
    parser.add_argument(
        "--attack_start_round",
        type=int,
        default=3,
        help="Round to start the attack",
    )
    parser.add_argument(
        "--attack_intensity",
        type=float,
        default=0.1,
        help="Initial attack intensity (0.0 to 1.0)",
    )
    parser.add_argument(
        "--attack_max_intensity",
        type=float,
        default=0.8,
        help="Maximum attack intensity",
    )
    parser.add_argument(
        "--attack_stealth",
        action="store_true",
        help="Enable stealth mode for attacks",
    )
    parser.add_argument(
        "--attack_ramp_rounds",
        type=int,
        default=5,
        help="Number of rounds to ramp up attack intensity",
    )
    
    # Differential privacy arguments
    parser.add_argument(
        "--enable_dp",
        action="store_true",
        help="Enable differential privacy",
    )
    parser.add_argument(
        "--epsilon", type=float, default=10.0, help="Target privacy budget epsilon"
    )
    parser.add_argument(
        "--delta", type=float, default=1e-5, help="Target privacy budget delta"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm"
    )
    
    # Resource configuration
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help="Ray cluster address (None for local)",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable network visualization"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trust_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.trust_aware_mnist_example")
    logger.info("Starting Trust-aware Decentralized MNIST Learning")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create CNN model
    base_model = MNISTModel()
    preprocessor = create_mnist_preprocessor()
    
    # Apply differential privacy if enabled
    if args.enable_dp:
        logger.info("Enabling differential privacy")
        dp_config = DPConfig(
            epsilon=args.epsilon,
            delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            mechanism="gaussian",
            adaptive_clipping=True,
        )
        
        model = DPTorchModelWrapper(
            model=base_model,
            dp_config=dp_config,
            input_shape=(1, 28, 28),
            num_classes=10,
            data_preprocessor=preprocessor,
        )
        
        # Initialize privacy accountant
        privacy_accountant = PrivacyAccountant(dp_config)
    else:
        # Create standard torch model wrapper
        from murmura.model.pytorch_model import TorchModelWrapper
        
        model = TorchModelWrapper(
            model=base_model,
            input_shape=(1, 28, 28),
            data_preprocessor=preprocessor,
        )

    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    dataset = MDataset.load(
        DatasetSource.HUGGING_FACE,
        dataset_name="mnist",
        split=None,  # Load all splits
    )
    logger.info(f"Dataset splits: {dataset.available_splits}")

    # Create a temporary config for partitioner
    temp_config = OrchestrationConfig(
        num_actors=args.num_actors,
        partition_strategy=args.partition_strategy,
        alpha=args.alpha,
        min_partition_size=args.min_partition_size,
        feature_columns=["image"],
        label_column="label",
        dataset_name="mnist",
    )
    
    # Partition the data
    logger.info(f"Creating {args.partition_strategy} partitions...")
    partitioner = PartitionerFactory.create(temp_config)
    partitioner.partition(dataset, "train")

    # Configure trust monitoring
    if args.enable_trust:
        logger.info(f"Configuring trust monitoring with {args.trust_profile} profile")
        
        if args.trust_profile == "default":
            trust_config = create_default_trust_config()
        elif args.trust_profile == "strict":
            trust_config = create_strict_trust_config()
        elif args.trust_profile == "relaxed":
            trust_config = create_relaxed_trust_config()
        else:  # custom
            from murmura.trust.trust_config import HSICConfig, TrustPolicyConfig
            trust_config = TrustMonitoringConfig(
                enabled=True,
                hsic_config=HSICConfig(
                    window_size=args.hsic_window_size,
                    threshold=args.hsic_threshold,
                ),
                trust_policy_config=TrustPolicyConfig(),
                trust_report_interval=args.trust_report_interval,
            )
    else:
        trust_config = TrustMonitoringConfig(enabled=False)

    # Configure attack simulation
    attack_config = None
    if args.attack_type != "none":
        logger.info(f"Configuring {args.attack_type} attack simulation...")
        
        if args.attack_type == "label_flipping":
            attack_config = create_progressive_label_attack(
                malicious_fraction=args.malicious_fraction,
                stealth=args.attack_stealth
            )
        elif args.attack_type == "backdoor":
            attack_config = create_backdoor_attack(
                malicious_fraction=args.malicious_fraction,
                stealth=args.attack_stealth
            )
        elif args.attack_type == "byzantine":
            attack_config = create_byzantine_attack(
                malicious_fraction=args.malicious_fraction,
                stealth=args.attack_stealth
            )
        
        # Override with custom parameters if provided
        if attack_config:
            attack_config.start_round = args.attack_start_round
            attack_config.initial_intensity = args.attack_intensity
            attack_config.max_intensity = args.attack_max_intensity
            attack_config.ramp_up_rounds = args.attack_ramp_rounds
            
            logger.info(f"Attack configuration:")
            logger.info(f"  Type: {attack_config.attack_type.value}")
            logger.info(f"  Malicious fraction: {attack_config.malicious_fraction}")
            logger.info(f"  Start round: {attack_config.start_round}")
            logger.info(f"  Initial intensity: {attack_config.initial_intensity}")
            logger.info(f"  Max intensity: {attack_config.max_intensity}")
            logger.info(f"  Stealth mode: {attack_config.stealth_mode}")
    else:
        logger.info("No attack simulation configured")

    # Configure topology
    topology_config = TopologyConfig(
        topology_type=TopologyType(args.topology_type),
        num_clients=args.num_actors,
    )

    # Validate topology-aggregation compatibility
    compatibility_manager = TopologyCompatibilityManager()
    aggregation_strategy = AggregationStrategyType.GOSSIP_AVG
    
    if not compatibility_manager.is_compatible(
        topology_config.topology_type, aggregation_strategy
    ):
        suggested = compatibility_manager.get_compatible_strategies(topology_config.topology_type)
        logger.warning(
            f"Topology {topology_config.topology_type} may not be optimal with "
            f"{aggregation_strategy}. Suggested strategies: {suggested}"
        )

    # Configure aggregation
    aggregation_config = AggregationConfig(
        strategy_type=aggregation_strategy,
        params={"mixing_parameter": 0.5},
    )

    # Configure Ray cluster
    ray_cluster_config = RayClusterConfig(
        address=args.ray_address,
        namespace="trust_aware_mnist",
        runtime_env={
            "env_vars": {
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "MURMURA_LOG_LEVEL": args.log_level,
            }
        },
    )

    # Configure resources - let Ray handle allocation automatically
    resource_config = ResourceConfig(
        placement_strategy="spread",
    )

    # Create orchestration config
    orchestration_config = OrchestrationConfig(
        num_actors=args.num_actors,
        topology=topology_config,
        aggregation=aggregation_config,
        ray_cluster=ray_cluster_config,
        resources=resource_config,
        dataset_name="mnist",
        partition_strategy=args.partition_strategy,
        alpha=args.alpha,
        min_partition_size=args.min_partition_size,
        feature_columns=["image"],
        label_column="label",
        rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        trust_monitoring=trust_config,  # Add trust config
    )

    # Create trust-aware learning process
    learning_process = TrustAwareDecentralizedLearningProcess(
        config=orchestration_config,
        dataset=dataset,
        model=model,
    )
    
    # Initialize the learning process with attack configuration
    learning_process.initialize(
        num_actors=orchestration_config.num_actors,
        topology_config=orchestration_config.topology,
        aggregation_config=orchestration_config.aggregation,
        partitioner=partitioner,
        attack_config=attack_config.to_dict() if attack_config else None,
    )

    # Enable visualization if requested
    if args.visualize:
        visualizer = NetworkVisualizer(
            save_dir=os.path.join(args.output_dir, "visualizations")
        )
        learning_process.set_training_monitor(visualizer)
        logger.info("Network visualization enabled")

    # Execute learning
    logger.info("Starting trust-aware decentralized learning...")
    results = learning_process.execute()

    # Log results
    logger.info("\n=== Trust-aware Learning Results ===")
    logger.info(
        f"Initial Accuracy: {results['initial_metrics']['accuracy'] * 100:.2f}%"
    )
    logger.info(f"Final Accuracy: {results['final_metrics']['accuracy'] * 100:.2f}%")
    logger.info(f"Improvement: {results['accuracy_improvement'] * 100:.2f}%")
    
    if args.enable_trust and "final_trust_report" in results:
        trust_report = results["final_trust_report"]
        logger.info("\n=== Trust Monitoring Summary ===")
        logger.info(
            f"Average Trust Score: {trust_report['global_stats']['avg_trust_score']:.3f}"
        )
        logger.info(
            f"Excluded Nodes: {trust_report['global_stats']['total_excluded']}"
        )
        logger.info(
            f"Downgraded Nodes: {trust_report['global_stats']['total_downgraded']}"
        )
    
    # Log attack statistics if available
    if args.attack_type != "none" and "attack_statistics" in results:
        attack_stats = results["attack_statistics"]
        logger.info("\n=== Attack Simulation Summary ===")
        logger.info(f"Attack Type: {args.attack_type}")
        logger.info(f"Malicious Fraction: {args.malicious_fraction}")
        logger.info(f"Total Attacks Applied: {attack_stats.get('total_attacks', 0)}")
        logger.info(f"Attack Detection Rate: {attack_stats.get('detection_rate', 0):.3f}")
        logger.info(f"Total Attackers: {attack_stats.get('total_attackers', 0)}")
        logger.info(f"Detected Attackers: {attack_stats.get('detected_attacks', 0)}")
        
        # Log per-attacker details
        for attacker_id, attacker_stats in attack_stats.get("per_attacker", {}).items():
            logger.info(f"  {attacker_id}: "
                       f"Attacks={attacker_stats.get('attacks_applied', 0)}, "
                       f"Detected={attacker_stats.get('detected', False)}, "
                       f"Final Intensity={attacker_stats.get('final_intensity', 0):.3f}")

    # Log privacy metrics if DP enabled
    if args.enable_dp:
        privacy_metrics = privacy_accountant.get_privacy_metrics()
        logger.info("\n=== Privacy Metrics ===")
        logger.info(f"Total privacy spent: (ε={privacy_metrics['epsilon']:.2f}, δ={privacy_metrics['delta']:.2e})")

    # Save results
    import json
    
    results_file = os.path.join(args.output_dir, "trust_aware_results.json")
    with open(results_file, "w") as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, "item"):
                return obj.item()
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info(f"\nResults saved to {results_file}")

    # Cleanup
    try:
        learning_process.cleanup()
    except AttributeError:
        # Cleanup method may not exist in all versions
        pass
    logger.info("Trust-aware learning completed successfully!")


if __name__ == "__main__":
    main()