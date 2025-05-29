#!/usr/bin/env python3
"""
DP-Enhanced MNIST Example - Central Differential Privacy for Federated Learning
Updated to include all arguments from the regular MNIST example for consistency.
"""

import argparse
import os
import logging

import torch
import torch.nn as nn

from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
from murmura.privacy.dp_config import (
    DifferentialPrivacyConfig,
    DPMechanism,
    NoiseApplication,
    ClippingStrategy,
    DPAccountant,
)
from murmura.privacy.dp_integration import (
    DPOrchestrationConfig,
    create_dp_federated_learning_process,
)
from murmura.visualization.network_visualizer import NetworkVisualizer


class MNISTModel(PyTorchModel):
    """
    Simple CNN model for MNIST classification
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        # Ensure input has the right shape (add channel dimension if needed)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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
        # Generic preprocessor not available, use automatic detection
        logging.getLogger("murmura.dp_mnist_example").info(
            "Using automatic data type detection"
        )
        return None


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dp_mnist_federated.log"),
        ],
    )


def create_dp_config(args) -> DifferentialPrivacyConfig:
    """Create differential privacy configuration."""
    return DifferentialPrivacyConfig(
        # Core privacy parameters
        epsilon=args.epsilon,
        delta=args.delta,
        # Mechanism configuration
        mechanism=DPMechanism.GAUSSIAN,
        noise_application=NoiseApplication.SERVER_SIDE,  # Central DP
        # Clipping configuration
        clipping_strategy=ClippingStrategy.ADAPTIVE,
        clipping_norm=args.clipping_norm,
        target_quantile=0.5,
        # Privacy accounting
        accountant=DPAccountant.RDP,
        # Client sampling for privacy amplification
        client_sampling_rate=args.client_sampling_rate,
        # Monitoring
        enable_privacy_monitoring=True,
        privacy_budget_warning_threshold=0.8,
        # Total rounds for budget allocation
        total_rounds=args.rounds,
    )


def main():
    """Main function for DP-enhanced MNIST federated learning."""
    parser = argparse.ArgumentParser(
        description="DP-Enhanced MNIST Federated Learning with Multi-Node Support"
    )

    # Core federated learning arguments (from regular MNIST example)
    parser.add_argument(
        "--num_actors", type=int, default=10, help="Total number of virtual clients"
    )
    parser.add_argument(
        "--partition_strategy",
        choices=["dirichlet", "iid"],
        default="dirichlet",
        help="Data partitioning strategy",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet concentration parameter"
    )
    parser.add_argument(
        "--min_partition_size",
        type=int,
        default=100,
        help="Minimum samples per partition",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--test_split", type=str, default="test", help="Test split to use"
    )
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        choices=["fedavg", "trimmed_mean"],
        default="fedavg",
        help="Aggregation strategy to use",
    )
    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.1,
        help="Trim ratio for trimmed_mean strategy",
    )

    # Topology arguments (from regular MNIST example)
    parser.add_argument(
        "--topology",
        type=str,
        default="star",
        choices=["star", "ring", "complete", "line", "custom"],
        help="Network topology between clients",
    )
    parser.add_argument(
        "--hub_index", type=int, default=0, help="Hub node index for star topology"
    )

    # Training arguments (from regular MNIST example)
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--save_path",
        type=str,
        default="dp_mnist_federated_model.pt",
        help="Path to save the final model",
    )

    # Multi-node Ray cluster arguments (from regular MNIST example)
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help="Ray cluster address. If None, uses local cluster.",
    )
    parser.add_argument(
        "--ray_namespace",
        type=str,
        default="murmura_dp_mnist",
        help="Ray namespace for isolation",
    )
    parser.add_argument(
        "--actors_per_node",
        type=int,
        default=None,
        help="Number of actors per physical node. If None, distributes evenly.",
    )
    parser.add_argument(
        "--cpus_per_actor",
        type=float,
        default=1.0,
        help="CPU resources per actor",
    )
    parser.add_argument(
        "--gpus_per_actor",
        type=float,
        default=None,
        help="GPU resources per actor. If None, auto-calculated.",
    )
    parser.add_argument(
        "--memory_per_actor",
        type=int,
        default=None,
        help="Memory (MB) per actor",
    )
    parser.add_argument(
        "--placement_strategy",
        type=str,
        choices=["spread", "pack", "strict_spread", "strict_pack"],
        default="spread",
        help="Actor placement strategy across nodes",
    )
    parser.add_argument(
        "--auto_detect_cluster",
        action="store_true",
        help="Auto-detect Ray cluster from environment variables",
    )

    # MNIST-specific arguments (from regular MNIST example)
    parser.add_argument(
        "--debug_data",
        action="store_true",
        help="Print debug information about MNIST data format",
    )

    # Logging and monitoring arguments (from regular MNIST example)
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--monitor_resources",
        action="store_true",
        help="Monitor and log resource usage during training",
    )
    parser.add_argument(
        "--health_check_interval",
        type=int,
        default=5,
        help="Interval (rounds) for actor health checks",
    )

    # Visualization arguments (from regular MNIST example)
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--create_animation",
        action="store_true",
        help="Create animation of the training process",
    )
    parser.add_argument(
        "--create_frames",
        action="store_true",
        help="Create individual frames of the training process",
    )
    parser.add_argument(
        "--create_summary",
        action="store_true",
        help="Create summary plot of the training process",
    )
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second for animation"
    )

    # DP-specific arguments
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Privacy budget (ε) - lower values = stronger privacy",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Failure probability (δ) - should be < 1/dataset_size",
    )
    parser.add_argument(
        "--clipping_norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold for DP",
    )
    parser.add_argument(
        "--client_sampling_rate",
        type=float,
        default=0.1,
        help="Client sampling rate for privacy amplification",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.dp_mnist_example")

    try:
        # Create DP configuration
        dp_config = create_dp_config(args)
        logger.info("Created DP configuration")

        # Create enhanced orchestration configuration
        from murmura.aggregation.aggregation_config import (
            AggregationConfig,
            AggregationStrategyType,
        )
        from murmura.network_management.topology import TopologyConfig, TopologyType
        from murmura.node.resource_config import RayClusterConfig, ResourceConfig

        # Create enhanced configuration with multi-node support
        ray_cluster_config = RayClusterConfig(
            address=args.ray_address,
            namespace=args.ray_namespace,
            logging_level=args.log_level,
            auto_detect_cluster=args.auto_detect_cluster,
        )

        resource_config = ResourceConfig(
            actors_per_node=args.actors_per_node,
            cpus_per_actor=args.cpus_per_actor,
            gpus_per_actor=args.gpus_per_actor,
            memory_per_actor=args.memory_per_actor,
            placement_strategy=args.placement_strategy,
        )

        config = DPOrchestrationConfig(
            # Core FL parameters
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,
            # Network topology
            topology=TopologyConfig(
                topology_type=TopologyType(args.topology), hub_index=args.hub_index
            ),
            # Aggregation strategy
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType(args.aggregation_strategy),
                params={"trim_ratio": args.trim_ratio}
                if args.aggregation_strategy == "trimmed_mean"
                else None,
            ),
            # Dataset configuration
            dataset_name="mnist",  # Fixed to MNIST
            ray_cluster=ray_cluster_config,
            resources=resource_config,
            # CRITICAL: Must specify feature and label columns
            feature_columns=["image"],  # MNIST images
            label_column="label",  # MNIST labels
            rounds=args.rounds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            test_split=args.test_split,
            monitor_resources=args.monitor_resources,
            health_check_interval=args.health_check_interval,
            # Differential Privacy configuration
            differential_privacy=dp_config,
            enable_privacy_dashboard=True,
        )

        logger.info("Created DP orchestration configuration")

        logger.info("=== Loading MNIST Dataset ===")
        # Load MNIST Dataset for training and testing
        train_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="mnist",
            split=config.split,
        )

        test_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="mnist",
            split=args.test_split,
        )

        # Merge datasets to have both splits available
        train_dataset.merge_splits(test_dataset)

        # Debug MNIST data format if requested
        if args.debug_data:
            logger.info("=== Debugging MNIST Data Format ===")
            try:
                split_dataset = train_dataset.get_split(config.split)
                feature_data = split_dataset["image"]

                logger.info(f"MNIST {config.split} split")
                logger.info(f"Number of samples: {len(feature_data)}")

                if len(feature_data) > 0:
                    sample = feature_data[0]
                    logger.info(f"Sample type: {type(sample)}")
                    logger.info(f"Sample shape: {getattr(sample, 'shape', 'N/A')}")
                    logger.info(f"Sample mode: {getattr(sample, 'mode', 'N/A')}")
                    if hasattr(sample, "size"):
                        logger.info(f"Sample size: {sample.size}")
            except Exception as e:
                logger.error(f"Error debugging MNIST data format: {e}")

        logger.info("=== Creating MNIST Model ===")
        model = MNISTModel()
        input_shape = (1, 28, 28)  # MNIST: 1 channel, 28x28 pixels

        # Create MNIST-specific data preprocessor
        mnist_preprocessor = create_mnist_preprocessor()

        global_model = TorchModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": args.lr},
            input_shape=input_shape,
            data_preprocessor=mnist_preprocessor,
        )
        logger.info("Created model wrapper")

        logger.info("=== Setting Up DP-Enhanced Learning Process ===")
        learning_process = create_dp_federated_learning_process(
            config, train_dataset, global_model
        )
        logger.info("Created DP-enhanced learning process")

        # Set up visualization if requested
        visualizer = None
        if args.create_animation or args.create_frames or args.create_summary:
            logger.info("=== Setting Up Visualization ===")
            vis_dir = os.path.join(
                args.vis_dir, f"dp_mnist_{args.topology}_{args.aggregation_strategy}"
            )
            os.makedirs(vis_dir, exist_ok=True)

            visualizer = NetworkVisualizer(output_dir=vis_dir)
            learning_process.register_observer(visualizer)
            logger.info("Registered visualizer with learning process")

        try:
            # Initialize the learning process
            partitioner = PartitionerFactory.create(config)
            logger.info("Created partitioner")

            logger.info("=== Initializing Learning Process ===")
            learning_process.initialize(
                num_actors=config.num_actors,
                topology_config=config.topology,
                aggregation_config=config.aggregation,
                partitioner=partitioner,
            )

            # Get and log cluster information
            cluster_summary = learning_process.get_cluster_summary()
            logger.info("=== Enhanced Cluster Summary ===")
            logger.info(
                f"Cluster type: {cluster_summary.get('cluster_type', 'unknown')}"
            )
            logger.info(f"Total nodes: {cluster_summary.get('total_nodes', 'unknown')}")
            logger.info(
                f"Total actors: {cluster_summary.get('total_actors', 'unknown')}"
            )
            logger.info(f"Topology: {cluster_summary.get('topology', 'unknown')}")
            logger.info(
                f"Placement strategy: {cluster_summary.get('placement_strategy', 'unknown')}"
            )
            logger.info(
                f"Has placement group: {cluster_summary.get('has_placement_group', False)}"
            )

            # Log privacy configuration
            privacy_desc = dp_config.get_privacy_description()
            utility_impact = dp_config.estimate_utility_impact()

            logger.info("=== Privacy Configuration ===")
            logger.info(f"Privacy Guarantee: {privacy_desc}")
            logger.info(f"Privacy Level: {utility_impact['privacy_level']}")
            logger.info(f"Expected Utility Impact: {utility_impact['utility_impact']}")
            logger.info(f"Communication Impact: {utility_impact['communication']}")
            logger.info("DP Mode: Central (server-side noise)")

            # Print initial summary
            logger.info("=== DP-Enhanced MNIST Federated Learning Setup ===")
            logger.info("Dataset: MNIST")
            logger.info(f"Partitioning: {config.partition_strategy}")
            logger.info(f"Clients: {config.num_actors}")
            logger.info(f"Aggregation strategy: {config.aggregation.strategy_type}")
            logger.info(f"Topology: {config.topology.topology_type}")
            logger.info(f"Privacy: Central DP (ε={args.epsilon}, δ={args.delta})")
            logger.info(f"Clipping norm: {args.clipping_norm}")
            logger.info(f"Client sampling rate: {args.client_sampling_rate}")
            logger.info(f"Rounds: {config.rounds}")  # From config, not hardcoded
            logger.info(f"Local epochs: {config.epochs}")  # From config, not hardcoded
            logger.info(f"Batch size: {config.batch_size}")
            logger.info(f"Learning rate: {config.learning_rate}")
            logger.info(f"Test split: {config.test_split}")
            logger.info(f"Resource monitoring: {config.monitor_resources}")
            logger.info(f"Health check interval: {config.health_check_interval} rounds")

            logger.info("=== Starting DP-Enhanced MNIST Federated Learning ===")

            # Monitor initial resource usage if enabled
            if args.monitor_resources:
                initial_resources = learning_process.monitor_resource_usage()
                logger.info(
                    f"Initial resource usage: {initial_resources.get('resource_utilization', {})}"
                )

            # Execute training
            results = learning_process.execute()

            # Perform periodic health checks and resource monitoring during training
            if args.monitor_resources:
                final_resources = learning_process.monitor_resource_usage()
                logger.info(
                    f"Final resource usage: {final_resources.get('resource_utilization', {})}"
                )

            # Get final health status
            health_status = learning_process.get_actor_health_status()
            if "error" not in health_status:
                logger.info(
                    f"Final actor health: {health_status['healthy']}/{health_status['sampled_actors']} healthy"
                )
                if health_status.get("degraded", 0) > 0:
                    logger.warning(f"Degraded actors: {health_status['degraded']}")
                if health_status.get("error", 0) > 0:
                    logger.error(f"Error actors: {health_status['error']}")

            # Generate visualizations if requested
            if visualizer and (
                args.create_animation or args.create_frames or args.create_summary
            ):
                logger.info("=== Generating Visualizations ===")

                if args.create_animation:
                    logger.info("Creating animation...")
                    visualizer.render_training_animation(
                        filename=f"dp_mnist_{args.topology}_{args.aggregation_strategy}_animation.mp4",
                        fps=args.fps,
                    )

                if args.create_frames:
                    logger.info("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"dp_mnist_{args.topology}_{args.aggregation_strategy}_step"
                    )

                if args.create_summary:
                    logger.info("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"dp_mnist_{args.topology}_{args.aggregation_strategy}_summary.png"
                    )

            # Save the final model
            logger.info("=== Saving Final Model ===")
            global_model.save(args.save_path)
            logger.info(f"DP-enhanced MNIST model saved to '{args.save_path}'")

            # Print final results with enhanced cluster context
            logger.info("=== DP-Enhanced MNIST Training Results ===")
            logger.info(
                f"Cluster type: {cluster_summary.get('cluster_type', 'unknown')}"
            )
            logger.info(
                f"Total physical nodes: {cluster_summary.get('total_nodes', 'unknown')}"
            )
            logger.info(
                f"Total virtual actors: {cluster_summary.get('total_actors', 'unknown')}"
            )
            logger.info(f"Topology used: {cluster_summary.get('topology', 'unknown')}")
            logger.info(
                f"Initial accuracy: {results['initial_metrics']['accuracy']:.4f}"
            )
            logger.info(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")

            # Privacy summary
            if "privacy_summary" in results:
                privacy_summary = results["privacy_summary"]
                if privacy_summary["dp_enabled"]:
                    logger.info("Differential Privacy: ENABLED (Central DP)")
                    if "accountant" in privacy_summary:
                        spent = privacy_summary["accountant"]["spent"]
                        total = privacy_summary["accountant"]["total_budget"]
                        logger.info(
                            f"Privacy Spent: ε={spent['epsilon']:.4f}/{total['epsilon']:.4f}, "
                            f"δ={spent['delta']:.2e}/{total['delta']:.2e}"
                        )

                        # Calculate privacy budget utilization
                        eps_utilization = (spent["epsilon"] / total["epsilon"]) * 100
                        delta_utilization = (spent["delta"] / total["delta"]) * 100
                        logger.info(
                            f"Privacy Budget Utilization: ε={eps_utilization:.1f}%, δ={delta_utilization:.1f}%"
                        )
                else:
                    logger.info("Differential Privacy: DISABLED")

            # Log topology-specific results
            if "topology" in results:
                topology_info = results["topology"]
                logger.info(
                    f"Network adjacency: {len(topology_info.get('adjacency_list', {}))} connections"
                )

        finally:
            logger.info("=== Shutting Down Enhanced System ===")
            learning_process.shutdown()

    except Exception as e:
        logger.error(f"DP-enhanced MNIST federated learning failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Clean up
        if "learning_process" in locals():
            learning_process.shutdown()
        logger.info("Training completed and resources cleaned up")


if __name__ == "__main__":
    main()
