import argparse
import os
import logging
import torch
import torch.nn as nn

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.node.resource_config import RayClusterConfig, ResourceConfig
from murmura.orchestration.learning_process.decentralized_learning_process import (
    DecentralizedLearningProcess,
)
from murmura.network_management.topology_compatibility import (
    TopologyCompatibilityManager,
)
from murmura.orchestration.orchestration_config import OrchestrationConfig
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
        logging.getLogger("murmura.decentralized_mnist_example").info(
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
            logging.FileHandler("decentralized_mnist.log"),
        ],
    )


def main() -> None:
    """
    MNIST Decentralized Learning with Multi-Node Support
    """
    parser = argparse.ArgumentParser(
        description="MNIST Decentralized Learning with Multi-Node Support"
    )

    # Core learning arguments
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
        choices=["gossip_avg"],  # Only decentralized strategies
        default="gossip_avg",
        help="Aggregation strategy to use (only decentralized strategies)",
    )
    parser.add_argument(
        "--mixing_parameter",
        type=float,
        default=0.5,
        help="Mixing parameter for gossip_avg strategy (0.5 = equal mixing)",
    )

    # Topology arguments (only decentralized-compatible topologies)
    parser.add_argument(
        "--topology",
        type=str,
        default="ring",  # Default to ring for decentralized learning
        choices=["ring", "complete", "line", "custom"],
        help="Network topology between clients (decentralized-compatible only)",
    )

    # Training arguments
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of learning rounds"
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
        default="mnist_decentralized_model.pt",
        help="Path to save the final model",
    )

    # Multi-node Ray cluster arguments
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help="Ray cluster address. If None, uses local cluster.",
    )
    parser.add_argument(
        "--ray_namespace",
        type=str,
        default="murmura_decentralized",
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

    # MNIST-specific arguments
    parser.add_argument(
        "--debug_data",
        action="store_true",
        help="Print debug information about MNIST data format",
    )

    # Logging and monitoring arguments
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

    # Visualization arguments
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

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.decentralized_mnist_example")

    # Check compatibility of topology and strategy before proceeding
    topology_type = TopologyType(args.topology)
    strategy_type = AggregationStrategyType(args.aggregation_strategy)

    # Validate decentralized compatibility
    from murmura.aggregation.strategies.gossip_avg import GossipAvg

    if not TopologyCompatibilityManager.is_compatible(GossipAvg, topology_type):
        compatible_topologies = TopologyCompatibilityManager.get_compatible_topologies(
            GossipAvg
        )
        logger.error(
            f"Strategy {args.aggregation_strategy} is not compatible with topology {args.topology}."
        )
        logger.error(
            f"Compatible topologies: {[t.value for t in compatible_topologies]}"
        )
        return

    try:
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

        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,
            topology=TopologyConfig(
                topology_type=topology_type,
                hub_index=0,  # Not used for decentralized topologies
            ),
            aggregation=AggregationConfig(
                strategy_type=strategy_type,
                params={"mixing_parameter": args.mixing_parameter},
            ),
            dataset_name="mnist",  # Fixed to MNIST
            ray_cluster=ray_cluster_config,
            resources=resource_config,
            feature_columns=["image"],
            label_column="label",
        )

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

        logger.info("=== Creating Data Partitions ===")
        # Create partitioner
        partitioner = PartitionerFactory.create(config)

        logger.info("=== Creating MNIST Model ===")
        # Create the MNIST model
        model = MNISTModel()
        input_shape = (1, 28, 28)  # MNIST: 1 channel, 28x28 pixels

        # Create MNIST-specific data preprocessor
        mnist_preprocessor = create_mnist_preprocessor()

        # Create model wrapper with MNIST-specific configuration
        global_model = TorchModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": args.lr},
            input_shape=input_shape,
            data_preprocessor=mnist_preprocessor,
        )

        logger.info("=== Setting Up Decentralized Learning Process ===")
        # Create learning process with enhanced config
        learning_process = DecentralizedLearningProcess(
            config=config,
            dataset=train_dataset,
            model=global_model,
        )

        # Set up visualization BEFORE executing the learning process
        visualizer = None
        if args.create_animation or args.create_frames or args.create_summary:
            logger.info("=== Setting Up Visualization ===")
            # Create visualization directory
            vis_dir = os.path.join(
                args.vis_dir,
                f"decentralized_mnist_{args.topology}_{args.aggregation_strategy}",
            )
            os.makedirs(vis_dir, exist_ok=True)

            # Create visualizer
            visualizer = NetworkVisualizer(output_dir=vis_dir)

            # Register visualizer with learning process
            learning_process.register_observer(visualizer)
            logger.info("Registered visualizer with learning process")

        try:
            # Initialize the learning process
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

            # Print initial summary
            logger.info("=== MNIST Decentralized Learning Setup ===")
            logger.info("Dataset: MNIST")
            logger.info(f"Partitioning: {config.partition_strategy}")
            logger.info(f"Clients: {config.num_actors}")
            logger.info(f"Aggregation strategy: {config.aggregation.strategy_type}")
            logger.info(f"Topology: {config.topology.topology_type}")
            logger.info(f"Rounds: {args.rounds}")
            logger.info(f"Local epochs: {args.epochs}")
            logger.info(f"Batch size: {args.batch_size}")
            logger.info(f"Learning rate: {args.lr}")
            logger.info(f"Mixing parameter: {args.mixing_parameter}")

            logger.info("=== Starting MNIST Decentralized Learning ===")

            # Monitor initial resource usage if enabled
            if args.monitor_resources:
                initial_resources = learning_process.monitor_resource_usage()
                logger.info(
                    f"Initial resource usage: {initial_resources.get('resource_utilization', {})}"
                )

            # Execute the learning process with enhanced monitoring
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
                        filename=f"decentralized_mnist_{args.topology}_{args.aggregation_strategy}_animation.mp4",
                        fps=args.fps,
                    )

                if args.create_frames:
                    logger.info("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"decentralized_mnist_{args.topology}_{args.aggregation_strategy}_step"
                    )

                if args.create_summary:
                    logger.info("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"decentralized_mnist_{args.topology}_{args.aggregation_strategy}_summary.png"
                    )

            # Save the final model
            logger.info("=== Saving Final Model ===")
            save_path = args.save_path
            global_model.save(save_path)
            logger.info(f"MNIST model saved to '{save_path}'")

            # Print final results with enhanced cluster context
            logger.info("=== MNIST Decentralized Training Results ===")
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
        logger.error(f"MNIST Decentralized Learning Process failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
