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
from murmura.orchestration.learning_process.decentralized_learning_process import (
    DecentralizedLearningProcess,
)
from murmura.network_management.topology_compatibility import (
    TopologyCompatibilityManager,
)
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.visualization.network_visualizer import NetworkVisualizer

# Import DP components
from murmura.privacy.dp_config import DPConfig
from murmura.privacy.dp_model_wrapper import DPTorchModelWrapper
from murmura.privacy.privacy_accountant import PrivacyAccountant


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
        logging.getLogger("murmura.dp_decentralized_mnist_example").info(
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
            logging.FileHandler("dp_decentralized_mnist.log"),
        ],
    )


def main() -> None:
    """
    MNIST Decentralized Learning with Differential Privacy
    """
    parser = argparse.ArgumentParser(
        description="MNIST Decentralized Learning with Differential Privacy"
    )

    # Core learning arguments
    parser.add_argument(
        "--num_actors", type=int, default=10, help="Total number of virtual clients"
    )
    parser.add_argument(
        "--partition_strategy",
        choices=["dirichlet", "iid", "sensitive_groups", "topology_correlated", "imbalanced_sensitive"],
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
        "--rounds", type=int, default=10, help="Number of learning rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )

    # Differential Privacy arguments
    parser.add_argument(
        "--enable_dp", action="store_true", help="Enable differential privacy"
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=8.0,
        help="Target privacy budget (epsilon)",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5,
        help="Target privacy parameter (delta)",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="Noise multiplier (auto if None)",
    )
    parser.add_argument(
        "--enable_client_dp",
        action="store_true",
        default=True,
        help="Enable client-level DP",
    )
    parser.add_argument(
        "--enable_central_dp", action="store_true", help="Enable central aggregation DP"
    )
    parser.add_argument(
        "--dp_preset",
        choices=["high_privacy", "medium_privacy", "low_privacy", "custom"],
        default="medium_privacy",
        help="DP preset configuration",
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
        default="murmura_dp_decentralized",
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
    parser.add_argument(
        "--save_path",
        type=str,
        default="dp_mnist_decentralized_model.pt",
        help="Path to save the final model",
    )

    # Subsampling arguments for privacy amplification
    parser.add_argument(
        "--client_sampling_rate",
        type=float,
        default=1.0,
        help="Fraction of clients to sample per round (for privacy amplification)",
    )
    parser.add_argument(
        "--data_sampling_rate",
        type=float,
        default=1.0,
        help="Fraction of local data to sample per client (for privacy amplification)",
    )
    parser.add_argument(
        "--enable_subsampling_amplification",
        action="store_true",
        help="Enable privacy amplification by subsampling",
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
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment name for visualization directory (overrides default naming)",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.dp_decentralized_mnist_example")

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
        # Select device
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        logger.info(f"Using {device.upper()} device for training")

        # Create DP configuration
        dp_config = None
        if args.enable_dp:
            logger.info("=== Configuring Differential Privacy ===")

            if args.dp_preset == "high_privacy":
                dp_config = DPConfig.create_high_privacy()
                # Override with user-specified epsilon if provided and different from default
                if args.target_epsilon != 8.0:  # 8.0 is the default
                    dp_config.target_epsilon = args.target_epsilon
            elif args.dp_preset == "medium_privacy":
                dp_config = DPConfig.create_for_mnist()
                # Override with user-specified epsilon if provided and different from default
                if args.target_epsilon != 8.0:  # 8.0 is the default
                    dp_config.target_epsilon = args.target_epsilon
            elif args.dp_preset == "low_privacy":
                dp_config = DPConfig(
                    target_epsilon=args.target_epsilon
                    if args.target_epsilon != 8.0
                    else 16.0,
                    target_delta=1e-4,
                    max_grad_norm=2.0,
                    enable_client_dp=True,
                    enable_central_dp=False,
                )
            else:  # custom
                dp_config = DPConfig(
                    target_epsilon=args.target_epsilon,
                    target_delta=args.target_delta,
                    max_grad_norm=args.max_grad_norm,
                    noise_multiplier=args.noise_multiplier,
                    enable_client_dp=args.enable_client_dp,
                    enable_central_dp=args.enable_central_dp,
                )

            logger.info(
                f"DP Configuration: ε={dp_config.target_epsilon}, δ={dp_config.target_delta}"
            )
            logger.info(f"Max grad norm: {dp_config.max_grad_norm}")
            logger.info(
                f"Client DP: {dp_config.enable_client_dp}, Central DP: {dp_config.enable_central_dp}"
            )

            # Update DP config with subsampling parameters if enabled
            if args.enable_subsampling_amplification:
                dp_config.client_sampling_rate = args.client_sampling_rate
                dp_config.data_sampling_rate = args.data_sampling_rate
                dp_config.use_amplification_by_subsampling = True

                logger.info("=== Subsampling Amplification Enabled ===")
                logger.info(f"Client sampling rate: {args.client_sampling_rate}")
                logger.info(f"Data sampling rate: {args.data_sampling_rate}")
                amplification_factor = dp_config.get_amplification_factor()
                logger.info(f"Privacy amplification factor: {amplification_factor:.3f}")

            # Initialize privacy accountant
            privacy_accountant = PrivacyAccountant(dp_config)
        else:
            logger.info("Differential privacy is DISABLED")

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
            dataset_name="ylecun/mnist",
            ray_cluster=ray_cluster_config,
            resources=resource_config,
            feature_columns=["image"],
            label_column="label",
            rounds=args.rounds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            test_split=args.test_split,
            monitor_resources=args.monitor_resources,
            health_check_interval=args.health_check_interval,
            client_sampling_rate=args.client_sampling_rate,
            data_sampling_rate=args.data_sampling_rate,
            enable_subsampling_amplification=args.enable_subsampling_amplification,
        )

        logger.info("=== Loading MNIST Dataset ===")
        # Load MNIST Dataset for training and testing
        train_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="ylecun/mnist",
            split=config.split,
        )

        test_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="ylecun/mnist",
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
        model = MNISTModel(
            use_dp_compatible_norm=True
        )  # Use GroupNorm for DP compatibility
        input_shape = (1, 28, 28)  # MNIST: 1 channel, 28x28 pixels

        # Create MNIST-specific data preprocessor
        mnist_preprocessor = create_mnist_preprocessor()

        # Create model wrapper (DP or regular)
        if args.enable_dp and dp_config:
            logger.info("Creating DP-aware model wrapper")
            from typing import Union
            from murmura.model.pytorch_model import TorchModelWrapper

            global_model: Union[DPTorchModelWrapper, TorchModelWrapper] = (
                DPTorchModelWrapper(
                    model=model,
                    dp_config=dp_config,
                    loss_fn=nn.CrossEntropyLoss(),
                    optimizer_class=torch.optim.SGD,  # SGD works better with DP
                    optimizer_kwargs={"lr": args.lr, "momentum": 0.9},
                    input_shape=input_shape,
                    device=device,
                    data_preprocessor=mnist_preprocessor,
                )
            )
        else:
            logger.info("Creating regular model wrapper")
            from murmura.model.pytorch_model import TorchModelWrapper

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
            if args.experiment_name:
                vis_dir = os.path.join(args.vis_dir, args.experiment_name)
            else:
                vis_dir = os.path.join(
                    args.vis_dir,
                    f"dp_decentralized_mnist_{args.topology}_{args.aggregation_strategy}"
                    + ("_dp" if args.enable_dp else "_no_dp"),
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

            # Print experiment summary
            logger.info("=== MNIST DP Decentralized Learning Setup ===")
            logger.info("Dataset: MNIST")
            logger.info(f"Clients: {config.num_actors}")
            logger.info(f"Partitioning: {config.partition_strategy} (α={args.alpha})")
            logger.info(f"Aggregation strategy: {config.aggregation.strategy_type}")
            logger.info(f"Topology: {config.topology.topology_type}")
            logger.info(f"Rounds: {config.rounds}")
            logger.info(f"Local epochs: {config.epochs}")
            logger.info(f"Batch size: {config.batch_size}")
            logger.info(f"Learning rate: {config.learning_rate}")
            logger.info(f"Device: {device}")
            logger.info(f"Test split: {config.test_split}")
            logger.info(f"Resource monitoring: {config.monitor_resources}")
            logger.info(f"Health check interval: {config.health_check_interval} rounds")

            if args.enable_dp and dp_config is not None:
                logger.info("=== Differential Privacy Settings ===")
                logger.info(
                    f"Privacy budget: ε={dp_config.target_epsilon}, δ={dp_config.target_delta}"
                )
                logger.info(f"Max gradient norm: {dp_config.max_grad_norm}")
                logger.info(f"Client DP: {dp_config.enable_client_dp}")
                logger.info(f"Central DP: {dp_config.enable_central_dp}")
                logger.info(f"Mechanism: {dp_config.mechanism.value}")
                logger.info(f"Accounting: {dp_config.accounting_method.value}")

                # Suggest optimal noise if auto-tuning
                if dp_config.auto_tune_noise and dp_config.noise_multiplier is None:
                    sample_rate = (
                        args.batch_size / 60000
                    )  # MNIST has ~60k training samples
                    suggested_noise = privacy_accountant.suggest_optimal_noise(
                        sample_rate=sample_rate,
                        epochs=args.epochs
                        * args.rounds,  # Total epochs across all rounds
                        dataset_size=60000,
                    )
                    logger.info(f"Suggested noise multiplier: {suggested_noise:.3f}")
            else:
                logger.info("Differential Privacy: DISABLED")

            logger.info("=== Starting MNIST DP Decentralized Learning ===")

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

            # Display results
            logger.info("=== Training Results ===")
            logger.info(
                f"Initial accuracy: {results['initial_metrics']['accuracy']:.4f}"
            )
            logger.info(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")

            # Display privacy results if DP was enabled
            privacy_spent = None
            if (
                args.enable_dp
                and dp_config is not None
                and hasattr(global_model, "get_privacy_spent")
            ):
                logger.info("=== Privacy Results ===")
                privacy_spent = global_model.get_privacy_spent()
                logger.info(
                    f"Privacy spent: ε={privacy_spent['epsilon']:.3f}, δ={privacy_spent['delta']:.2e}"
                )
                logger.info(
                    f"Privacy budget: ε={dp_config.target_epsilon}, δ={dp_config.target_delta}"
                )

                remaining_eps = dp_config.target_epsilon - privacy_spent["epsilon"]
                logger.info(f"Remaining budget: ε={remaining_eps:.3f}")

                if privacy_spent["epsilon"] > dp_config.target_epsilon:
                    logger.warning("Privacy budget exceeded!")
                else:
                    logger.info("Privacy budget respected ✓")

                # Get privacy summary from accountant
                if "privacy_accountant" in locals():
                    privacy_summary = privacy_accountant.get_privacy_summary()
                    logger.info(
                        f"Global privacy utilization: {privacy_summary['global_privacy']['utilization_percentage']:.1f}%"
                    )

            # Generate visualizations if requested
            if visualizer and (
                args.create_animation or args.create_frames or args.create_summary
            ):
                logger.info("=== Generating Visualizations ===")

                if args.create_animation:
                    logger.info("Creating animation...")
                    visualizer.render_training_animation(
                        filename=f"dp_decentralized_mnist_{args.topology}_{args.aggregation_strategy}_animation.mp4",
                        fps=args.fps,
                    )

                if args.create_frames:
                    logger.info("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"dp_decentralized_mnist_{args.topology}_{args.aggregation_strategy}_step"
                    )

                if args.create_summary:
                    logger.info("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"dp_decentralized_mnist_{args.topology}_{args.aggregation_strategy}_summary.png"
                    )

            # Save the final model
            logger.info("=== Saving Final Model ===")
            save_path = args.save_path
            if args.enable_dp:
                # Add DP suffix to filename
                name, ext = os.path.splitext(save_path)
                save_path = f"{name}_dp{ext}"

            # Create comprehensive checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": global_model.optimizer.state_dict(),
                "config": {
                    k: v for k, v in vars(args).items() if not k.startswith("_")
                },
                "results": results,
                "differential_privacy": {
                    "enabled": args.enable_dp,
                    "config": dp_config.model_dump() if dp_config is not None else None,
                    "privacy_spent": privacy_spent,
                },
            }

            os.makedirs(
                os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True
            )
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to '{save_path}'")

            # Print final results with enhanced cluster context
            logger.info("=== MNIST DP Decentralized Training Results ===")
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
                f"Training completed with {config.rounds} rounds of {config.epochs} epochs each"
            )

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
        logger.error(f"MNIST DP Decentralized Learning Process failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
