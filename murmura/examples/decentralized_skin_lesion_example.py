#!/usr/bin/env python3
"""
Decentralized Skin Lesion Classification - Gossip-based Learning for Medical FL

This example demonstrates decentralized federated learning for medical image classification
using skin lesion data. Unlike centralized approaches, this uses gossip protocols where
nodes communicate directly with their neighbors without a central aggregator.
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
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.model.pytorch_model import TorchModelWrapper
from murmura.models.skin_lesion_models import WideResNet
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.network_management.topology_compatibility import (
    TopologyCompatibilityManager,
)
from murmura.node.resource_config import RayClusterConfig, ResourceConfig
from murmura.orchestration.learning_process.decentralized_learning_process import (
    DecentralizedLearningProcess,
)
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.visualization.network_visualizer import NetworkVisualizer


def create_skin_lesion_preprocessor(image_size: int = 128):
    """
    Create skin lesion specific data preprocessor.

    Args:
        image_size: Target image size for preprocessing

    Returns:
        Configured preprocessor for skin lesion images
    """
    try:
        from murmura.data_processing.data_preprocessor import create_image_preprocessor

        # Skin lesion specific configuration
        return create_image_preprocessor(
            grayscale=False,  # Medical images are typically RGB
            normalize=True,  # Normalize to [0,1]
            target_size=(image_size, image_size),  # Resize for consistent input
        )
    except ImportError:
        # Generic preprocessor not available, use automatic detection
        logging.getLogger("murmura.decentralized_skin_lesion").info(
            "Using automatic data type detection"
        )
        return None


def select_device(device_arg="auto"):
    """
    Select the appropriate device based on availability.

    Args:
        device_arg: Requested device ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        device: Device to use
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("decentralized_skin_lesion.log"),
        ],
    )


def add_integer_labels_to_dataset(
    dataset: MDataset, logger: logging.Logger
) -> tuple[list[str], int, dict[str, int]]:
    """
    Add integer label column to dataset by converting string dx categories.
    Enhanced to store preprocessing metadata for multi-node compatibility.

    Args:
        dataset: The MDataset object to modify
        logger: Logger instance for logging

    Returns:
        tuple: (dx_categories, num_classes, dx_to_label_mapping)
    """
    # Get a sample from the training split to check label type
    train_split = dataset.get_split("train")
    sample_label = train_split["dx"][0]

    if isinstance(sample_label, str):
        logger.info("Converting string diagnostic categories to integer labels...")

        # Get all unique diagnostic categories across all splits
        all_dx_categories = set()

        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_dx = set(split_data["dx"])
            all_dx_categories.update(split_dx)

        # Sort for consistent ordering across all nodes/actors
        dx_categories = sorted(list(all_dx_categories))
        num_classes = len(dx_categories)

        # Create mapping from dx to integer label
        dx_to_label = {dx: i for i, dx in enumerate(dx_categories)}

        logger.info(f"Diagnostic categories ({num_classes}): {dx_categories}")
        logger.info(f"Label mapping: {dx_to_label}")

        # Define function to add label column
        def add_label_column(example):
            example["label"] = dx_to_label[example["dx"]]
            return example

        # Apply to all splits
        logger.info("Adding integer 'label' column to all dataset splits...")
        for split_name in dataset.available_splits:
            logger.debug(f"Processing split: {split_name}")
            split_data = dataset.get_split(split_name)

            # Apply the mapping
            split_data = split_data.map(add_label_column)

            # Update the dataset split
            dataset._splits[split_name] = split_data

            # Verify the mapping worked
            sample_new = split_data[0]
            logger.debug(
                f"Split {split_name}: dx='{sample_new['dx']}' -> label={sample_new['label']}"
            )

        logger.info("Successfully added integer 'label' column to all dataset splits")

        # CRITICAL: Store preprocessing information in dataset metadata for multi-node compatibility
        if (
            not hasattr(dataset, "_dataset_metadata")
            or dataset._dataset_metadata is None
        ):
            dataset._dataset_metadata = {}

        # Store the preprocessing information that remote nodes will need
        preprocessing_info = {
            "label_encoding": {
                "source_column": "dx",
                "target_column": "label",
                "mapping": dx_to_label,
            }
        }

        # Add to dataset metadata
        dataset._dataset_metadata["preprocessing_applied"] = preprocessing_info

        logger.info("Stored preprocessing metadata for multi-node compatibility")
        logger.debug(f"Preprocessing metadata: {preprocessing_info}")

        return dx_categories, num_classes, dx_to_label

    else:
        # Labels are already integers
        logger.info("Integer labels detected in dx column")

        # Get unique values to determine categories
        all_labels = set()
        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_labels = set(split_data["dx"])
            all_labels.update(split_labels)

        dx_categories = sorted(list(all_labels))
        num_classes = len(dx_categories)

        logger.info(
            f"Integer labels detected. Categories ({num_classes}): {dx_categories}"
        )

        # Add label column that's just a copy of dx
        def copy_dx_to_label(example):
            example["label"] = int(example["dx"])
            return example

        # Apply to all splits
        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_data = split_data.map(copy_dx_to_label)
            dataset._splits[split_name] = split_data

        # Create identity mapping
        dx_to_label = {str(dx): dx for dx in dx_categories}

        return [str(cat) for cat in dx_categories], num_classes, dx_to_label


def main() -> None:
    """
    Decentralized Skin Lesion Classification with Gossip-based Federated Learning
    """
    parser = argparse.ArgumentParser(
        description="Decentralized Federated Learning for Skin Cancer Classification"
    )

    # Core decentralized learning arguments
    parser.add_argument(
        "--num_actors",
        type=int,
        default=6,
        help="Number of virtual clients (hospitals)",
    )
    parser.add_argument(
        "--partition_strategy",
        choices=["dirichlet", "iid"],
        default="dirichlet",
        help="Data partitioning strategy",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Dirichlet concentration parameter (lower = more heterogeneous)",
    )
    parser.add_argument(
        "--min_partition_size",
        type=int,
        default=50,
        help="Minimum samples per partition",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--test_split", type=str, default="test", help="Test split to use"
    )
    parser.add_argument(
        "--validation_split",
        type=str,
        default="validation",
        help="Validation split to use",
    )

    # Decentralized aggregation (only decentralized-compatible strategies)
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        choices=["gossip_avg"],  # Only decentralized strategies
        default="gossip_avg",
        help="Aggregation strategy (decentralized only)",
    )
    parser.add_argument(
        "--mixing_parameter",
        type=float,
        default=0.4,
        help="Mixing parameter for gossip averaging (0.4 = slightly favor neighbors)",
    )

    # Decentralized topology arguments (only decentralized-compatible topologies)
    parser.add_argument(
        "--topology",
        type=str,
        default="ring",
        choices=["ring", "complete", "line", "custom"],
        help="Network topology (decentralized-compatible only)",
    )

    # Training arguments
    parser.add_argument(
        "--rounds", type=int, default=8, help="Number of decentralized learning rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate in WideResNet"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="decentralized_skin_cancer_model.pt",
        help="Path to save the final model",
    )

    # Skin lesion specific arguments
    parser.add_argument(
        "--image_size", type=int, default=128, help="Size to resize images to"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="marmal88/skin_cancer",
        help="Skin lesion dataset name",
    )
    parser.add_argument(
        "--widen_factor", type=int, default=8, help="WideResNet widen factor"
    )
    parser.add_argument("--depth", type=int, default=16, help="WideResNet depth")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--debug_data",
        action="store_true",
        help="Print debug information about skin lesion data format",
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
        default="murmura_decentralized_skin_lesion",
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

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.decentralized_skin_lesion")

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
        # Determine the device to use
        device = select_device(args.device)
        logger.info(f"Using {device.upper()} device for training")

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

        logger.info("=== Loading Skin Lesion Dataset ===")
        # Load skin lesion dataset for training and testing
        train_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split=args.split,
        )

        test_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split=args.test_split,
        )

        validation_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split=args.validation_split,
        )

        # Merge datasets to have all splits available
        train_dataset.merge_splits(test_dataset)
        train_dataset.merge_splits(validation_dataset)

        # Convert string labels to integers and get diagnostic categories
        logger.info("=== Processing Labels ===")
        dx_categories, num_classes, dx_to_label = add_integer_labels_to_dataset(
            train_dataset, logger
        )

        # Debug skin lesion data format if requested
        if args.debug_data:
            logger.info("=== Debugging Skin Lesion Data Format ===")
            try:
                split_dataset = train_dataset.get_split(args.split)
                feature_data = split_dataset["image"]

                logger.info(f"Dataset: {args.dataset_name}")
                logger.info(f"Split: {args.split}")
                logger.info(f"Number of samples: {len(feature_data)}")
                logger.info(f"Diagnostic categories: {dx_categories}")
                logger.info(f"Label mapping: {dx_to_label}")

                if len(feature_data) > 0:
                    sample = feature_data[0]
                    logger.info(f"Sample type: {type(sample)}")
                    logger.info(f"Sample shape: {getattr(sample, 'shape', 'N/A')}")
                    logger.info(f"Sample mode: {getattr(sample, 'mode', 'N/A')}")
                    if hasattr(sample, "size"):
                        logger.info(f"Sample size: {sample.size}")

                    # Check label conversion
                    sample_full = split_dataset[0]
                    logger.info(
                        f"Sample dx: '{sample_full['dx']}' -> label: {sample_full['label']}"
                    )

            except Exception as e:
                logger.error(f"Error debugging skin lesion data format: {e}")

        # Create configuration with proper label column for decentralized learning
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
            dataset_name=args.dataset_name,
            ray_cluster=ray_cluster_config,
            resources=resource_config,
            # Skin lesion dataset-specific column configuration
            feature_columns=["image"],  # Skin lesion images are in 'image' column
            label_column="label",  # Use the integer label column we created
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

        logger.info("=== Creating Data Partitions ===")
        # Create partitioner
        partitioner = PartitionerFactory.create(config)

        logger.info("=== Creating WideResNet Model ===")
        # Create the WideResNet model for skin lesion classification
        model = WideResNet(
            depth=args.depth,
            num_classes=num_classes,
            widen_factor=args.widen_factor,
            drop_rate=args.dropout,
            use_dp_compatible_norm=True,  # Use GroupNorm for better compatibility
        )

        logger.info(
            f"Created WideResNet: depth={args.depth}, widen_factor={args.widen_factor}, "
            f"num_classes={num_classes}, dropout={args.dropout}"
        )

        # Create skin lesion specific data preprocessor
        skin_lesion_preprocessor = create_skin_lesion_preprocessor(args.image_size)

        # Create model wrapper with skin lesion specific configuration
        input_shape = (3, args.image_size, args.image_size)  # RGB images
        global_model = TorchModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
            input_shape=input_shape,
            device=device,
            data_preprocessor=skin_lesion_preprocessor,
        )

        logger.info("=== Setting Up Decentralized Learning Process ===")
        # Create decentralized learning process
        learning_process = DecentralizedLearningProcess(
            config=config,
            dataset=train_dataset,
            model=global_model,
        )

        # Set up visualization if requested
        visualizer = None
        if args.create_animation or args.create_frames or args.create_summary:
            logger.info("=== Setting Up Visualization ===")
            vis_dir = os.path.join(
                args.vis_dir,
                f"decentralized_skin_cancer_{args.topology}_{args.aggregation_strategy}",
            )
            os.makedirs(vis_dir, exist_ok=True)

            visualizer = NetworkVisualizer(output_dir=vis_dir)
            learning_process.register_observer(visualizer)
            logger.info("Registered visualizer with learning process")

        try:
            # Initialize the decentralized learning process
            logger.info("=== Initializing Decentralized Learning Process ===")
            learning_process.initialize(
                num_actors=config.num_actors,
                topology_config=config.topology,
                aggregation_config=config.aggregation,
                partitioner=partitioner,
            )

            # Monitor dataset distribution status
            dist_status = learning_process.get_dataset_distribution_status()
            logger.info(f"Dataset distribution: {dist_status['distribution_strategy']}")
            logger.info(
                f"Hospitals ready: {dist_status['healthy_actors']}/{dist_status['total_actors']}"
            )
            if dist_status["distribution_strategy"] == "lazy":
                logger.info(
                    f"Lazy loading enabled on {dist_status['lazy_loading_actors']} hospitals"
                )

            # Monitor memory usage
            if args.monitor_resources:
                memory_status = learning_process.monitor_memory_usage()
                logger.info(
                    f"Cluster memory: {memory_status.get('available_memory_gb', 'N/A')}GB available"
                )
                if memory_status.get("high_usage_nodes", 0) > 0:
                    logger.warning(
                        f"{memory_status['high_usage_nodes']} nodes have high memory usage"
                    )

            # Get and log cluster information
            cluster_summary = learning_process.get_cluster_summary()
            logger.info("=== Decentralized Cluster Summary ===")
            logger.info(
                f"Cluster type: {cluster_summary.get('cluster_type', 'unknown')}"
            )
            logger.info(f"Total nodes: {cluster_summary.get('total_nodes', 'unknown')}")
            logger.info(
                f"Total hospitals: {cluster_summary.get('total_actors', 'unknown')}"
            )

            # Print initial summary
            logger.info("=== Decentralized Skin Lesion Learning Setup ===")
            logger.info(f"Dataset: {args.dataset_name}")
            logger.info(f"Partitioning: {config.partition_strategy} (α={args.alpha})")
            logger.info(f"Hospitals: {config.num_actors}")
            logger.info(
                f"Aggregation: {config.aggregation.strategy_type} (decentralized)"
            )
            logger.info(f"Topology: {config.topology.topology_type}")
            logger.info(f"Mixing parameter: {args.mixing_parameter}")
            logger.info(f"Rounds: {config.rounds}")
            logger.info(f"Local epochs: {config.epochs}")
            logger.info(f"Batch size: {config.batch_size}")
            logger.info(f"Learning rate: {config.learning_rate}")
            logger.info(f"Weight decay: {args.weight_decay}")
            logger.info(f"Device: {device}")
            logger.info(f"Image size: {args.image_size}")
            logger.info(f"Classes ({num_classes}): {dx_categories}")
            logger.info(f"Using feature column: {config.feature_columns}")
            logger.info(f"Using label column: {config.label_column}")
            logger.info(f"Test split: {config.test_split}")
            logger.info(f"Resource monitoring: {config.monitor_resources}")
            logger.info(f"Health check interval: {config.health_check_interval} rounds")

            logger.info("=== Starting Decentralized Skin Lesion Learning ===")
            logger.info(
                "Note: In decentralized learning, hospitals communicate directly with neighbors"
            )
            logger.info(
                "No central server coordinates the process - pure peer-to-peer learning"
            )

            # Execute the decentralized learning process
            results = learning_process.execute()

            # Generate visualizations if requested
            if visualizer and (
                args.create_animation or args.create_frames or args.create_summary
            ):
                logger.info("=== Generating Visualizations ===")

                if args.create_animation:
                    logger.info("Creating animation...")
                    visualizer.render_training_animation(
                        filename=f"decentralized_skin_cancer_{args.topology}_{args.aggregation_strategy}_animation.mp4",
                        fps=args.fps,
                    )

                if args.create_frames:
                    logger.info("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"decentralized_skin_cancer_{args.topology}_{args.aggregation_strategy}_step"
                    )

                if args.create_summary:
                    logger.info("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"decentralized_skin_cancer_{args.topology}_{args.aggregation_strategy}_summary.png"
                    )

            # Save the final model with comprehensive metadata
            logger.info("=== Saving Final Model ===")
            save_path = args.save_path

            # Create comprehensive checkpoint for decentralized medical model
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": global_model.optimizer.state_dict(),
                "dx_categories": dx_categories,
                "dx_to_label_mapping": dx_to_label,
                "num_classes": num_classes,
                "depth": args.depth,
                "widen_factor": args.widen_factor,
                "dropout": args.dropout,
                "image_size": args.image_size,
                "input_shape": input_shape,
                "device": device,
                "learning_type": "decentralized",
                "topology": args.topology,
                "aggregation_strategy": args.aggregation_strategy,
                "mixing_parameter": args.mixing_parameter,
                "config": {
                    k: v for k, v in vars(args).items() if not k.startswith("_")
                },
            }

            os.makedirs(
                os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True
            )
            torch.save(checkpoint, save_path)
            logger.info(f"Decentralized skin lesion model saved to '{save_path}'")

            # Print final results
            logger.info("=== Decentralized Skin Lesion Training Results ===")
            logger.info(
                f"Initial accuracy: {results['initial_metrics']['accuracy']:.4f}"
            )
            logger.info(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")
            logger.info(f"Training device: {device}")
            logger.info(f"Diagnostic categories used: {dx_categories}")
            logger.info(f"Label mapping: {dx_to_label}")
            logger.info(
                f"Training completed with {config.rounds} rounds of {config.epochs} epochs each"
            )

            # Log topology-specific results
            if "topology" in results:
                topology_info = results["topology"]
                logger.info(
                    f"Network connections: {len(topology_info.get('adjacency_list', {}))}"
                )
                logger.info(
                    f"Decentralized topology: {topology_info.get('type', 'unknown')}"
                )

            # Display detailed metrics if available
            if "round_metrics" in results:
                logger.info("=== Detailed Round Metrics ===")
                for round_data in results["round_metrics"]:
                    round_num = round_data["round"]
                    logger.info(f"Round {round_num}:")
                    logger.info(
                        f"  Train Loss: {round_data.get('train_loss', 'N/A'):.4f}"
                    )
                    logger.info(
                        f"  Train Accuracy: {round_data.get('train_accuracy', 'N/A'):.4f}"
                    )
                    logger.info(
                        f"  Test Loss: {round_data.get('test_loss', 'N/A'):.4f}"
                    )
                    logger.info(
                        f"  Test Accuracy: {round_data.get('test_accuracy', 'N/A'):.4f}"
                    )

            # Decentralized learning summary
            logger.info("=== Decentralized Learning Summary ===")
            logger.info("Learning paradigm: Fully decentralized (no central server)")
            logger.info(
                f"Communication pattern: {topology_info.get('type', 'peer-to-peer')}"
            )
            logger.info("Privacy: Each hospital's data never leaves their premises")
            logger.info("Robustness: No single point of failure")
            logger.info("Scalability: Can add hospitals without affecting others")

        finally:
            logger.info("=== Shutting Down ===")
            learning_process.shutdown()

    except Exception as e:
        logger.error(f"Decentralized Skin Lesion Learning Process failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
