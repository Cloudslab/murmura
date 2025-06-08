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
from murmura.models.skin_lesion_models import WideResNet
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.node.resource_config import RayClusterConfig, ResourceConfig
from murmura.orchestration.learning_process.federated_learning_process import (
    FederatedLearningProcess,
)
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.visualization.network_visualizer import NetworkVisualizer

# Import DP components
from murmura.privacy.dp_config import DPConfig
from murmura.privacy.dp_model_wrapper import DPTorchModelWrapper
from murmura.privacy.privacy_accountant import PrivacyAccountant
from murmura.model.pytorch_model import TorchModelWrapper
from murmura.data_processing.data_preprocessor import create_image_preprocessor
from typing import Union


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dp_skin_lesion_federated.log"),
        ],
    )


def create_skin_lesion_preprocessor(image_size: int = 128):
    """
    Create image preprocessor for skin lesion datasets.
    
    Args:
        image_size: Target image size (will be resized to image_size x image_size)
        
    Returns:
        GenericDataPreprocessor configured for skin lesion images
    """
    return create_image_preprocessor(
        grayscale=False,  # Keep RGB for medical images
        normalize=True,   # Normalize to [0,1]
        target_size=(image_size, image_size)  # Resize to specified size
    )


def add_integer_labels_to_dataset(
    dataset: MDataset, logger: logging.Logger
) -> tuple[list[str], int, dict[str, int]]:
    """Add integer label column to dataset by converting string dx categories."""
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

        # Add label column
        def add_label_column(example):
            example["label"] = dx_to_label[example["dx"]]
            return example

        # Apply to all splits
        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_data = split_data.map(add_label_column)
            dataset._splits[split_name] = split_data

        # Store preprocessing metadata
        if (
            not hasattr(dataset, "_dataset_metadata")
            or dataset._dataset_metadata is None
        ):
            dataset._dataset_metadata = {}

        dataset._dataset_metadata["preprocessing_applied"] = {
            "label_encoding": {
                "source_column": "dx",
                "target_column": "label",
                "mapping": dx_to_label,
            }
        }

        return dx_categories, num_classes, dx_to_label

    else:
        # Labels are already integers
        logger.info("Integer labels detected in dx column")
        all_labels = set()
        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_labels = set(split_data["dx"])
            all_labels.update(split_labels)

        dx_categories = sorted(list(all_labels))
        num_classes = len(dx_categories)

        # Add label column that's just a copy of dx
        def copy_dx_to_label(example):
            example["label"] = int(example["dx"])
            return example

        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_data = split_data.map(copy_dx_to_label)
            dataset._splits[split_name] = split_data

        dx_to_label = {str(dx): dx for dx in dx_categories}
        return [str(cat) for cat in dx_categories], num_classes, dx_to_label


def main() -> None:
    """
    Skin Lesion Classification Federated Learning with Differential Privacy
    """
    parser = argparse.ArgumentParser(
        description="Federated Learning for Skin Cancer Classification with DP"
    )

    # Core federated learning arguments
    parser.add_argument(
        "--num_actors", type=int, default=5, help="Number of virtual clients"
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
        default=50,
        help="Minimum samples per partition",
    )
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        choices=["fedavg", "trimmed_mean"],
        default="fedavg",
        help="Aggregation strategy to use",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="star",
        choices=["star", "ring", "complete", "line"],
        help="Network topology between clients",
    )

    # Training arguments
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # Model arguments
    parser.add_argument(
        "--widen_factor",
        type=int,
        default=4,
        help="WideResNet widen factor (reduced for DP)",
    )
    parser.add_argument("--depth", type=int, default=16, help="WideResNet depth")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--image_size", type=int, default=128, help="Image size")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="marmal88/skin_cancer",
        help="Skin lesion dataset name",
    )

    # Differential Privacy arguments
    parser.add_argument(
        "--enable_dp", action="store_true", help="Enable differential privacy"
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=10.0,
        help="Target privacy budget (epsilon)",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-4,
        help="Target privacy parameter (delta)",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.2, help="Gradient clipping norm"
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

    # System arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="dp_skin_lesion_federated_model.pt",
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

    # Visualization
    parser.add_argument(
        "--create_summary",
        action="store_true",
        help="Create summary plot of the training process",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations",
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
    logger = logging.getLogger("murmura.dp_skin_lesion_example")

    try:
        # Select device
        if args.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = args.device
        logger.info(f"Using {device.upper()} device for training")

        # Create DP configuration
        dp_config = None
        if args.enable_dp:
            logger.info("=== Configuring Differential Privacy ===")

            if args.dp_preset == "high_privacy":
                dp_config = DPConfig(
                    target_epsilon=args.target_epsilon
                    if args.target_epsilon != 8.0
                    else 3.0,  # Very private for medical data
                    target_delta=1e-5,
                    max_grad_norm=0.8,
                    enable_client_dp=True,
                    enable_central_dp=False,
                )
            elif args.dp_preset == "medium_privacy":
                dp_config = DPConfig.create_for_skin_lesion()
                # Override with user-specified epsilon if provided and different from default
                if args.target_epsilon != 8.0:  # 8.0 is the default
                    dp_config.target_epsilon = args.target_epsilon
            elif args.dp_preset == "low_privacy":
                dp_config = DPConfig(
                    target_epsilon=args.target_epsilon
                    if args.target_epsilon != 8.0
                    else 20.0,
                    target_delta=1e-3,
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

            logger.info(
                f"DP Configuration: ε={dp_config.target_epsilon}, δ={dp_config.target_delta}"
            )
            logger.info(f"Max grad norm: {dp_config.max_grad_norm}")
            logger.info(
                f"Client DP: {dp_config.enable_client_dp}, Central DP: {dp_config.enable_central_dp}"
            )

            # Initialize privacy accountant
            privacy_accountant = PrivacyAccountant(dp_config)
        else:
            logger.info("Differential privacy is DISABLED")

        # Ray cluster configuration
        ray_cluster_config = RayClusterConfig(logging_level=args.log_level)
        resource_config = ResourceConfig()

        logger.info("=== Loading Skin Lesion Dataset ===")
        # Load skin lesion dataset
        train_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split="train",
        )

        test_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split="test",
        )

        validation_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split="validation",
        )

        # Merge datasets
        train_dataset.merge_splits(test_dataset)
        train_dataset.merge_splits(validation_dataset)

        # Process labels
        logger.info("=== Processing Labels ===")
        dx_categories, num_classes, dx_to_label = add_integer_labels_to_dataset(
            train_dataset, logger
        )

        # Create configuration
        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split="train",
            topology=TopologyConfig(topology_type=TopologyType(args.topology)),
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType(args.aggregation_strategy)
            ),
            dataset_name=args.dataset_name,
            ray_cluster=ray_cluster_config,
            resources=resource_config,
            feature_columns=["image"],
            label_column="label",
            rounds=args.rounds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            test_split="test",
            client_sampling_rate=args.client_sampling_rate,
            data_sampling_rate=args.data_sampling_rate,
            enable_subsampling_amplification=args.enable_subsampling_amplification,
        )

        logger.info("=== Creating Data Partitions ===")
        partitioner = PartitionerFactory.create(config)

        logger.info("=== Creating WideResNet Model ===")
        model = WideResNet(
            depth=args.depth,
            num_classes=num_classes,
            widen_factor=args.widen_factor,
            drop_rate=args.dropout,
            use_dp_compatible_norm=True,  # Use GroupNorm for DP compatibility
        )

        logger.info(
            f"Created WideResNet: depth={args.depth}, widen_factor={args.widen_factor}, "
            f"num_classes={num_classes}, dropout={args.dropout}"
        )

        # Create skin lesion preprocessor
        logger.info("=== Creating Image Preprocessor ===")
        skin_lesion_preprocessor = create_skin_lesion_preprocessor(args.image_size)
        logger.info(f"Configured image preprocessor for {args.image_size}x{args.image_size} images")

        # Create model wrapper (DP or regular)
        input_shape = (3, args.image_size, args.image_size)
        global_model: Union[DPTorchModelWrapper, TorchModelWrapper]
        if args.enable_dp and dp_config:
            logger.info("Creating DP-aware model wrapper")
            global_model = DPTorchModelWrapper(
                model=model,
                dp_config=dp_config,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.SGD,  # SGD works better with DP
                optimizer_kwargs={
                    "lr": args.lr,
                    "momentum": 0.9,
                    "weight_decay": args.weight_decay,
                },
                input_shape=input_shape,
                device=device,
                data_preprocessor=skin_lesion_preprocessor,
            )
        else:
            logger.info("Creating regular model wrapper")

            global_model = TorchModelWrapper(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs={
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
                input_shape=input_shape,
                device=device,
                data_preprocessor=skin_lesion_preprocessor,
            )

        logger.info("=== Setting Up Learning Process ===")
        learning_process = FederatedLearningProcess(
            config=config,
            dataset=train_dataset,
            model=global_model,
        )

        # Set up visualization if requested
        visualizer = None
        if args.create_summary:
            logger.info("=== Setting Up Visualization ===")
            if args.experiment_name:
                vis_dir = os.path.join(args.vis_dir, args.experiment_name)
            else:
                vis_dir = os.path.join(
                    args.vis_dir,
                    f"dp_skin_lesion_{args.topology}_{args.aggregation_strategy}"
                    + ("_dp" if args.enable_dp else "_no_dp"),
                )
            os.makedirs(vis_dir, exist_ok=True)
            visualizer = NetworkVisualizer(output_dir=vis_dir)
            learning_process.register_observer(visualizer)

        try:
            # Initialize learning process
            logger.info("=== Initializing Learning Process ===")
            learning_process.initialize(
                num_actors=config.num_actors,
                topology_config=config.topology,
                aggregation_config=config.aggregation,
                partitioner=partitioner,
            )

            # Print experiment summary
            logger.info("=== Skin Lesion DP Federated Learning Setup ===")
            logger.info(f"Dataset: {args.dataset_name}")
            logger.info(f"Classes ({num_classes}): {dx_categories}")
            logger.info(f"Clients: {config.num_actors}")
            logger.info(f"Partitioning: {config.partition_strategy} (α={args.alpha})")
            logger.info(f"Aggregation: {config.aggregation.strategy_type}")
            logger.info(f"Topology: {config.topology.topology_type}")
            logger.info(f"Rounds: {config.rounds}")
            logger.info(f"Local epochs: {config.epochs}")
            logger.info(f"Batch size: {config.batch_size}")
            logger.info(f"Learning rate: {config.learning_rate}")
            logger.info(f"Device: {device}")
            logger.info(f"Image size: {args.image_size}")

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
                    # Estimate dataset size (skin cancer dataset is ~10k samples)
                    estimated_dataset_size = 10000
                    sample_rate = args.batch_size / estimated_dataset_size
                    suggested_noise = privacy_accountant.suggest_optimal_noise(
                        sample_rate=sample_rate,
                        epochs=args.epochs * args.rounds,
                        dataset_size=estimated_dataset_size,
                    )
                    logger.info(f"Suggested noise multiplier: {suggested_noise:.3f}")
            else:
                logger.info("Differential Privacy: DISABLED")

            logger.info("=== Starting Training ===")
            # Execute learning process
            results = learning_process.execute()

            # Display results
            logger.info("=== Training Results ===")
            logger.info(
                f"Initial accuracy: {results['initial_metrics']['accuracy']:.4f}"
            )
            logger.info(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")

            # Display privacy results if DP was enabled
            privacy_spent = None
            if args.enable_dp and hasattr(global_model, "get_privacy_spent"):
                logger.info("=== Privacy Results ===")
                privacy_spent = global_model.get_privacy_spent()
                logger.info(
                    f"Privacy spent: ε={privacy_spent['epsilon']:.3f}, δ={privacy_spent['delta']:.2e}"
                )
                if dp_config is not None:
                    logger.info(
                        f"Privacy budget: ε={dp_config.target_epsilon}, δ={dp_config.target_delta}"
                    )

                    remaining_eps = dp_config.target_epsilon - privacy_spent["epsilon"]
                    logger.info(f"Remaining budget: ε={remaining_eps:.3f}")

                    if privacy_spent["epsilon"] > dp_config.target_epsilon:
                        logger.warning("Privacy budget exceeded!")
                    else:
                        logger.info("Privacy budget respected ✓")

            # Create visualization if requested
            if visualizer and args.create_summary:
                logger.info("=== Generating Visualization ===")
                visualizer.render_summary_plot(
                    filename=f"dp_skin_lesion_{args.topology}_{args.aggregation_strategy}"
                    + ("_dp" if args.enable_dp else "_no_dp")
                    + "_summary.png"
                )

            # Save model
            logger.info("=== Saving Model ===")
            save_path = args.save_path
            if args.enable_dp:
                name, ext = os.path.splitext(save_path)
                save_path = f"{name}_dp{ext}"

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": global_model.optimizer.state_dict(),
                "dx_categories": dx_categories,
                "dx_to_label_mapping": dx_to_label,
                "num_classes": num_classes,
                "config": {
                    k: v for k, v in vars(args).items() if not k.startswith("_")
                },
                "results": results,
                "differential_privacy": {
                    "enabled": args.enable_dp,
                    "config": dp_config.model_dump() if dp_config else None,
                    "privacy_spent": privacy_spent
                    if args.enable_dp and hasattr(global_model, "get_privacy_spent")
                    else None,
                },
            }

            os.makedirs(
                os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True
            )
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to '{save_path}'")

        finally:
            logger.info("=== Shutting Down ===")
            learning_process.shutdown()

    except Exception as e:
        logger.error(f"DP Skin Lesion Learning Process failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
