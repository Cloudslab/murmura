# type: ignore
import argparse
import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.models.ham10000_models import HAM10000Model, HAM10000ModelComplex
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.data_processing.data_preprocessor import create_image_preprocessor
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
from typing import Union


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dp_ham10000_federated.log"),
        ],
    )


def get_ham10000_transforms(image_size: int = 128):
    """Get appropriate transforms for HAM10000 dataset"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main() -> None:
    """
    HAM10000 Federated Learning with Differential Privacy
    """
    parser = argparse.ArgumentParser(
        description="Federated Learning for HAM10000 with Differential Privacy"
    )

    # Core federated learning arguments
    parser.add_argument(
        "--num_actors",
        type=int,
        default=7,
        help="Number of virtual clients (7 matches HAM10000 classes)",
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

    # Topology arguments
    parser.add_argument(
        "--topology",
        type=str,
        default="star",
        choices=["star", "ring", "complete", "line"],
        help="Network topology between clients",
    )

    # Training arguments
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )

    # HAM10000 specific arguments
    parser.add_argument(
        "--image_size", type=int, default=128, help="Size to resize images to"
    )
    parser.add_argument(
        "--model_complexity",
        type=str,
        default="simple",
        choices=["simple", "complex"],
        help="Model complexity level",
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

    # Logging and monitoring
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
        default="dp_ham10000_federated_model.pt",
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
        default="./visualizations_phase1",
        help="Directory to save visualizations_phase1",
    )
    parser.add_argument(
        "--create_summary",
        action="store_true",
        help="Create summary plot of the training process",
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
    logger = logging.getLogger("murmura.dp_ham10000_example")

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
                dp_config = DPConfig(
                    target_epsilon=args.target_epsilon,
                    target_delta=1e-5,
                    max_grad_norm=1.0,
                    enable_client_dp=True,
                    enable_central_dp=False,
                )
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
        ray_cluster_config = RayClusterConfig(
            logging_level=args.log_level,
        )
        resource_config = ResourceConfig()

        logger.info("=== Loading HAM10000 Dataset ===")
        # Load HAM10000 dataset from Hugging Face
        train_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="kuchikihater/HAM10000",
            split=args.split,
        )

        # For HAM10000, check if test split exists and create if not
        # HAM10000 typically only has a train split, so we need to create train/test splits
        logger.info(f"Initial dataset splits: {list(train_dataset._splits.keys())}")

        if args.test_split not in train_dataset._splits:
            logger.warning(
                f"Test split '{args.test_split}' not found. Creating train/test split from train data."
            )
            # Create train/test split from the single train split
            train_dataset = train_dataset.train_test_split(
                source_split=args.split,
                test_size=0.2,
                seed=42,
                new_split_names=("train", "test"),
            )
            logger.info(
                f"Created train/test split. Available splits: {list(train_dataset._splits.keys())}"
            )
        else:
            logger.info(f"Test split '{args.test_split}' found. Using existing splits.")

        # Create configuration
        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,
            topology=TopologyConfig(topology_type=TopologyType(args.topology)),
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType(args.aggregation_strategy)
            ),
            dataset_name="kuchikihater/HAM10000",
            ray_cluster=ray_cluster_config,
            resources=resource_config,
            feature_columns=["image"],
            label_column="label",  # HAM10000 uses 'label' as the label column
            rounds=args.rounds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            test_split=args.test_split,
            client_sampling_rate=args.client_sampling_rate,
            data_sampling_rate=args.data_sampling_rate,
            enable_subsampling_amplification=args.enable_subsampling_amplification,
        )

        logger.info("=== Creating Data Partitions ===")
        partitioner = PartitionerFactory.create(config)

        logger.info("=== Creating HAM10000 Model ===")
        # Select model based on complexity
        # Always use GroupNorm for federated learning with many clients to avoid
        # BatchNorm issues with small batches
        use_groupnorm = args.enable_dp or args.num_actors > 10

        if args.model_complexity == "complex":
            model = HAM10000ModelComplex(
                input_size=args.image_size, use_dp_compatible_norm=use_groupnorm
            )
        else:
            model = HAM10000Model(
                input_size=args.image_size, use_dp_compatible_norm=use_groupnorm
            )

        # Create image preprocessor for HAM10000 dataset
        logger.info("Creating image preprocessor for HAM10000 dataset")
        image_preprocessor = create_image_preprocessor(
            grayscale=False,  # HAM10000 is RGB
            normalize=True,  # Normalize pixel values to [0,1]
            target_size=(
                args.image_size,
                args.image_size,
            ),  # Resize to model input size
        )

        # Create model wrapper (DP or regular)
        global_model: Union[DPTorchModelWrapper, TorchModelWrapper]
        if args.enable_dp and dp_config:
            logger.info("Creating DP-aware model wrapper")
            global_model = DPTorchModelWrapper(
                model=model,
                dp_config=dp_config,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.SGD,  # SGD works better with DP
                optimizer_kwargs={"lr": args.lr, "momentum": 0.9},
                input_shape=(3, args.image_size, args.image_size),
                device=device,
                data_preprocessor=image_preprocessor,
            )
        else:
            logger.info("Creating regular model wrapper")

            global_model = TorchModelWrapper(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs={"lr": args.lr},
                input_shape=(3, args.image_size, args.image_size),
                device=device,
                data_preprocessor=image_preprocessor,
            )

        logger.info("=== Setting Up Learning Process ===")
        logger.info(
            f"Dataset splits before learning process: {list(train_dataset._splits.keys())}"
        )
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
                    f"dp_ham10000_{args.topology}_{args.aggregation_strategy}"
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
            logger.info("=== HAM10000 DP Federated Learning Setup ===")
            logger.info("Dataset: HAM10000 (7 skin lesion classes)")
            logger.info(f"Clients: {config.num_actors}")
            logger.info(f"Partitioning: {config.partition_strategy} (α={args.alpha})")
            logger.info(f"Aggregation: {config.aggregation.strategy_type}")
            logger.info(f"Topology: {config.topology.topology_type}")
            logger.info(f"Rounds: {config.rounds}")
            logger.info(f"Local epochs: {config.epochs}")
            logger.info(f"Batch size: {config.batch_size}")
            logger.info(f"Learning rate: {config.learning_rate}")
            logger.info(f"Image size: {args.image_size}x{args.image_size}")
            logger.info(f"Model: {args.model_complexity}")
            logger.info(f"Device: {device}")

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
                    sample_rate = args.batch_size / 10015  # HAM10000 has ~10015 images
                    suggested_noise = privacy_accountant.suggest_optimal_noise(
                        sample_rate=sample_rate,
                        epochs=args.epochs
                        * args.rounds,  # Total epochs across all rounds
                        dataset_size=10015,
                    )
                    logger.info(f"Suggested noise multiplier: {suggested_noise:.3f}")
            else:
                logger.info("Differential Privacy: DISABLED")

            logger.info("=== Starting Training ===")
            # Execute learning process
            results = learning_process.execute()

            # Display results_phase1
            logger.info("=== Training Results ===")
            logger.info(
                f"Initial accuracy: {results['initial_metrics']['accuracy']:.4f}"
            )
            logger.info(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")

            # Display privacy results_phase1 if DP was enabled
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

                # Get privacy summary from accountant
                if "privacy_accountant" in locals():
                    privacy_summary = privacy_accountant.get_privacy_summary()
                    logger.info(
                        f"Global privacy utilization: {privacy_summary['global_privacy']['utilization_percentage']:.1f}%"
                    )

            # Create visualization if requested
            if visualizer and args.create_summary:
                logger.info("=== Generating Visualization ===")
                visualizer.render_summary_plot(
                    filename=f"dp_ham10000_{args.topology}_{args.aggregation_strategy}"
                    + ("_dp" if args.enable_dp else "_no_dp")
                    + "_summary.png"
                )

            # Save model
            logger.info("=== Saving Model ===")
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
                "results_phase1": results,
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
        logger.error(f"DP HAM10000 Learning Process failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
