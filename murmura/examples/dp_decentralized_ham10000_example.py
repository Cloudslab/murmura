# type: ignore
import argparse
import os
import logging
import torch
import torch.nn as nn

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

# Import attack components
from murmura.attacks.attack_config import AttackConfig


def create_ham10000_preprocessor():
    """
    Create HAM10000-specific data preprocessor.
    """
    try:
        from murmura.data_processing.generic_preprocessor import (  # type: ignore[import-untyped]
            create_image_preprocessor,
        )

        # HAM10000-specific configuration
        return create_image_preprocessor(
            grayscale=False,  # HAM10000 is RGB
            normalize=True,
            target_size=(128, 128),  # Resize to standard size
        )
    except ImportError:
        # Generic preprocessor not available, use automatic detection
        logging.getLogger("murmura.dp_decentralized_ham10000_example").info(
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
            logging.FileHandler("dp_decentralized_ham10000.log"),
        ],
    )


def main() -> None:
    """
    HAM10000 Decentralized Learning with Differential Privacy
    """
    parser = argparse.ArgumentParser(
        description="HAM10000 Decentralized Learning with Differential Privacy"
    )

    # Core learning arguments
    parser.add_argument(
        "--num_actors",
        type=int,
        default=7,
        help="Total number of virtual clients (7 matches HAM10000 classes)",
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
        choices=["gossip_avg"],
        default="gossip_avg",
        help="Aggregation strategy (gossip_avg for decentralized)",
    )

    # Topology arguments (excluding star for decentralized)
    parser.add_argument(
        "--topology",
        type=str,
        default="ring",
        choices=["ring", "complete", "line"],
        help="Network topology between clients (no star for decentralized)",
    )

    # Training arguments
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of decentralized learning rounds"
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
        "--target_epsilon_per_round",
        type=float,
        default=1.0,
        help="Target privacy budget per round (epsilon)",
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
        default="dp_decentralized_ham10000_model.pt",
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

    # Attack configuration arguments
    parser.add_argument(
        "--malicious_clients_ratio",
        type=float,
        default=0.0,
        help="Fraction of clients to make malicious (0.0-1.0)",
    )
    parser.add_argument(
        "--attack_type",
        choices=["label_flipping", "gradient_manipulation", "both"],
        default="label_flipping",
        help="Type of poisoning attack to perform",
    )
    parser.add_argument(
        "--attack_intensity_start",
        type=float,
        default=0.1,
        help="Initial attack intensity (0.0-1.0)",
    )
    parser.add_argument(
        "--attack_intensity_end",
        type=float,
        default=1.0,
        help="Final attack intensity (0.0-1.0)",
    )
    parser.add_argument(
        "--intensity_progression",
        choices=["linear", "exponential", "step"],
        default="linear",
        help="How attack intensity increases over rounds",
    )
    parser.add_argument(
        "--label_flip_target",
        type=int,
        default=None,
        help="Target label for label flipping attacks",
    )
    parser.add_argument(
        "--label_flip_source",
        type=int,
        default=None,
        help="Source label for label flipping attacks",
    )
    parser.add_argument(
        "--gradient_noise_scale",
        type=float,
        default=1.0,
        help="Scale factor for gradient noise injection",
    )
    parser.add_argument(
        "--gradient_sign_flip_prob",
        type=float,
        default=0.1,
        help="Probability of flipping gradient signs",
    )
    parser.add_argument(
        "--attack_start_round",
        type=int,
        default=1,
        help="Round to start attacks",
    )

    # Decentralized-specific arguments
    parser.add_argument(
        "--gossip_rounds",
        type=int,
        default=5,
        help="Number of gossip rounds per iteration",
    )
    parser.add_argument(
        "--gossip_subset_size",
        type=int,
        default=None,
        help="Subset size for gossip averaging (None for all neighbors)",
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
    logger = logging.getLogger("murmura.dp_decentralized_ham10000_example")

    try:
        # Select device
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        logger.info(f"Using {device.upper()} device for training")

        # Check compatibility of topology and strategy before proceeding
        topology_type = TopologyType(args.topology)
        strategy_type = AggregationStrategyType(args.aggregation_strategy)

        # Validate decentralized compatibility
        from murmura.aggregation.strategies.gossip_avg import GossipAvg

        if not TopologyCompatibilityManager.is_compatible(GossipAvg, topology_type):
            compatible_topologies = (
                TopologyCompatibilityManager.get_compatible_topologies(GossipAvg)
            )
            logger.error(
                f"Strategy {args.aggregation_strategy} is not compatible with topology {args.topology}."
            )
            logger.error(
                f"Compatible topologies: {[t.value for t in compatible_topologies]}"
            )
            return

        # Create DP configuration
        dp_config = None
        if args.enable_dp:
            logger.info("=== Configuring Differential Privacy ===")

            # Privacy budget allocation:
            # - Total epsilon budget: What each client spends across ALL rounds and epochs
            # - Per-round epsilon: For tracking purposes only
            per_round_epsilon = args.target_epsilon_per_round
            total_epsilon_budget = per_round_epsilon * args.rounds
            total_epochs = args.epochs * args.rounds  # Total epochs across all rounds

            logger.info(f"Privacy budget per round per client: {per_round_epsilon:.2f}")
            logger.info(
                f"Total privacy budget per client: {total_epsilon_budget:.2f} (across {args.rounds} rounds)"
            )
            logger.info(f"Total epochs across all rounds: {total_epochs}")
            logger.info(
                f"NOTE: Opacus will receive total budget ({total_epsilon_budget:.2f}) for {total_epochs} epochs"
            )

            if args.dp_preset == "high_privacy":
                dp_config = DPConfig.create_high_privacy()
                # Override with calculated total budget
                dp_config.target_epsilon = total_epsilon_budget
            elif args.dp_preset == "medium_privacy":
                dp_config = DPConfig(
                    target_epsilon=total_epsilon_budget,
                    target_delta=1e-5,
                    max_grad_norm=1.0,
                    enable_client_dp=True,
                    enable_central_dp=False,
                )
            elif args.dp_preset == "low_privacy":
                dp_config = DPConfig(
                    target_epsilon=total_epsilon_budget,
                    target_delta=1e-4,
                    max_grad_norm=2.0,
                    enable_client_dp=True,
                    enable_central_dp=False,
                )
            else:  # custom
                dp_config = DPConfig(
                    target_epsilon=total_epsilon_budget,
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
                f"DP Configuration: ε={dp_config.target_epsilon} (total), δ={dp_config.target_delta}"
            )
            logger.info(f"Max grad norm: {dp_config.max_grad_norm}")
            logger.info(
                f"Client DP: {dp_config.enable_client_dp}, Central DP: {dp_config.enable_central_dp}"
            )

            # Initialize privacy accountant
            privacy_accountant = PrivacyAccountant(dp_config)
        else:
            logger.info("Differential privacy is DISABLED")

        # Create attack configuration if attacks are enabled
        attack_config = None
        if args.malicious_clients_ratio > 0.0:
            logger.info("=== Configuring Model Poisoning Attacks ===")
            attack_config = AttackConfig(
                malicious_clients_ratio=args.malicious_clients_ratio,
                attack_type=args.attack_type,
                attack_intensity_start=args.attack_intensity_start,
                attack_intensity_end=args.attack_intensity_end,
                intensity_progression=args.intensity_progression,
                label_flip_target=args.label_flip_target,
                label_flip_source=args.label_flip_source,
                gradient_noise_scale=args.gradient_noise_scale,
                gradient_sign_flip_prob=args.gradient_sign_flip_prob,
                attack_start_round=args.attack_start_round,
                log_attack_details=True
            )
            
            logger.info("Attack configuration created:")
            logger.info(f"  - Malicious clients ratio: {args.malicious_clients_ratio}")
            logger.info(f"  - Attack type: {args.attack_type}")
            logger.info(f"  - Attack intensity: {args.attack_intensity_start} -> {args.attack_intensity_end}")
            logger.info(f"  - Intensity progression: {args.intensity_progression}")
            logger.info(f"  - Attack start round: {args.attack_start_round}")
            
            if args.attack_type in ["label_flipping", "both"]:
                logger.info(f"  - Label flip target: {args.label_flip_target}")
                logger.info(f"  - Label flip source: {args.label_flip_source}")
            
            if args.attack_type in ["gradient_manipulation", "both"]:
                logger.info(f"  - Gradient noise scale: {args.gradient_noise_scale}")
                logger.info(f"  - Gradient sign flip prob: {args.gradient_sign_flip_prob}")
                
        else:
            logger.info("Model poisoning attacks are DISABLED")

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

        # Create configuration with gossip-specific settings
        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,
            topology=TopologyConfig(topology_type=TopologyType(args.topology)),
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType(args.aggregation_strategy),
                gossip_rounds=args.gossip_rounds,
                gossip_subset_size=args.gossip_subset_size,
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
            attack_config=attack_config,
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
            from murmura.model.pytorch_model import TorchModelWrapper

            global_model = TorchModelWrapper(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs={"lr": args.lr},
                input_shape=(3, args.image_size, args.image_size),
                device=device,
                data_preprocessor=image_preprocessor,
            )

        logger.info("=== Setting Up Decentralized Learning Process ===")
        logger.info(
            f"Dataset splits before learning process: {list(train_dataset._splits.keys())}"
        )
        learning_process = DecentralizedLearningProcess(
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
                    f"dp_decentralized_ham10000_{args.topology}_{args.aggregation_strategy}"
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
            logger.info("=== HAM10000 DP Decentralized Learning Setup ===")
            logger.info("Dataset: HAM10000 (7 skin lesion classes)")
            logger.info(f"Nodes: {config.num_actors}")
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
            logger.info(f"Gossip rounds: {args.gossip_rounds}")

            if args.enable_dp and dp_config is not None:
                logger.info("=== Differential Privacy Settings ===")
                logger.info(
                    f"Total privacy budget per client: ε={dp_config.target_epsilon}, δ={dp_config.target_delta}"
                )
                logger.info(
                    f"Per-round privacy budget: ε={per_round_epsilon}, δ={dp_config.target_delta}"
                )
                logger.info(f"Max gradient norm: {dp_config.max_grad_norm}")
                logger.info(f"Client DP: {dp_config.enable_client_dp}")
                logger.info(f"Central DP: {dp_config.enable_central_dp}")
                logger.info(f"Mechanism: {dp_config.mechanism.value}")
                logger.info(f"Accounting: {dp_config.accounting_method.value}")

                # Note: Noise multiplier is auto-calculated by each client based on actual partition size
                if dp_config.auto_tune_noise and dp_config.noise_multiplier is None:
                    logger.info(
                        "Noise multiplier will be auto-calculated by each client based on actual partition size"
                    )
            else:
                logger.info("Differential Privacy: DISABLED")

            logger.info("=== Starting Decentralized Training ===")
            # Execute learning process
            results = learning_process.execute()

            # Display results_phase1
            logger.info("=== Training Results ===")
            logger.info(
                f"Initial accuracy: {results['initial_metrics']['accuracy']:.4f}"
            )
            logger.info(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")

            # Display privacy results if DP was enabled
            privacy_spent = None
            if args.enable_dp and "privacy_metrics" in results:
                logger.info("=== Privacy Results ===")
                privacy_metrics = results["privacy_metrics"]

                # Display per-round privacy information
                if "per_round_privacy" in privacy_metrics:
                    logger.info("Per-round privacy consumption:")
                    for round_idx, round_privacy in enumerate(
                        privacy_metrics["per_round_privacy"]
                    ):
                        logger.info(
                            f"  Round {round_idx + 1}: ε={round_privacy['epsilon']:.3f}, δ={round_privacy['delta']:.2e}"
                        )

                # Display total privacy spent
                if "total_privacy_spent" in privacy_metrics:
                    privacy_spent = privacy_metrics["total_privacy_spent"]
                    logger.info(
                        f"Total privacy spent: ε={privacy_spent['epsilon']:.3f}, δ={privacy_spent['delta']:.2e}"
                    )

                    if dp_config is not None:
                        logger.info(
                            f"Per-round budget: ε={args.target_epsilon_per_round:.2f}"
                        )
                        logger.info(
                            f"Total budget per client: ε={dp_config.target_epsilon:.2f}"
                        )
                        logger.info(f"Total epochs: {total_epochs}")

                        remaining_eps = (
                            dp_config.target_epsilon - privacy_spent["epsilon"]
                        )
                        logger.info(f"Remaining budget: ε={remaining_eps:.3f}")

                        if privacy_spent["epsilon"] > dp_config.target_epsilon:
                            logger.warning("Privacy budget exceeded!")
                        else:
                            logger.info("Privacy budget respected ✓")

                        # Show budget utilization
                        utilization = (
                            privacy_spent["epsilon"] / dp_config.target_epsilon
                        ) * 100
                        logger.info(f"Privacy budget utilization: {utilization:.1f}%")

                # Display privacy summary if available
                if "privacy_summary" in privacy_metrics:
                    privacy_summary = privacy_metrics["privacy_summary"]
                    if "global_privacy" in privacy_summary:
                        logger.info(
                            f"Global privacy utilization: {privacy_summary['global_privacy']['utilization_percentage']:.1f}%"
                        )

                # Get privacy summary from accountant
                if "privacy_accountant" in locals():
                    privacy_summary = privacy_accountant.get_privacy_summary()
                    logger.info(
                        f"Global privacy utilization from accountant: {privacy_summary['global_privacy']['utilization_percentage']:.1f}%"
                    )
            elif args.enable_dp:
                logger.warning("Privacy metrics not available in results")

            # Create visualization if requested
            if visualizer and args.create_summary:
                logger.info("=== Generating Visualization ===")
                visualizer.render_summary_plot(
                    filename=f"dp_decentralized_ham10000_{args.topology}_{args.aggregation_strategy}"
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
                    if args.enable_dp and "privacy_metrics" in results
                    else None,
                    "privacy_metrics": results.get("privacy_metrics")
                    if args.enable_dp
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
        logger.error(f"DP Decentralized HAM10000 Learning Process failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
