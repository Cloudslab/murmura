import argparse
import os
import logging
import numpy as np
import torch
import torch.nn as nn

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.models.eurosat_models import EuroSATModel, EuroSATModelComplex, EuroSATModelLite
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
from murmura.model.pytorch_model import TorchModelWrapper
from typing import Union

# Import attack components
from murmura.attacks.attack_config import AttackConfig

# Import trust monitoring components
from murmura.trust_monitoring.trust_config import TrustMonitorConfig


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dp_decentralized_eurosat.log"),
        ],
    )


def main() -> None:
    """
    EuroSAT Decentralized Learning with Differential Privacy
    """
    parser = argparse.ArgumentParser(
        description="EuroSAT Decentralized Learning with Differential Privacy"
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
        "--data_partitioning_seed",
        type=int,
        default=42,
        help="Seed for reproducible data partitioning across experiments",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=42,
        help="Seed for reproducible model initialization",
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
        choices=["gossip_avg", "trust_weighted_gossip"],
        default="gossip_avg",
        help="Aggregation strategy to use",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple", "complex", "lite"],
        default="simple",
        help="Model architecture to use",
    )

    # Topology arguments
    parser.add_argument(
        "--topology",
        type=str,
        default="ring",
        choices=["ring", "complete", "line"],
        help="Network topology between clients (star not supported in decentralized mode)",
    )

    # Gossip-specific arguments
    parser.add_argument(
        "--mixing_parameter",
        type=float,
        default=0.5,
        help="Mixing parameter for gossip averaging (0.0-1.0)",
    )

    # Training arguments
    parser.add_argument(
        "--rounds", type=int, default=20, help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of local epochs per round"
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

    # Differential Privacy arguments
    parser.add_argument(
        "--enable_dp", action="store_true", help="Enable differential privacy"
    )
    parser.add_argument(
        "--target_epsilon_per_round",
        type=float,
        default=1.0,
        help="Per-round per-node privacy budget (epsilon) - total budget will be multiplied by number of rounds",
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
    parser.add_argument(
        "--malicious_node_seed",
        type=int,
        default=None,
        help="Seed for reproducible malicious node selection (None = random)",
    )

    # Trust monitoring arguments
    parser.add_argument(
        "--enable_trust_monitoring",
        action="store_true",
        help="Enable trust monitoring for malicious behavior detection",
    )
    parser.add_argument(
        "--enable_trust_weighted_aggregation",
        action="store_true",
        default=True,
        help="Apply trust scores as weights during aggregation (default: True)",
    )
    parser.add_argument(
        "--enable_exponential_decay",
        action="store_true",
        help="Use exponential decay for repeated trust violations (more aggressive)",
    )
    parser.add_argument(
        "--exponential_decay_base",
        type=float,
        default=0.8,
        help="Base for exponential decay (lower = more aggressive, default: 0.8)",
    )
    parser.add_argument(
        "--trust_scaling_factor",
        type=float,
        default=1.0,
        help="Scaling factor for trust-to-weight conversion (lower = more aggressive, default: 1.0)",
    )
    parser.add_argument(
        "--anomaly_detection_method",
        choices=["cusum", "z_score", "iqr"],
        default="cusum",
        help="Anomaly detection method for trust monitoring",
    )
    parser.add_argument(
        "--suspicion_threshold",
        type=float,
        default=0.7,
        help="Threshold for marking nodes as suspicious (0.0-1.0)",
    )
    parser.add_argument(
        "--trust_decay_rate",
        type=float,
        default=0.1,
        help="Rate at which trust decays for suspicious nodes",
    )
    parser.add_argument(
        "--min_trust_score",
        type=float,
        default=0.1,
        help="Minimum trust score (prevents complete exclusion)",
    )
    parser.add_argument(
        "--trust_weight_exponent",
        type=float,
        default=1.0,
        help="Exponent for trust score scaling (higher = more aggressive, default: 1.0)",
    )
    parser.add_argument(
        "--enable_trust_resource_monitoring",
        action="store_true",
        help="Enable resource monitoring specifically for trust monitor operations",
    )

    # Visualization arguments
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="./visualizations_decentralized_eurosat",
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
    parser.add_argument(
        "--monitor_resources",
        action="store_true",
        help="Monitor and log resource usage during training",
    )
    parser.add_argument(
        "--health_check_interval",
        type=int,
        default=5,
        help="Interval for health checks in rounds",
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
        default="dp_decentralized_eurosat_model.pt",
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

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.dp_decentralized_eurosat_example")

    try:
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
                # Override with total epsilon budget
                dp_config.target_epsilon = total_epsilon_budget
            elif args.dp_preset == "medium_privacy":
                dp_config = DPConfig(
                    target_epsilon=total_epsilon_budget,
                    target_delta=args.target_delta,
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
                malicious_node_seed=args.malicious_node_seed,
                log_attack_details=True,
            )

            logger.info("Attack configuration created:")
            logger.info(f"  - Malicious clients ratio: {args.malicious_clients_ratio}")
            logger.info(f"  - Attack type: {args.attack_type}")
            logger.info(f"  - Attack intensity: {args.attack_intensity_start} -> {args.attack_intensity_end}")
            logger.info(f"  - Intensity progression: {args.intensity_progression}")
            logger.info(f"  - Attack start round: {args.attack_start_round}")
            logger.info(f"  - Malicious node seed: {args.malicious_node_seed}")

            if args.attack_type in ["label_flipping", "both"]:
                logger.info(f"  - Label flip target: {args.label_flip_target}")
                logger.info(f"  - Label flip source: {args.label_flip_source}")

            if args.attack_type in ["gradient_manipulation", "both"]:
                logger.info(f"  - Gradient noise scale: {args.gradient_noise_scale}")
                logger.info(f"  - Gradient sign flip prob: {args.gradient_sign_flip_prob}")

        else:
            logger.info("Model poisoning attacks are DISABLED")

        # Create trust monitoring configuration if enabled
        trust_config = None
        if args.enable_trust_monitoring:
            logger.info("=== Trust monitoring ENABLED ===")
            trust_config = TrustMonitorConfig(
                enable_trust_monitoring=True,
                enable_trust_weighted_aggregation=args.enable_trust_weighted_aggregation,
                enable_exponential_decay=args.enable_exponential_decay,
                exponential_decay_base=args.exponential_decay_base,
                trust_scaling_factor=args.trust_scaling_factor,
                trust_weight_exponent=args.trust_weight_exponent,
                enable_trust_resource_monitoring=args.enable_trust_resource_monitoring,
            )
            if args.enable_trust_weighted_aggregation:
                logger.info("=== Trust-weighted aggregation ENABLED ===")
                if args.enable_exponential_decay:
                    logger.info(
                        f"=== Exponential decay ENABLED (base: {args.exponential_decay_base}) ==="
                    )
                logger.info(
                    f"=== Trust scaling factor: {args.trust_scaling_factor} ==="
                )
                logger.info(
                    f"=== Trust weight exponent: {args.trust_weight_exponent} ==="
                )
            else:
                logger.info("=== Trust-weighted aggregation DISABLED ===")
        else:
            logger.info("Trust monitoring is DISABLED")

        # Ray cluster configuration
        ray_cluster_config = RayClusterConfig(
            logging_level=args.log_level,
        )
        resource_config = ResourceConfig()

        logger.info("=== Loading EuroSAT Dataset ===")
        # Load EuroSAT dataset
        train_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="cm93/eurosat",
            split=args.split,
        )

        test_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="cm93/eurosat",
            split=args.test_split,
        )

        # Merge test split into main dataset
        train_dataset.merge_splits(test_dataset)

        # Create data preprocessor for EuroSAT
        preprocessor = create_image_preprocessor(
            normalize=True,
            target_size=(64, 64)
        )

        # Create configuration
        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            data_partitioning_seed=args.data_partitioning_seed,
            model_seed=args.model_seed,
            split=args.split,
            topology=TopologyConfig(
                topology_type=topology_type,
                hub_index=0,  # Not used for decentralized topologies
            ),
            aggregation=AggregationConfig(
                strategy_type=strategy_type,
                params={"mixing_parameter": args.mixing_parameter},
            ),
            dataset_name="cm93/eurosat",
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
            attack_config=attack_config,
            trust_monitoring=trust_config,
        )

        logger.info("=== Creating Data Partitions ===")
        partitioner = PartitionerFactory.create(config)

        logger.info("=== Creating EuroSAT Model ===")
        # Set random seeds for reproducible model initialization
        torch.manual_seed(config.model_seed)
        torch.cuda.manual_seed_all(config.model_seed)
        np.random.seed(config.model_seed)
        logger.info(f"Set model initialization seeds to {config.model_seed}")

        # Select model based on complexity
        # Use GroupNorm for many clients or when DP is enabled
        use_groupnorm = args.enable_dp or args.num_actors > 10

        if args.model == "complex":
            model = EuroSATModelComplex(
                input_size=64, use_dp_compatible_norm=use_groupnorm
            )
        elif args.model == "lite":
            model = EuroSATModelLite(
                input_size=64, use_dp_compatible_norm=use_groupnorm
            )
        else:  # simple
            model = EuroSATModel(
                input_size=64, use_dp_compatible_norm=use_groupnorm
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
                input_shape=(3, 64, 64),
                device=device,
                data_preprocessor=preprocessor,
            )
        else:
            logger.info("Creating regular model wrapper")

            global_model = TorchModelWrapper(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs={"lr": args.lr},
                input_shape=(3, 64, 64),
                data_preprocessor=preprocessor,
            )

        logger.info("=== Setting Up Decentralized Learning Process ===")
        learning_process = DecentralizedLearningProcess(
            config=config,
            dataset=train_dataset,
            model=global_model,
        )

        # Set up visualization if requested
        visualizer = None
        if args.create_animation or args.create_frames or args.create_summary:
            logger.info("=== Setting Up Visualization ===")
            # Create visualization directory
            if args.experiment_name:
                vis_dir = os.path.join(args.vis_dir, args.experiment_name)
            else:
                vis_dir = os.path.join(
                    args.vis_dir,
                    f"dp_decentralized_eurosat_{args.topology}_{args.aggregation_strategy}"
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

            # Get cluster information
            cluster_summary = learning_process.get_cluster_summary()
            logger.info("=== Cluster Summary ===")
            logger.info(f"Total actors: {cluster_summary.get('total_actors', 'unknown')}")
            logger.info(f"Topology: {cluster_summary.get('topology', 'unknown')}")

            # Print experiment summary
            logger.info("=== EuroSAT Decentralized Learning Setup ===")
            logger.info("Dataset: EuroSAT")
            logger.info(f"Clients: {config.num_actors}")
            logger.info(f"Partitioning: {config.partition_strategy} (α={args.alpha})")
            logger.info(f"Aggregation strategy: {config.aggregation.strategy_type}")
            logger.info(f"Topology: {config.topology.topology_type}")
            logger.info(f"Mixing parameter: {args.mixing_parameter}")
            logger.info(f"Rounds: {config.rounds}")
            logger.info(f"Local epochs: {config.epochs}")
            logger.info(f"Batch size: {config.batch_size}")
            logger.info(f"Learning rate: {config.learning_rate}")
            logger.info(f"Device: {device}")
            logger.info(f"Model: {args.model}")

            if args.enable_dp and dp_config is not None:
                logger.info("=== Differential Privacy Settings ===")
                logger.info(f"Total privacy budget per client: ε={dp_config.target_epsilon}, δ={dp_config.target_delta}")
                logger.info(f"Per-round privacy budget: ε={per_round_epsilon}, δ={dp_config.target_delta}")
                logger.info(f"Max gradient norm: {dp_config.max_grad_norm}")
                logger.info(f"Client DP: {dp_config.enable_client_dp}")
                logger.info(f"Central DP: {dp_config.enable_central_dp}")
            else:
                logger.info("Differential Privacy: DISABLED")

            logger.info("=== Starting EuroSAT Decentralized Learning ===")

            # Execute the learning process
            results = learning_process.execute()

            # Display results
            logger.info("=== EuroSAT Decentralized Learning Results ===")
            logger.info(
                f"Initial accuracy: {results['initial_metrics']['accuracy']:.4f}"
            )
            logger.info(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")
            logger.info("Training completed successfully!")

            # Save the final model
            if args.save_path:
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
                        "config": dp_config.model_dump() if dp_config else None,
                        "privacy_spent": results.get("privacy_metrics") if args.enable_dp else None,
                    },
                    "learning_mode": "decentralized",
                }

                os.makedirs(
                    os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True
                )
                torch.save(checkpoint, args.save_path)
                logger.info(f"Model saved to {args.save_path}")

            # Generate visualization summary
            if visualizer:
                try:
                    if args.create_summary:
                        visualizer.create_summary()
                        logger.info(f"Training summary plot saved to {vis_dir}")

                    if args.create_animation:
                        visualizer.create_animation(fps=args.fps)
                        logger.info(f"Training animation saved to {vis_dir}")

                    logger.info(
                        f"All visualizations saved to: {os.path.abspath(vis_dir)}"
                    )
                except Exception as e:
                    logger.warning(f"Error creating visualizations: {e}")

        except Exception as e:
            logger.error(f"Error during decentralized learning: {e}")
            raise

        finally:
            # Clean up resources
            logger.info("=== Shutting Down ===")
            learning_process.shutdown()

    except Exception as e:
        logger.error(f"DP Decentralized EuroSAT Learning Process failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()