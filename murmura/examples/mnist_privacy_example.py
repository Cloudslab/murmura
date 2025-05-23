import argparse
import torch
import torch.nn as nn

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.orchestration.learning_process.federated_learning_process import (
    FederatedLearningProcess,
)
from murmura.orchestration.learning_process.decentralized_learning_process import (
    DecentralizedLearningProcess,
)
from murmura.privacy.privacy_config import (
    PrivacyConfig,
    PrivacyMode,
    PrivacyMechanismType,
)
from murmura.visualization.network_visualizer import NetworkVisualizer


class MNISTModel(PyTorchModel):
    """
    Improved CNN model for MNIST classification
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
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout for better generalization
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # Ensure input has the right shape (add channel dimension if needed)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main() -> None:
    """
    Orchestrate Learning Process with Improved Differential Privacy
    """
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Distributed Learning Example"
    )

    # General parameters
    parser.add_argument(
        "--num_actors", type=int, default=10, help="Number of virtual clients"
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

    # Learning process parameters
    parser.add_argument(
        "--learning_process",
        choices=["federated", "decentralized"],
        default="federated",
        help="Learning process type",
    )

    # Aggregation parameters
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        choices=["fedavg", "trimmed_mean", "gossip_avg"],
        default="fedavg",
        help="Aggregation strategy to use",
    )
    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.1,
        help="Trim ratio for trimmed_mean strategy (0.1 = 10% trimmed from each end)",
    )
    parser.add_argument(
        "--mixing_parameter",
        type=float,
        default=0.5,
        help="Mixing parameter for gossip_avg strategy (0.5 = equal mixing)",
    )

    # Topology arguments
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

    # Training arguments
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of learning rounds (reduced default)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (increased default)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="mnist_model_with_privacy.pt",
        help="Path to save the final model",
    )

    # Privacy arguments
    parser.add_argument(
        "--privacy_enabled",
        action="store_true",
        help="Enable differential privacy",
    )
    parser.add_argument(
        "--privacy_mode",
        choices=["local", "central"],
        default="central",
        help="Privacy mode (local or central)",
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=5.0,
        help="Target privacy budget (epsilon)",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5,
        help="Target failure probability (delta)",
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="Initial noise multiplier (if None, calculated adaptively)",
    )
    parser.add_argument(
        "--adaptive_clipping",
        action="store_true",
        help="Enable adaptive clipping",
    )
    parser.add_argument(
        "--clipping_norm",
        type=float,
        default=None,
        help="L2 norm for clipping (if None and adaptive_clipping is True, adaptive clipping is used)",
    )
    parser.add_argument(
        "--per_layer_clipping",
        action="store_true",
        help="Enable per-layer clipping",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Stop training early if privacy budget is exhausted",
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
        "--create_privacy_plot",
        action="store_true",
        help="Create privacy plot",
    )
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second for animation"
    )

    args = parser.parse_args()

    # Validate compatibility of choices
    if args.learning_process == "decentralized" and args.privacy_mode == "central":
        print(
            "WARNING: Decentralized learning only supports Local DP. Forcing Local DP mode."
        )
        args.privacy_mode = "local"

    if args.learning_process == "decentralized" and args.aggregation_strategy not in [
        "gossip_avg"
    ]:
        print(
            "WARNING: Decentralized learning requires a decentralized aggregation strategy. Using gossip_avg."
        )
        args.aggregation_strategy = "gossip_avg"

    if (
        args.learning_process == "federated"
        and args.aggregation_strategy == "gossip_avg"
    ):
        print(
            "WARNING: Federated learning typically uses centralized aggregation strategies. Using fedavg instead."
        )
        args.aggregation_strategy = "fedavg"

    if args.topology == "star" and args.aggregation_strategy == "gossip_avg":
        print(
            "WARNING: Star topology and gossip averaging are not typically used together. This may affect results."
        )

    if args.topology != "star" and args.privacy_mode == "central":
        print(
            "WARNING: Central DP requires a star topology. Switching to star topology."
        )
        args.topology = "star"

    try:
        # Create aggregation configuration
        agg_params = None
        if args.aggregation_strategy == "trimmed_mean":
            agg_params = {"trim_ratio": args.trim_ratio}
        elif args.aggregation_strategy == "gossip_avg":
            agg_params = {"mixing_parameter": args.mixing_parameter}

        # Create privacy configuration if enabled with improved settings
        privacy_config = None
        if args.privacy_enabled:
            # Determine clipping norm based on arguments
            clipping_norm = None
            if args.adaptive_clipping:
                print("Using adaptive clipping for gradient/parameter norms")
                clipping_norm = None  # adaptive
            else:
                clipping_norm = (
                    args.clipping_norm
                )  # Either None or user-specified value
                if clipping_norm is not None:
                    print(f"Using fixed clipping norm: {clipping_norm}")
                else:
                    print("Using default initial clipping norm (1.0)")

            # Determine noise multiplier approach
            adaptive_noise = args.noise_multiplier is None
            if adaptive_noise:
                print(
                    f"Using adaptive noise calibration to target ε={args.target_epsilon}"
                )
            else:
                print(f"Using fixed noise multiplier: {args.noise_multiplier}")

            privacy_config = PrivacyConfig(
                enabled=True,
                mechanism_type=PrivacyMechanismType.GAUSSIAN,
                privacy_mode=PrivacyMode(args.privacy_mode),
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                noise_multiplier=args.noise_multiplier,
                clipping_norm=clipping_norm,
                per_layer_clipping=args.per_layer_clipping,
                max_grad_norm=1.0,  # Default starting value for adaptive clipping
                adaptive_clipping_quantile=0.8,  # Use 90th percentile for clipping
                early_stopping=args.early_stopping,
                adaptive_noise=adaptive_noise,  # New field for adaptive noise calibration
                monitor_frequency=1,  # Monitor privacy budget every round
            )

            # Store additional parameters for proper privacy accounting
            if privacy_config.params is None:
                privacy_config.params = {}

            privacy_config.params["rounds"] = args.rounds
            privacy_config.params["local_epochs"] = args.epochs
            privacy_config.params["batch_size"] = args.batch_size

        # Create configuration
        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,
            topology=TopologyConfig(
                topology_type=TopologyType(args.topology), hub_index=args.hub_index
            ),
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType(args.aggregation_strategy),
                params=agg_params,
            ),
            privacy=privacy_config or PrivacyConfig(),
        )

        # Add additional configuration needed for the learning process
        process_config = config.model_dump()
        process_config.update(
            {
                "rounds": args.rounds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "test_split": args.test_split,
                "feature_columns": ["image"],
                "label_column": "label",
                "learning_rate": args.lr,
            }
        )

        print("\n=== Loading MNIST Dataset ===")
        # Load MNIST Dataset for training and testing
        train_dataset = MDataset.load(
            DatasetSource.HUGGING_FACE,
            dataset_name=config.dataset_name,
            split=config.split,
        )

        test_dataset = MDataset.load(
            DatasetSource.HUGGING_FACE,
            dataset_name=config.dataset_name,
            split=args.test_split,
        )

        # Merge datasets to have both splits available
        train_dataset.merge_splits(test_dataset)

        print("\n=== Creating Data Partitions ===")
        # Create partitioner
        partitioner = PartitionerFactory.create(config)

        print("\n=== Creating and Initializing Model ===")
        # Create the MNIST model with PyTorch wrapper
        model = MNISTModel()
        input_shape = (1, 28, 28)  # (channels, height, width)

        global_model = TorchModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": args.lr},
            input_shape=input_shape,
        )

        print("\n=== Setting Up Learning Process ===")
        # Create appropriate learning process
        if args.learning_process == "federated":
            learning_process = FederatedLearningProcess(
                config=process_config,
                dataset=train_dataset,
                model=global_model,
            )
            print("Using Federated Learning Process")
        else:  # decentralized
            learning_process = DecentralizedLearningProcess(
                config=process_config,
                dataset=train_dataset,
                model=global_model,
            )
            print("Using Decentralized Learning Process")

        # Set up visualization BEFORE executing the learning process
        visualizer = None
        if any(
            [
                args.create_animation,
                args.create_frames,
                args.create_summary,
                args.create_privacy_plot,
            ]
        ):
            print("\n=== Setting Up Visualization ===")
            # Create visualization directory
            privacy_str = "_with_privacy" if args.privacy_enabled else ""
            vis_dir = os.path.join(
                args.vis_dir,
                f"{args.learning_process}_{args.topology}_{args.aggregation_strategy}{privacy_str}",
            )
            os.makedirs(vis_dir, exist_ok=True)

            # Create visualizer
            visualizer = NetworkVisualizer(output_dir=vis_dir)

            # Register visualizer with learning process
            learning_process.register_observer(visualizer)
            print("Registered visualizer with learning process")
            print(
                f"Current observers: {len(learning_process.training_monitor.observers)}"
            )

        try:
            # Initialize the learning process
            print("\n=== Initializing Learning Process ===")
            learning_process.initialize(
                num_actors=config.num_actors,
                topology_config=config.topology,
                aggregation_config=config.aggregation,
                partitioner=partitioner,
                privacy_config=config.privacy if args.privacy_enabled else None,
            )

            # Print initial summary
            print(f"\n=== {args.learning_process.capitalize()} Learning Setup ===")
            print(f"Strategy: {config.partition_strategy}")
            print(f"Clients: {config.num_actors}")
            print(f"Aggregation strategy: {config.aggregation.strategy_type}")
            print(f"Topology: {config.topology.topology_type}")
            print(f"Rounds: {args.rounds}")
            print(f"Local epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Learning rate: {args.lr}")

            if args.privacy_enabled:
                print(f"\n=== Privacy Settings ===")
                print(f"Privacy mode: {args.privacy_mode.upper()}")
                print(f"Target epsilon: {args.target_epsilon}")
                print(f"Target delta: {args.target_delta}")
                if args.noise_multiplier is not None:
                    print(f"Noise multiplier: {args.noise_multiplier}")
                else:
                    print("Noise multiplier: Adaptive calibration")

                if args.adaptive_clipping:
                    print("Clipping: Adaptive")
                elif args.clipping_norm is not None:
                    print(f"Clipping norm: {args.clipping_norm}")
                else:
                    print("Clipping norm: Default (1.0)")

                if args.per_layer_clipping:
                    print("Per-layer clipping: Enabled")
                else:
                    print("Per-layer clipping: Disabled (global clipping)")

            print(
                f"\n=== Starting {args.learning_process.capitalize()} Learning with{'' if args.privacy_enabled else 'out'} Privacy ==="
            )
            # Execute the learning process
            results = learning_process.execute()

            # Generate visualizations if requested
            if visualizer:
                if any(
                    [
                        args.create_animation,
                        args.create_frames,
                        args.create_summary,
                        args.create_privacy_plot,
                    ]
                ):
                    print("\n=== Generating Visualizations ===")

                if args.create_animation:
                    print("Creating animation...")
                    visualizer.render_training_animation(
                        filename=f"{args.learning_process}_{args.topology}_{args.aggregation_strategy}{privacy_str}_animation.mp4",
                        fps=args.fps,
                    )

                if args.create_frames:
                    print("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"{args.learning_process}_{args.topology}_{args.aggregation_strategy}{privacy_str}_step"
                    )

                if args.create_summary:
                    print("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"{args.learning_process}_{args.topology}_{args.aggregation_strategy}{privacy_str}_summary.png"
                    )

                if args.create_privacy_plot and args.privacy_enabled:
                    print("Creating privacy plot...")
                    visualizer.render_privacy_plot(
                        filename=f"{args.learning_process}_{args.topology}_{args.aggregation_strategy}_privacy.png"
                    )

            # Save the final model
            print("\n=== Saving Final Model ===")
            save_path = args.save_path
            global_model.save(save_path)
            print(f"Model saved to '{save_path}'")

            # Print final results
            print("\n=== Final Results ===")
            print(
                f"Initial accuracy: {results['initial_metrics']['accuracy'] * 100:.2f}%"
            )
            print(f"Final accuracy: {results['final_metrics']['accuracy'] * 100:.2f}%")
            print(f"Accuracy improvement: {results['accuracy_improvement'] * 100:.2f}%")

            if args.privacy_enabled and "privacy" in results:
                print(f"\n=== Final Privacy Budget ===")
                print(f"Epsilon: {results['privacy']['epsilon']:.4f}")
                print(f"Delta: {results['privacy']['delta']:.6f}")
                print(f"Noise multiplier: {results['privacy']['noise_multiplier']:.4f}")
                if "clipping_norm" in results["privacy"]:
                    print(f"Clipping norm: {results['privacy']['clipping_norm']:.4f}")

        finally:
            print("\n=== Shutting Down ===")
            learning_process.shutdown()

    except Exception as e:
        print(f"Learning Process failed: {str(e)}")
        raise


if __name__ == "__main__":
    import os

    main()
