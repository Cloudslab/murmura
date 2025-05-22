import argparse
import os
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
from murmura.visualization.network_visualizer import NetworkVisualizer
from murmura.privacy.accounting.privacy_meter import PrivacyMeter


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


def main() -> None:
    """
    Orchestrate Learning Process
    """
    parser = argparse.ArgumentParser(
        description="Federated Data Distribution Orchestrator"
    )
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
        help="Trim ratio for trimmed_mean strategy (0.1 = 10% trimmed from each end)",
    )

    # Topology arguments
    parser.add_argument(
        "--topology",
        type=str,
        default="star",  # Changed default to star for compatibility with fedavg/trimmed_mean
        choices=["star", "ring", "complete", "line", "custom"],
        help="Network topology between clients",
    )
    parser.add_argument(
        "--hub_index", type=int, default=0, help="Hub node index for star topology"
    )

    # Training arguments
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
        default="mnist_federated_model.pt",
        help="Path to save the final model",
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

    # Differential Privacy arguments
    parser.add_argument(
        "--dp_mechanism",
        type=str,
        choices=["laplace", "gaussian"],
        default=None,
        help="Differential privacy mechanism (laplace or gaussian)",
    )
    parser.add_argument(
        "--dp_epsilon",
        type=float,
        default=None,
        help="Differential privacy epsilon (privacy budget)",
    )
    parser.add_argument(
        "--dp_delta",
        type=float,
        default=None,
        help="Differential privacy delta (for (epsilon, delta)-DP)",
    )
    parser.add_argument(
        "--dp_clipping",
        type=float,
        default=None,
        help="Gradient clipping threshold for DP",
    )
    parser.add_argument(
        "--dp_accountant",
        type=str,
        choices=["rdp", "gdp"],
        default=None,
        help="Privacy accounting method (rdp or gdp)",
    )
    parser.add_argument(
        "--dp_scope",
        type=str,
        choices=["local", "central"],
        default=None,
        help="Apply DP locally (LDP) or centrally (CDP)",
    )

    args = parser.parse_args()

    # Add DP config if any DP argument is set
    dp_enabled = any([
        args.dp_mechanism,
        args.dp_epsilon,
        args.dp_delta,
        args.dp_clipping,
        args.dp_accountant,
        args.dp_scope
    ])
    if dp_enabled:
        process_config["privacy"] = {
            k: v for k, v in {
                "mechanism": args.dp_mechanism,
                "epsilon": args.dp_epsilon,
                "delta": args.dp_delta,
                "clipping": args.dp_clipping,
                "accountant": args.dp_accountant,
                "scope": args.dp_scope
            }.items() if v is not None
        }

    # Set up privacy accounting if DP is enabled
    privacy_meter = None
    if dp_enabled and args.dp_epsilon is not None:
        privacy_meter = PrivacyMeter(epsilon=args.dp_epsilon, delta=args.dp_delta, accountant=args.dp_accountant or 'rdp')

    try:
        # Create configuration from command-line arguments
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
                params={"trim_ratio": args.trim_ratio}
                if args.aggregation_strategy == "trimmed_mean"
                else None,
            ),
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
        # Add DP config if any DP argument is set
        if any([
            args.dp_mechanism,
            args.dp_epsilon,
            args.dp_delta,
            args.dp_clipping,
            args.dp_accountant,
            args.dp_scope
        ]):
            process_config["privacy"] = {
                k: v for k, v in {
                    "mechanism": args.dp_mechanism,
                    "epsilon": args.dp_epsilon,
                    "delta": args.dp_delta,
                    "clipping": args.dp_clipping,
                    "accountant": args.dp_accountant,
                    "scope": args.dp_scope
                }.items() if v is not None
            }

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
        # Create learning process
        learning_process = FederatedLearningProcess(
            config=process_config,
            dataset=train_dataset,
            model=global_model,
        )

        # Set up visualization BEFORE running the learning process
        visualizer = None
        if args.create_animation or args.create_frames or args.create_summary:
            print("\n=== Setting Up Visualization ===")
            # Create visualization directory
            vis_dir = os.path.join(
                args.vis_dir, f"mnist_{args.topology}_{args.aggregation_strategy}"
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
            learning_process.initialize(
                num_actors=config.num_actors,
                topology_config=config.topology,
                aggregation_config=config.aggregation,
                partitioner=partitioner,
            )

            # Print initial summary
            print("\n=== Federated Learning Setup ===")
            print(f"Strategy: {config.partition_strategy}")
            print(f"Clients: {config.num_actors}")
            print(f"Aggregation strategy: {config.aggregation.strategy_type}")
            print(f"Topology: {config.topology.topology_type}")
            print(f"Rounds: {args.rounds}")
            print(f"Local epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Learning rate: {args.lr}")

            print("\n=== Starting Federated Learning ===")
            # Execute the learning process, tracking privacy budget if enabled
            if privacy_meter:
                # Monkey-patch the learning process to log epsilon spend after each round
                orig_execute = learning_process.execute
                def execute_with_privacy_accounting(*a, **kw):
                    results = orig_execute(*a, **kw)
                    rounds = process_config.get("rounds", 5)
                    eps_per_round = args.dp_epsilon / rounds if rounds > 0 else args.dp_epsilon
                    for r in range(rounds):
                        privacy_meter.consume(eps_per_round)
                        print(f"[DP] Privacy budget after round {r+1}: ε = {privacy_meter.get_usage()[0]:.4f}")
                    if privacy_meter.is_exceeded():
                        print(f"[DP] WARNING: Privacy budget exceeded! ε = {privacy_meter.get_usage()[0]:.4f}")
                    else:
                        print(f"[DP] Final privacy budget: ε = {privacy_meter.get_usage()[0]:.4f}")
                    return results
                learning_process.execute = execute_with_privacy_accounting
            _ = learning_process.execute()

            # Generate visualizations if requested
            if visualizer and (
                args.create_animation or args.create_frames or args.create_summary
            ):
                print("\n=== Generating Visualizations ===")

                if args.create_animation:
                    print("Creating animation...")
                    visualizer.render_training_animation(
                        filename=f"mnist_{args.topology}_{args.aggregation_strategy}_animation.mp4",
                        fps=2,
                    )

                if args.create_frames:
                    print("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"mnist_{args.topology}_{args.aggregation_strategy}_step"
                    )

                if args.create_summary:
                    print("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"mnist_{args.topology}_{args.aggregation_strategy}_summary.png"
                    )

            # Save the final model
            print("\n=== Saving Final Model ===")
            save_path = args.save_path
            global_model.save(save_path)
            print(f"Model saved to '{save_path}'")

        finally:
            print("\n=== Shutting Down ===")
            learning_process.shutdown()

    except Exception as e:
        print(f"Learning Process orchestration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
