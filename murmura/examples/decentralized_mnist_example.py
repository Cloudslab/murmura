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
from murmura.orchestration.learning_process.decentralized_learning_process import DecentralizedLearningProcess
from murmura.network_management.topology_compatibility import TopologyCompatibilityManager
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


def main() -> None:
    """
    Orchestrate Decentralized Learning Process
    """
    parser = argparse.ArgumentParser(
        description="Decentralized Learning Orchestrator"
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
        choices=["gossip_avg"],  # Only decentralized strategies
        default="gossip_avg",
        help="Aggregation strategy to use",
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
        default="ring",  # Default to ring for decentralized learning
        choices=["star", "ring", "complete", "line", "custom"],
        help="Network topology between clients",
    )

    # Training arguments
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of learning rounds"
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
        help="Path to save the final model"
    )

    # Visualization arguments
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--create_animation",
        action="store_true",
        help="Create animation of the training process"
    )
    parser.add_argument(
        "--create_frames",
        action="store_true",
        help="Create individual frames of the training process"
    )
    parser.add_argument(
        "--create_summary",
        action="store_true",
        help="Create summary plot of the training process"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for animation"
    )

    args = parser.parse_args()

    # Check compatibility of topology and strategy before proceeding
    topology_type = TopologyType(args.topology)
    strategy_type = AggregationStrategyType(args.aggregation_strategy)

    # Get compatible strategies for the selected topology
    from murmura.aggregation.strategies.gossip_avg import GossipAvg
    if not TopologyCompatibilityManager.is_compatible(GossipAvg, topology_type):
        compatible_topologies = TopologyCompatibilityManager.get_compatible_topologies(GossipAvg)
        print(f"Error: Strategy {args.aggregation_strategy} is not compatible with topology {args.topology}.")
        print(f"Compatible topologies: {[t.value for t in compatible_topologies]}")
        return

    try:
        # Create configuration from command-line arguments
        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,
            topology=TopologyConfig(
                topology_type=topology_type,
                hub_index=0,  # Default hub index (not used for non-star topologies)
            ),
            aggregation=AggregationConfig(
                strategy_type=strategy_type,
                params={"mixing_parameter": args.mixing_parameter},
            ),
        )

        # Add additional configuration needed for the learning process
        process_config = config.model_dump()
        process_config.update({
            "rounds": args.rounds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_split": args.test_split,
            "feature_columns": ["image"],
            "label_column": "label",
            "learning_rate": args.lr,
        })

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

        print("\n=== Setting Up Decentralized Learning Process ===")
        # Create learning process
        learning_process = DecentralizedLearningProcess(
            config=process_config,
            dataset=train_dataset,
            model=global_model,
        )

        # Set up visualization BEFORE executing the learning process
        visualizer = None
        if args.create_animation or args.create_frames or args.create_summary:
            print("\n=== Setting Up Visualization ===")
            # Create visualization directory
            vis_dir = os.path.join(args.vis_dir, f"decentralized_{args.topology}_{args.aggregation_strategy}")
            os.makedirs(vis_dir, exist_ok=True)

            # Create visualizer
            visualizer = NetworkVisualizer(output_dir=vis_dir)

            # Register visualizer with learning process
            learning_process.register_observer(visualizer)
            print(f"Registered visualizer with learning process")
            print(f"Current observers: {len(learning_process.training_monitor.observers)}")

        try:
            # Initialize the learning process
            learning_process.initialize(
                num_actors=config.num_actors,
                topology_config=config.topology,
                aggregation_config=config.aggregation,
                partitioner=partitioner,
            )

            # Print initial summary
            print("\n=== Decentralized Learning Setup ===")
            print(f"Strategy: {config.partition_strategy}")
            print(f"Clients: {config.num_actors}")
            print(f"Aggregation strategy: {config.aggregation.strategy_type}")
            print(f"Topology: {config.topology.topology_type}")
            print(f"Rounds: {args.rounds}")
            print(f"Local epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Learning rate: {args.lr}")

            print("\n=== Starting Decentralized Learning ===")
            # Execute the learning process
            _ = learning_process.execute()

            # Generate visualizations if requested
            if visualizer and (args.create_animation or args.create_frames or args.create_summary):
                print("\n=== Generating Visualizations ===")

                if args.create_animation:
                    print("Creating animation...")
                    visualizer.render_training_animation(
                        filename=f"decentralized_{args.topology}_{args.aggregation_strategy}_animation.mp4",
                        fps=args.fps
                    )

                if args.create_frames:
                    print("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"decentralized_{args.topology}_{args.aggregation_strategy}_step"
                    )

                if args.create_summary:
                    print("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"decentralized_{args.topology}_{args.aggregation_strategy}_summary.png"
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
        print(f"Decentralized Learning Process failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()