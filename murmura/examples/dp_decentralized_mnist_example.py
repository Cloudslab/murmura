#!/usr/bin/env python3
"""
DP-Enhanced Decentralized MNIST Example - Local Differential Privacy for Decentralized Learning
"""

import argparse
import logging

import torch
import torch.nn as nn

from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
from murmura.privacy.dp_config import (
    DifferentialPrivacyConfig, DPMechanism, NoiseApplication,
    ClippingStrategy, DPAccountant
)
from murmura.privacy.dp_integration import (
    DPOrchestrationConfig, create_dp_decentralized_learning_process
)


class MNISTModel(PyTorchModel):
    """Simple CNN model for MNIST classification."""

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
            nn.Linear(128, 10)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dp_decentralized_mnist.log"),
        ],
    )


def create_dp_config(args) -> DifferentialPrivacyConfig:
    """Create differential privacy configuration for decentralized learning."""
    return DifferentialPrivacyConfig(
        # Core privacy parameters
        epsilon=args.epsilon,
        delta=args.delta,

        # CRITICAL: Decentralized learning requires Local DP (client-side noise)
        mechanism=DPMechanism.GAUSSIAN,
        noise_application=NoiseApplication.CLIENT_SIDE,  # Local DP for decentralized

        # Clipping configuration
        clipping_strategy=ClippingStrategy.ADAPTIVE,
        clipping_norm=args.clipping_norm,
        target_quantile=0.5,

        # Privacy accounting
        accountant=DPAccountant.RDP,

        # Client sampling for privacy amplification (less effective with Local DP)
        client_sampling_rate=args.client_sampling_rate,

        # Monitoring
        enable_privacy_monitoring=True,
        privacy_budget_warning_threshold=0.8,

        # Total rounds for budget allocation
        total_rounds=args.rounds,
    )


def main():
    """Main function for DP-enhanced decentralized MNIST learning."""
    parser = argparse.ArgumentParser(
        description="DP-Enhanced Decentralized MNIST Learning (Local DP)"
    )

    # Core decentralized learning arguments
    parser.add_argument("--num_actors", type=int, default=8,
                        help="Number of virtual clients")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of decentralized learning rounds")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    # Decentralized topology arguments (only decentralized-compatible topologies)
    parser.add_argument("--topology", type=str, default="ring",
                        choices=["ring", "complete", "line", "custom"],
                        help="Network topology (decentralized-compatible only)")
    parser.add_argument("--aggregation_strategy", type=str, default="gossip_avg",
                        choices=["gossip_avg"],
                        help="Aggregation strategy (decentralized only)")
    parser.add_argument("--mixing_parameter", type=float, default=0.5,
                        help="Mixing parameter for gossip averaging")

    # Differential Privacy arguments
    parser.add_argument("--epsilon", type=float, default=2.0,
                        help="Privacy budget per client per round (ε) - higher for Local DP")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Failure probability (δ)")
    parser.add_argument("--clipping_norm", type=float, default=1.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--client_sampling_rate", type=float, default=0.5,
                        help="Client sampling rate (less effective with Local DP)")

    # Data partitioning
    parser.add_argument("--partition_strategy", type=str, default="dirichlet",
                        choices=["dirichlet", "iid"])
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter")

    # System configuration
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--save_path", type=str, default="dp_decentralized_mnist_model.pt",
                        help="Path to save final model")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.dp_decentralized_mnist")

    try:
        # Create DP configuration for decentralized learning
        dp_config = create_dp_config(args)
        logger.info("Created Local DP configuration for decentralized learning")

        # Create enhanced orchestration configuration
        from murmura.aggregation.aggregation_config import AggregationConfig, AggregationStrategyType
        from murmura.network_management.topology import TopologyConfig, TopologyType
        from murmura.node.resource_config import RayClusterConfig, ResourceConfig

        config = DPOrchestrationConfig(
            # Core FL parameters
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            split="train",

            # CRITICAL: Must specify feature and label columns
            feature_columns=["image"],  # MNIST images
            label_column="label",       # MNIST labels

            # Network topology (decentralized-compatible)
            topology=TopologyConfig(
                topology_type=TopologyType(args.topology.upper()),
                hub_index=0  # Not used for decentralized topologies
            ),

            # Aggregation strategy (decentralized only)
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType.GOSSIP_AVG,
                params={"mixing_parameter": args.mixing_parameter}
            ),

            # Dataset configuration
            dataset_name="mnist",

            # Ray cluster configuration
            ray_cluster=RayClusterConfig(
                namespace="murmura_dp_decentralized_mnist",
                logging_level=args.log_level
            ),

            resources=ResourceConfig(
                cpus_per_actor=1.0,
                placement_strategy="spread"
            ),

            # Differential Privacy configuration (Local DP)
            differential_privacy=dp_config,
            enable_privacy_dashboard=True
        )

        logger.info("Created DP orchestration configuration")

        # Validate that we're using Local DP for decentralized learning
        if not dp_config.is_local_dp():
            raise ValueError(
                "Decentralized learning requires Local DP (client-side noise). "
                "The configuration has been set to use CLIENT_SIDE noise application."
            )

        logger.info("=== Loading MNIST Dataset ===")
        train_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="mnist",
            split="train"
        )

        test_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name="mnist",
            split="test"
        )

        train_dataset.merge_splits(test_dataset)
        logger.info("Loaded and merged MNIST dataset")

        logger.info("=== Creating MNIST Model ===")
        model = MNISTModel()
        input_shape = (1, 28, 28)

        global_model = TorchModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": args.lr},
            input_shape=input_shape,
        )
        logger.info("Created model wrapper")

        logger.info("=== Setting Up DP-Enhanced Decentralized Learning Process ===")
        learning_process = create_dp_decentralized_learning_process(
            config, train_dataset, global_model
        )
        logger.info("Created DP-enhanced decentralized learning process")

        # Initialize the learning process
        partitioner = PartitionerFactory.create(config)
        logger.info("Created partitioner")

        logger.info("=== Initializing Learning Process ===")
        learning_process.initialize(
            num_actors=config.num_actors,
            topology_config=config.topology,
            aggregation_config=config.aggregation,
            partitioner=partitioner
        )

        # Log privacy configuration
        privacy_desc = dp_config.get_privacy_description()
        utility_impact = dp_config.estimate_utility_impact()

        logger.info("=== Privacy Configuration ===")
        logger.info(f"Privacy Guarantee: {privacy_desc}")
        logger.info(f"Privacy Level: {utility_impact['privacy_level']}")
        logger.info(f"Expected Utility Impact: {utility_impact['utility_impact']}")
        logger.info(f"Communication Impact: {utility_impact['communication']}")
        logger.info(f"DP Mode: Local (client-side noise)")

        logger.info("=== Starting DP-Enhanced Decentralized Training ===")
        logger.info(f"Clients: {args.num_actors}")
        logger.info(f"Rounds: {args.rounds}")
        logger.info(f"Topology: {args.topology}")
        logger.info(f"Aggregation: {args.aggregation_strategy}")
        logger.info(f"Mixing Parameter: {args.mixing_parameter}")

        # Execute training
        results = learning_process.execute()

        # Save model
        logger.info("=== Saving Model ===")
        global_model.save(args.save_path)
        logger.info(f"Model saved to {args.save_path}")

        # Print final summary
        logger.info("=== Training Summary ===")
        logger.info(f"Initial Accuracy: {results['initial_metrics']['accuracy']:.4f}")
        logger.info(f"Final Accuracy: {results['final_metrics']['accuracy']:.4f}")
        logger.info(f"Accuracy Improvement: {results['accuracy_improvement']:.4f}")

        if 'privacy_summary' in results:
            privacy_summary = results['privacy_summary']
            if privacy_summary['dp_enabled']:
                logger.info("Differential Privacy: ENABLED (Local DP)")
                if 'accountant' in privacy_summary:
                    spent = privacy_summary['accountant']['spent']
                    logger.info(f"Privacy Spent: ε={spent['epsilon']:.4f}, δ={spent['delta']:.2e}")
            else:
                logger.info("Differential Privacy: DISABLED")

        # Log topology-specific results
        if 'topology' in results:
            topology_info = results['topology']
            logger.info(f"Network connections: {len(topology_info.get('adjacency_list', {}))}")

    except Exception as e:
        logger.error(f"DP-enhanced decentralized training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Clean up
        if 'learning_process' in locals():
            learning_process.shutdown()
        logger.info("Training completed and resources cleaned up")


if __name__ == "__main__":
    main()
