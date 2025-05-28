#!/usr/bin/env python3
"""
Fixed DP MNIST Example - addresses the "Model is not set" error
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
    DPOrchestrationConfig, create_dp_federated_learning_process
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
            logging.FileHandler("dp_mnist_training_fixed.log"),
        ],
    )


def create_dp_config(args) -> DifferentialPrivacyConfig:
    """Create differential privacy configuration."""
    return DifferentialPrivacyConfig(
        # Core privacy parameters
        epsilon=args.epsilon,
        delta=args.delta,

        # Mechanism configuration
        mechanism=DPMechanism.GAUSSIAN,
        noise_application=NoiseApplication.SERVER_SIDE,  # Central DP

        # Clipping configuration
        clipping_strategy=ClippingStrategy.ADAPTIVE,
        clipping_norm=args.clipping_norm,
        target_quantile=0.5,

        # Privacy accounting
        accountant=DPAccountant.RDP,

        # Client sampling for privacy amplification
        client_sampling_rate=args.client_sampling_rate,

        # Monitoring
        enable_privacy_monitoring=True,
        privacy_budget_warning_threshold=0.8,

        # Total rounds for budget allocation
        total_rounds=args.rounds,
    )


def main():
    """Main function for fixed DP-enhanced MNIST federated learning."""
    parser = argparse.ArgumentParser(
        description="Fixed Differential Privacy Enhanced MNIST Federated Learning"
    )

    # Core federated learning arguments
    parser.add_argument("--num_actors", type=int, default=5,
                        help="Number of virtual clients")
    parser.add_argument("--rounds", type=int, default=5,  # Reduced for testing
                        help="Number of federated learning rounds")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    # Differential Privacy arguments
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Privacy budget (ε)")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Failure probability (δ)")
    parser.add_argument("--clipping_norm", type=float, default=1.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--client_sampling_rate", type=float, default=0.1,
                        help="Client sampling rate")

    # System configuration
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--save_path", type=str, default="dp_mnist_model_fixed.pt",
                        help="Path to save final model")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.dp_mnist_example_fixed")

    try:
        # Create DP configuration
        dp_config = create_dp_config(args)
        logger.info("Created DP configuration")

        # Create enhanced orchestration configuration
        from murmura.aggregation.aggregation_config import AggregationConfig, AggregationStrategyType
        from murmura.network_management.topology import TopologyConfig, TopologyType
        from murmura.node.resource_config import RayClusterConfig, ResourceConfig

        config = DPOrchestrationConfig(
            # Core FL parameters
            num_actors=args.num_actors,
            partition_strategy="dirichlet",
            alpha=0.5,
            split="train",

            # CRITICAL: Must specify feature and label columns
            feature_columns=["image"],  # MNIST images
            label_column="label",       # MNIST labels

            # Network topology (centralized for Central DP)
            topology=TopologyConfig(
                topology_type=TopologyType.STAR,
                hub_index=0
            ),

            # Aggregation strategy
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType.FEDAVG
            ),

            # Dataset configuration
            dataset_name="mnist",

            # Ray cluster configuration
            ray_cluster=RayClusterConfig(
                namespace="murmura_dp_mnist_fixed",
                logging_level=args.log_level
            ),

            resources=ResourceConfig(
                cpus_per_actor=1.0,
                placement_strategy="spread"
            ),

            # Differential Privacy configuration
            differential_privacy=dp_config,
            enable_privacy_dashboard=True
        )

        logger.info("Created orchestration configuration")

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

        logger.info("=== Setting Up DP-Enhanced Learning Process ===")
        learning_process = create_dp_federated_learning_process(
            config, train_dataset, global_model
        )
        logger.info("Created DP-enhanced learning process")

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

        logger.info("=== Starting DP-Enhanced Training ===")
        logger.info(f"Clients: {args.num_actors}")
        logger.info(f"Rounds: {args.rounds}")
        logger.info(f"Topology: star")
        logger.info(f"Aggregation: fedavg")
        logger.info(f"DP Mode: central")

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
                logger.info("Differential Privacy: ENABLED")
                if 'accountant' in privacy_summary:
                    spent = privacy_summary['accountant']['spent']
                    logger.info(f"Privacy Spent: ε={spent['epsilon']:.4f}, δ={spent['delta']:.2e}")
            else:
                logger.info("Differential Privacy: DISABLED")

    except Exception as e:
        logger.error(f"DP-enhanced training failed: {e}")
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
