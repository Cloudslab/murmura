#!/usr/bin/env python3
"""
Example: Differential Privacy Enhanced MNIST Federated Learning

This example demonstrates how to use the differential privacy features
with the murmura framework for privacy-preserving federated learning.

It shows both Central DP (server-side noise) and Local DP (client-side noise)
configurations with comprehensive privacy monitoring.
"""

import argparse
import logging
from typing import Dict, Any

import torch
import torch.nn as nn

# Murmura framework imports
from murmura.aggregation.aggregation_config import AggregationConfig, AggregationStrategyType
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.node.resource_config import RayClusterConfig, ResourceConfig
# Differential Privacy imports
from murmura.privacy.dp_config import (
    DifferentialPrivacyConfig, DPMechanism, NoiseApplication,
    ClippingStrategy, DPAccountant
)
from murmura.privacy.dp_integration import (
    DPOrchestrationConfig, create_dp_federated_learning_process,
    create_dp_decentralized_learning_process
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
    """Set up logging configuration with privacy-specific loggers."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dp_mnist_training.log"),
        ],
    )

    # Set specific privacy logger levels
    logging.getLogger("murmura.privacy").setLevel(logging.INFO)
    logging.getLogger("murmura.privacy.accountant").setLevel(logging.DEBUG)


def create_central_dp_config(args) -> DifferentialPrivacyConfig:
    """
    Create Central Differential Privacy configuration.

    Central DP adds noise at the server after aggregation,
    providing better utility than Local DP.
    """
    return DifferentialPrivacyConfig(
        # Core privacy parameters
        epsilon=args.epsilon,
        delta=args.delta,

        # Mechanism configuration
        mechanism=DPMechanism.GAUSSIAN,
        noise_application=NoiseApplication.SERVER_SIDE,

        # Clipping configuration
        clipping_strategy=ClippingStrategy.ADAPTIVE,  # Automatic threshold adjustment
        clipping_norm=args.clipping_norm,
        target_quantile=0.5,  # Clip at median gradient norm

        # Privacy accounting
        accountant=DPAccountant.RDP,  # Rényi DP for tighter bounds

        # Client sampling for privacy amplification
        client_sampling_rate=args.client_sampling_rate,

        # Monitoring
        enable_privacy_monitoring=True,
        privacy_budget_warning_threshold=0.8,

        # Total rounds for budget allocation
        total_rounds=args.rounds,
    )


def create_local_dp_config(args) -> DifferentialPrivacyConfig:
    """
    Create Local Differential Privacy configuration.

    Local DP adds noise at each client before transmission,
    providing privacy without trusting the server.
    """
    return DifferentialPrivacyConfig(
        # Core privacy parameters - higher epsilon needed for same utility
        epsilon=args.epsilon * 2.0,  # Local DP requires more noise
        delta=args.delta,

        # Mechanism configuration
        mechanism=DPMechanism.GAUSSIAN,
        noise_application=NoiseApplication.CLIENT_SIDE,

        # Clipping configuration
        clipping_strategy=ClippingStrategy.FIXED,  # Fixed threshold for client-side
        clipping_norm=args.clipping_norm,

        # Privacy accounting
        accountant=DPAccountant.ZCDP,  # Simpler accounting for local DP

        # Per-client privacy
        per_client_clipping=True,
        max_clients_per_user=1,

        # Monitoring
        enable_privacy_monitoring=True,
        privacy_budget_warning_threshold=0.8,

        # Total rounds
        total_rounds=args.rounds,
    )


def analyze_privacy_utility_tradeoff(results: Dict[str, Any]) -> None:
    """
    Analyze and report on privacy-utility trade-offs.

    Args:
        results: Training results including privacy summary
    """
    logger = logging.getLogger("murmura.privacy.analysis")

    # Extract metrics
    initial_acc = results['initial_metrics']['accuracy']
    final_acc = results['final_metrics']['accuracy']
    accuracy_improvement = results['accuracy_improvement']

    # Extract privacy information
    if 'privacy_summary' in results:
        privacy_info = results['privacy_summary']

        logger.info("=== Privacy-Utility Analysis ===")

        # Privacy guarantees
        if 'config' in privacy_info:
            config = privacy_info['config']
            logger.info(f"Privacy Mechanism: {config['mechanism']}")
            logger.info(f"Privacy Parameters: ε={config['epsilon']}, δ={config.get('delta', 'N/A')}")
            logger.info(f"Noise Application: {config['noise_application']}")
            logger.info(f"Clipping Strategy: {config['clipping_strategy']}")

        # Privacy consumption
        if 'accountant' in privacy_info:
            accountant = privacy_info['accountant']
            spent = accountant['spent']
            total = accountant['total_budget']
            utilization = accountant['utilization']

            logger.info(f"Privacy Budget Used: ε={spent['epsilon']:.4f}/{total['epsilon']:.4f} "
                        f"({utilization['epsilon_used_pct']:.1f}%)")
            logger.info(f"Delta Budget Used: δ={spent['delta']:.2e}/{total['delta']:.2e} "
                        f"({utilization['delta_used_pct']:.1f}%)")

        # Utility impact
        logger.info(f"Model Performance:")
        logger.info(f"  Initial Accuracy: {initial_acc:.4f}")
        logger.info(f"  Final Accuracy: {final_acc:.4f}")
        logger.info(f"  Improvement: {accuracy_improvement:.4f}")

        # Privacy-utility assessment
        if accuracy_improvement > 0.1:
            utility_assessment = "Excellent - minimal impact from DP"
        elif accuracy_improvement > 0.05:
            utility_assessment = "Good - acceptable DP impact"
        elif accuracy_improvement > 0.01:
            utility_assessment = "Fair - noticeable DP impact"
        else:
            utility_assessment = "Poor - significant DP impact, consider tuning parameters"

        logger.info(f"Privacy-Utility Assessment: {utility_assessment}")

        # Recommendations
        logger.info("=== Recommendations ===")
        if 'config' in privacy_info:
            epsilon = privacy_info['config']['epsilon']
            if epsilon < 0.1:
                logger.info("- Very strong privacy (ε < 0.1) may significantly impact utility")
                logger.info("- Consider increasing ε slightly or using privacy amplification techniques")
            elif epsilon > 10.0:
                logger.info("- Weak privacy (ε > 10.0) provides limited privacy protection")
                logger.info("- Consider decreasing ε for stronger privacy guarantees")

            if privacy_info['config']['noise_application'] == 'client_side':
                logger.info("- Using Local DP - consider Central DP for better utility if server can be trusted")
            else:
                logger.info("- Using Central DP - good utility-privacy balance")


def main():
    """Main function for DP-enhanced MNIST federated learning."""
    parser = argparse.ArgumentParser(
        description="Differential Privacy Enhanced MNIST Federated Learning"
    )

    # Core federated learning arguments
    parser.add_argument("--num_actors", type=int, default=5,
                        help="Number of virtual clients")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of federated learning rounds")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    # Differential Privacy arguments
    parser.add_argument("--dp_mode", type=str, choices=["central", "local", "none"],
                        default="central", help="Differential privacy mode")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Privacy budget (ε). Lower = stronger privacy")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Failure probability (δ)")
    parser.add_argument("--clipping_norm", type=float, default=1.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--client_sampling_rate", type=float, default=0.1,
                        help="Client sampling rate for privacy amplification")

    # Learning configuration
    parser.add_argument("--topology", type=str, default="star",
                        choices=["star", "ring", "complete"],
                        help="Network topology")
    parser.add_argument("--aggregation", type=str, default="fedavg",
                        choices=["fedavg", "trimmed_mean"],
                        help="Aggregation strategy")
    parser.add_argument("--partition_strategy", type=str, default="dirichlet",
                        choices=["dirichlet", "iid"],
                        help="Data partitioning strategy")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter")

    # System configuration
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--save_path", type=str, default="dp_mnist_model.pt",
                        help="Path to save final model")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.dp_mnist_example")

    try:
        # Create differential privacy configuration
        dp_config = None
        if args.dp_mode == "central":
            dp_config = create_central_dp_config(args)
            logger.info("Using Central Differential Privacy (server-side noise)")
        elif args.dp_mode == "local":
            dp_config = create_local_dp_config(args)
            logger.info("Using Local Differential Privacy (client-side noise)")
        else:
            logger.info("Differential Privacy disabled")

        # Create enhanced orchestration configuration
        config = DPOrchestrationConfig(
            # Core FL parameters
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            split="train",

            # Network topology
            topology=TopologyConfig(
                topology_type=TopologyType(args.topology),
                hub_index=0
            ),

            # Aggregation strategy
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType(args.aggregation)
            ),

            # Dataset configuration
            dataset_name="mnist",
            feature_columns=["image"],
            label_column="label",

            # Ray cluster configuration
            ray_cluster=RayClusterConfig(
                namespace="murmura_dp_mnist",
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

        logger.info("=== Setting Up DP-Enhanced Learning Process ===")
        # Create appropriate learning process based on topology
        if args.topology in ["star", "complete"] or args.dp_mode == "central":
            # Federated learning with Central DP
            learning_process = create_dp_federated_learning_process(
                config, train_dataset, global_model
            )
            logger.info("Created DP-enhanced Federated Learning Process")
        else:
            # Decentralized learning with Local DP
            if args.dp_mode == "central":
                logger.warning("Forcing Local DP for decentralized topology")
                config.differential_privacy.noise_application = NoiseApplication.CLIENT_SIDE

            learning_process = create_dp_decentralized_learning_process(
                config, train_dataset, global_model
            )
            logger.info("Created DP-enhanced Decentralized Learning Process")

        # Initialize the learning process
        partitioner = PartitionerFactory.create(config)

        learning_process.initialize(
            num_actors=config.num_actors,
            topology_config=config.topology,
            aggregation_config=config.aggregation,
            partitioner=partitioner
        )

        # Log privacy configuration
        if dp_config:
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
        logger.info(f"Topology: {args.topology}")
        logger.info(f"Aggregation: {args.aggregation}")
        logger.info(f"DP Mode: {args.dp_mode}")

        # Set training parameters
        config_dict = {
            "rounds": args.rounds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_split": "test"
        }

        # Execute training
        results = learning_process.execute()

        # Analyze privacy-utility trade-off
        analyze_privacy_utility_tradeoff(results)

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
