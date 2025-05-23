from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np

from murmura.aggregation.aggregation_config import AggregationConfig
from murmura.data_processing.dataset import MDataset
from murmura.data_processing.partitioner import Partitioner
from murmura.model.model_interface import ModelInterface
from murmura.network_management.topology import TopologyConfig
from murmura.orchestration.cluster_manager import ClusterManager
from murmura.privacy.privacy_config import PrivacyConfig
from murmura.visualization.training_event import InitialStateEvent, PrivacyEvent
from murmura.visualization.training_observer import TrainingMonitor


class LearningProcess(ABC):
    """
    Abstract base class for learning processes.
    Defines a common interface for different learning process and implements common functionality.
    """

    def __init__(
        self, config: Dict[str, Any], dataset: MDataset, model: ModelInterface
    ):
        """
        Initialize the learning process.

        :param config: Configuration dictionary
        :param dataset: Dataset instance
        :param model: Model instance
        """
        self.config = config
        self.dataset = dataset
        self.model = model
        self.cluster_manager: Optional[ClusterManager] = None
        self.training_monitor = TrainingMonitor()

    def register_observer(self, observer) -> None:
        """
        Register an observer to receive training events.

        :param observer: The observer to register.
        """
        self.training_monitor.register_observer(observer)

    def initialize(
        self,
        num_actors: int,
        topology_config: TopologyConfig,
        aggregation_config: AggregationConfig,
        partitioner: Partitioner,
        privacy_config: Optional[PrivacyConfig] = None,
    ) -> None:
        """
        Initialize the learning process by setting up the cluster, creating actors, and distributing data and models.

        :param num_actors: Number of actors to create
        :param topology_config: Topology configuration
        :param aggregation_config: Aggregation configuration
        :param partitioner: Partitioner instance
        :param privacy_config: Optional privacy configuration
        """
        print("\n=== Initializing Learning Process ===")

        # Initialize the cluster manager
        self.cluster_manager = ClusterManager(self.config)

        print("Setting up aggregation strategy...")
        self.cluster_manager.set_aggregation_strategy(aggregation_config)

        # Partition the dataset and get data information before initializing privacy
        print("Partitioning dataset...")
        split = self.config.get("split", "train")
        partitioner.partition(self.dataset, split)
        partitions = list(self.dataset.get_partitions(split).values())
        self.config["data_partitions"] = partitions
        self.config["batch_size"] = self.config.get("batch_size", 32)

        # Calculate total samples and batch size for privacy accounting
        total_samples = sum(len(p) for p in partitions)
        batch_size = self.config.get("batch_size", 32)

        # Pass this information to the cluster manager config
        self.cluster_manager.config["data_partitions"] = partitions
        self.cluster_manager.config["batch_size"] = batch_size
        self.cluster_manager.config["total_samples"] = total_samples
        self.cluster_manager.config["epochs"] = self.config.get("epochs", 1)
        self.cluster_manager.config['learning_rate'] = self.config.get('learning_rate', 0.001)

        # Create the actors first so privacy manager has client count
        print(
            f"Creating {num_actors} virtual clients with {topology_config.topology_type.value} topology..."
        )
        self.cluster_manager.create_actors(num_actors, topology_config)

        # Set up privacy after creating actors but before data distribution
        if privacy_config:
            print("Setting up privacy manager...")
            # Update privacy config with dataset information
            if privacy_config.params is None:
                privacy_config.params = {}

            privacy_config.params["total_samples"] = total_samples
            privacy_config.params["batch_size"] = batch_size
            privacy_config.params["num_actors"] = num_actors
            privacy_config.params["epochs"] = self.config.get("epochs", 1)
            privacy_config.params["learning_rate"] = self.config.get('learning_rate', 0.001)

            # Initialize privacy manager
            self.cluster_manager.set_privacy_manager(privacy_config)

            # Explicitly call setup_privacy_accounting
            if (
                hasattr(self.cluster_manager, "privacy_manager")
                and self.cluster_manager.privacy_manager
            ):
                privacy_manager = self.cluster_manager.privacy_manager
                if hasattr(privacy_manager, "setup_privacy_accounting"):
                    print(
                        f"Initializing privacy accounting with {total_samples} samples and batch size {batch_size}"
                    )
                    privacy_manager.setup_privacy_accounting(total_samples, batch_size)

            # Emit privacy event
            if privacy_config.enabled:
                privacy_info = {
                    "enabled": privacy_config.enabled,
                    "mode": privacy_config.privacy_mode.value,
                    "mechanism": privacy_config.mechanism_type.value,
                    "target_epsilon": privacy_config.target_epsilon,
                    "target_delta": privacy_config.target_delta,
                    "noise_multiplier": privacy_config.noise_multiplier or 1.0,
                }
                self.training_monitor.emit_event(
                    PrivacyEvent(
                        round_num=0,
                        privacy_info=privacy_info,
                    )
                )

        self.training_monitor.emit_event(
            InitialStateEvent(
                topology_type=topology_config.topology_type.value, num_nodes=num_actors
            )
        )

        for observer in self.training_monitor.observers:
            if hasattr(observer, "set_topology"):
                observer.set_topology(self.cluster_manager.topology_manager)

        print("Distributing data partitions...")
        self.cluster_manager.distribute_data(
            partitions,
            metadata={
                "split": split,
                "dataset": self.config.get("dataset_name", "unknown"),
                "topology": topology_config.topology_type.value,
            },
        )

        print("Distributing dataset...")
        feature_columns = self.config.get("feature_columns", None)
        label_column = self.config.get("label_column", None)
        self.cluster_manager.distribute_dataset(
            self.dataset, feature_columns=feature_columns, label_column=label_column
        )

        print("Distributing model...")
        self.cluster_manager.distribute_model(self.model)

        print("Learning process initialized successfully.")

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the learning process.

        :return: Results of the learning process
        """
        pass

    def shutdown(self) -> None:
        """
        Shutdown the learning process and clean up resources.
        """
        if self.cluster_manager:
            self.cluster_manager.shutdown()
            print("Learning process shut down successfully.")
        else:
            print("No cluster manager to shut down.")

    @staticmethod
    def _calculate_parameter_convergence(
        node_params: Dict[int, Dict[str, Any]], global_params: Dict[str, Any]
    ) -> float:
        """
        Calculate a measure of parameter convergence across nodes.

        :param node_params: Dictionary mapping node indices to their parameters.
        :param global_params: Dictionary of global parameters.

        :return: A float representing the average distance of node parameters from the global parameters. Lower the
        better.
        """
        distances = []

        for node_idx, params in node_params.items():
            distance = 0.0
            for key in params:
                if key in global_params:
                    # L2 norm of parameter difference
                    diff = params[key] - global_params[key]
                    distance += float(np.sum(diff * diff))

            if distance > 0:
                distances.append(np.sqrt(distance))

        # Average distance
        return float(np.mean(distances)) if distances else 0.0

    @staticmethod
    def _create_parameter_summaries(
        node_params: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Create summaries of model parameters for visualization.

        :param node_params: Dictionary mapping node indices to their parameters.

        :return: A dictionary containing summaries of model parameters for each node.
        """
        param_summaries = {}

        for node_idx, params in node_params.items():
            if not params:
                continue

            # Choose a representative parameter (first layer is usually good)
            first_key = next(iter(params))

            param_summaries[node_idx] = {
                "norm": float(np.linalg.norm(params[first_key])),
                "mean": float(np.mean(params[first_key])),
                "std": float(np.std(params[first_key])),
            }

        return param_summaries
