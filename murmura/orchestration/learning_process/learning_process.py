import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

import numpy as np

from murmura.aggregation.aggregation_config import AggregationConfig
from murmura.data_processing.dataset import MDataset
from murmura.data_processing.partitioner import Partitioner
from murmura.model.model_interface import ModelInterface
from murmura.network_management.topology import TopologyConfig
from murmura.orchestration.cluster_manager import ClusterManager
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.visualization.training_event import InitialStateEvent
from murmura.visualization.training_observer import TrainingMonitor


class LearningProcess(ABC):
    """
    Enhanced abstract base class for learning processes with multi-node support.
    Defines a common interface for different learning processes and implements common functionality.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], OrchestrationConfig],
        dataset: MDataset,
        model: ModelInterface,
    ):
        """
        Initialize the learning process.

        :param config: Configuration dictionary or OrchestrationConfig
        :param dataset: Dataset instance
        :param model: Model instance
        """
        # Handle both dict and OrchestrationConfig inputs
        if isinstance(config, OrchestrationConfig):
            self.config = config
            self.config_dict = config.model_dump()
        else:
            self.config_dict = config
            # For backward compatibility, create OrchestrationConfig if we have dict
            self.config = self._create_orchestration_config_from_dict(config)

        self.dataset = dataset
        self.model = model
        self.cluster_manager: Optional[ClusterManager] = None
        self.training_monitor = TrainingMonitor()

        # Set up logging for the learning process
        self.logger = logging.getLogger(
            f"murmura.learning_process.{self.__class__.__name__}"
        )

    def register_observer(self, observer) -> None:
        """
        Register an observer to receive training events.

        :param observer: The observer to register.
        """
        self.training_monitor.register_observer(observer)
        self.logger.debug(f"Registered observer: {observer.__class__.__name__}")

    def initialize(
        self,
        num_actors: int,
        topology_config: TopologyConfig,
        aggregation_config: AggregationConfig,
        partitioner: Partitioner,
    ) -> None:
        """
        Initialize the learning process by setting up the cluster, creating actors, and distributing data and models.

        :param num_actors: Number of actors to create
        :param topology_config: Topology configuration
        :param aggregation_config: Aggregation configuration
        :param partitioner: Partitioner instance
        """
        self.logger.info("=== Initializing Learning Process ===")

        # Update config with actor count if needed
        if hasattr(self.config, "num_actors"):
            self.config.num_actors = num_actors

        # Initialize the cluster manager with enhanced config
        self.cluster_manager = ClusterManager(self.config)

        self.logger.info("Setting up aggregation strategy...")
        self.cluster_manager.set_aggregation_strategy(aggregation_config)

        self.logger.info(
            f"Creating {num_actors} virtual clients with {topology_config.topology_type.value} topology..."
        )

        # Log cluster information
        cluster_stats = self.cluster_manager.get_cluster_stats()
        self.logger.info(
            f"Cluster stats: {cluster_stats['cluster_info']['total_nodes']} nodes, "
            f"Multi-node: {cluster_stats['cluster_info']['is_multinode']}"
        )

        self.cluster_manager.create_actors(num_actors, topology_config)

        # Emit initial state event
        self.training_monitor.emit_event(
            InitialStateEvent(
                topology_type=topology_config.topology_type.value, num_nodes=num_actors
            )
        )

        # Set topology for observers
        for observer in self.training_monitor.observers:
            if hasattr(observer, "set_topology"):
                observer.set_topology(self.cluster_manager.topology_manager)

        self.logger.info("Partitioning dataset...")
        split = self.get_config_value("split", "train")
        partitioner.partition(self.dataset, split)

        self.logger.info("Distributing data partitions...")
        partitions = list(self.dataset.get_partitions(split).values())
        self.cluster_manager.distribute_data(
            partitions,
            metadata={
                "split": split,
                "dataset": self.get_config_value("dataset_name", "unknown"),
                "topology": topology_config.topology_type.value,
                "cluster_nodes": cluster_stats["cluster_info"]["total_nodes"],
                "is_multinode": cluster_stats["cluster_info"]["is_multinode"],
            },
        )

        self.logger.info("Distributing dataset...")
        feature_columns = self.get_config_value("feature_columns", None)
        label_column = self.get_config_value("label_column", None)
        self.cluster_manager.distribute_dataset(
            self.dataset, feature_columns=feature_columns, label_column=label_column
        )

        self.logger.info("Distributing model...")
        self.cluster_manager.distribute_model(self.model)

        self.logger.info("Learning process initialized successfully.")

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value, supporting both dict and OrchestrationConfig

        :param key: Configuration key
        :param default: Default value if key not found
        :return: Configuration value
        """
        if isinstance(self.config, OrchestrationConfig):
            # Handle OrchestrationConfig objects
            if hasattr(self.config, key):
                return getattr(self.config, key)
            elif key in self.config_dict:
                return self.config_dict[key]
            else:
                return default
        else:
            # Handle dictionary configs (backward compatibility)
            return self.config_dict.get(key, default)

    def _create_orchestration_config_from_dict(
        self, config_dict: Dict[str, Any]
    ) -> OrchestrationConfig:
        """
        Create OrchestrationConfig from legacy dict config for backward compatibility
        """
        from murmura.orchestration.orchestration_config import (
            RayClusterConfig,
            ResourceConfig,
        )
        from murmura.aggregation.aggregation_config import (
            AggregationConfig,
            AggregationStrategyType,
        )
        from murmura.network_management.topology import TopologyConfig, TopologyType

        # Extract Ray cluster config
        ray_cluster_config = RayClusterConfig()
        if "ray_address" in config_dict:
            ray_cluster_config.address = config_dict["ray_address"]

        # Extract resource config
        resource_config = ResourceConfig()

        # Extract aggregation config if not provided separately
        aggregation_config = AggregationConfig()
        if "aggregation_strategy" in config_dict:
            aggregation_config.strategy_type = AggregationStrategyType(
                config_dict["aggregation_strategy"]
            )

        # Extract topology config if not provided separately
        topology_config = TopologyConfig()
        if "topology" in config_dict:
            topology_config.topology_type = TopologyType(config_dict["topology"])

        # Create orchestration config
        orchestration_config = OrchestrationConfig(
            num_actors=config_dict.get("num_actors", 10),
            topology=topology_config,
            aggregation=aggregation_config,
            dataset_name=config_dict.get("dataset_name", "unknown"),
            partition_strategy=config_dict.get("partition_strategy", "dirichlet"),
            alpha=config_dict.get("alpha", 0.5),
            min_partition_size=config_dict.get("min_partition_size", 100),
            split=config_dict.get("split", "train"),
            ray_cluster=ray_cluster_config,
            resources=resource_config,
        )

        return orchestration_config

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
        self.logger.info("Shutting down learning process...")

        if self.cluster_manager:
            try:
                self.cluster_manager.shutdown()
                self.logger.info("Learning process shut down successfully.")
            except Exception as e:
                self.logger.error(f"Error during cluster manager shutdown: {e}")
        else:
            self.logger.info("No cluster manager to shut down.")

    def get_actor_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all actors in the cluster

        :return: Health status summary
        """
        if not self.cluster_manager:
            return {"error": "Cluster manager not initialized"}

        try:
            import ray

            # Sample health check from a few actors to avoid overwhelming the cluster
            sample_size = min(5, len(self.cluster_manager.actors))
            sample_actors = self.cluster_manager.actors[:sample_size]

            health_checks = []
            for actor in sample_actors:
                try:
                    health = ray.get(actor.health_check.remote(), timeout=10)
                    health_checks.append(health)
                except Exception as e:
                    health_checks.append(
                        {"status": "error", "error": f"Health check failed: {e}"}
                    )

            # Summarize health status
            healthy_count = sum(
                1 for h in health_checks if h.get("status") == "healthy"
            )
            degraded_count = sum(
                1 for h in health_checks if h.get("status") == "degraded"
            )
            error_count = sum(1 for h in health_checks if h.get("status") == "error")

            return {
                "total_actors": len(self.cluster_manager.actors),
                "sampled_actors": sample_size,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "error": error_count,
                "health_checks": health_checks,
            }

        except Exception as e:
            self.logger.error(f"Failed to get actor health status: {e}")
            return {"error": str(e)}

    def get_cluster_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the cluster state

        :return: Cluster summary
        """
        if not self.cluster_manager:
            return {"error": "Cluster manager not initialized"}

        try:
            cluster_stats = self.cluster_manager.get_cluster_stats()
            topology_info = self.cluster_manager.get_topology_information()

            summary = {
                "cluster_type": "multi-node"
                if cluster_stats["cluster_info"]["is_multinode"]
                else "single-node",
                "total_nodes": cluster_stats["cluster_info"]["total_nodes"],
                "total_actors": cluster_stats["num_actors"],
                "topology": topology_info.get("type", "unknown"),
                "placement_strategy": cluster_stats["placement_strategy"],
                "has_placement_group": cluster_stats["has_placement_group"],
                "available_resources": cluster_stats["cluster_info"].get(
                    "available_resources", {}
                ),
                "resource_config": cluster_stats["resource_config"],
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get cluster summary: {e}")
            return {"error": str(e)}

    @staticmethod
    def _calculate_parameter_convergence(
        node_params: Dict[int, Dict[str, Any]], global_params: Dict[str, Any]
    ) -> float:
        """
        Calculate a measure of parameter convergence across nodes.

        :param node_params: Dictionary mapping node indices to their parameters.
        :param global_params: Dictionary of global parameters.

        :return: A float representing the average distance of node parameters from the global parameters.
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

    def log_training_progress(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """
        Log training progress with cluster context

        :param round_num: Current round number
        :param metrics: Training metrics to log
        """
        cluster_info = ""
        if self.cluster_manager:
            stats = self.cluster_manager.get_cluster_stats()
            cluster_type = (
                "multi-node" if stats["cluster_info"]["is_multinode"] else "single-node"
            )
            cluster_info = (
                f"[{cluster_type}:{stats['cluster_info']['total_nodes']} nodes] "
            )

        # Log key metrics
        metric_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
        )

        self.logger.info(f"{cluster_info}Round {round_num} - {metric_str}")

    def monitor_resource_usage(self) -> Dict[str, Any]:
        """
        Monitor resource usage across the cluster

        :return: Resource usage summary
        """
        if not self.cluster_manager:
            return {"error": "Cluster manager not initialized"}

        try:
            import ray

            # Get cluster-wide resource information
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()

            # Calculate resource utilization
            resource_utilization = {}
            for resource, total in cluster_resources.items():
                available = available_resources.get(resource, 0)
                used = total - available
                utilization_pct = (used / total * 100) if total > 0 else 0
                resource_utilization[resource] = {
                    "total": total,
                    "used": used,
                    "available": available,
                    "utilization_percent": utilization_pct,
                }

            return {
                "timestamp": ray.util.get_current_time_ms(),
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
                "resource_utilization": resource_utilization,
            }

        except Exception as e:
            self.logger.error(f"Failed to monitor resource usage: {e}")
            return {"error": str(e)}
