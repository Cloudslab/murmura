import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import ray

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.aggregation.strategy_factory import AggregationStrategyFactory
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.data_processing.dataset import MDataset
from murmura.model.model_interface import ModelInterface
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.network_management.topology_compatibility import (
    TopologyCompatibilityManager,
)
from murmura.network_management.topology_manager import TopologyManager
from murmura.node.client_actor import VirtualClientActor
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.orchestration.topology_coordinator import TopologyCoordinator


class ClusterManager:
    """
    Enhanced cluster manager with support for multi-node Ray clusters
    """

    def __init__(self, config: OrchestrationConfig) -> None:
        self.config = config
        self.actors: List[Any] = []
        self.topology_manager: Optional[TopologyManager] = None
        self.aggregation_strategy: Optional[AggregationStrategy] = None
        self.topology_coordinator: Optional[TopologyCoordinator] = None
        self.cluster_info: Dict[str, Any] = {}

        # Set up logging
        self._setup_logging()

        # Initialize Ray cluster
        self._initialize_ray_cluster()

        # Gather cluster information
        self._gather_cluster_info()

    def _setup_logging(self) -> None:
        """Set up distributed logging compatible with multi-node clusters"""
        log_level = getattr(logging, self.config.ray_cluster.logging_level)

        # Configure Ray logging
        ray_logger = logging.getLogger("ray")
        ray_logger.setLevel(log_level)

        # Configure framework logging
        logger = logging.getLogger("murmura")
        logger.setLevel(log_level)

        # Create formatter for distributed logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Add console handler if not exists
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    def _initialize_ray_cluster(self) -> None:
        """Initialize Ray cluster with multi-node support"""
        if ray.is_initialized():
            logging.getLogger("murmura").info("Ray already initialized, using existing cluster")
            return

        ray_config = {
            "namespace": self.config.ray_cluster.namespace,
            "include_dashboard": self.config.ray_cluster.include_dashboard,
            "logging_level": getattr(logging, self.config.ray_cluster.logging_level),
        }

        # Add address if specified
        if self.config.ray_cluster.address:
            ray_config["address"] = self.config.ray_cluster.address

        # Add runtime environment if specified
        if self.config.ray_cluster.runtime_env:
            ray_config["runtime_env"] = self.config.ray_cluster.runtime_env

        # Auto-detect cluster setup if enabled
        if self.config.ray_cluster.auto_detect_cluster and not self.config.ray_cluster.address:
            # Check for Ray cluster environment variables
            if "RAY_ADDRESS" in os.environ:
                ray_config["address"] = os.environ["RAY_ADDRESS"]
                logging.getLogger("murmura").info(
                    f"Auto-detected Ray cluster address: {ray_config['address']}"
                )

        # Initialize Ray
        try:
            ray.init(**{k: v for k, v in ray_config.items() if v is not None})
            logging.getLogger("murmura").info("Ray cluster initialized successfully")
        except Exception as e:
            logging.getLogger("murmura").error(f"Failed to initialize Ray cluster: {e}")
            raise

    def _gather_cluster_info(self) -> None:
        """Gather information about the Ray cluster"""
        try:
            # Get cluster resources
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()

            # Get node information
            nodes = ray.nodes()

            self.cluster_info = {
                "total_nodes": len([n for n in nodes if n["Alive"]]),
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
                "nodes": nodes,
                "is_multinode": len([n for n in nodes if n["Alive"]]) > 1
            }

            # Log cluster information
            logger = logging.getLogger("murmura")
            logger.info(f"Ray cluster info: {self.cluster_info['total_nodes']} nodes")
            logger.info(f"Total CPUs: {cluster_resources.get('CPU', 0)}")
            logger.info(f"Total GPUs: {cluster_resources.get('GPU', 0)}")
            logger.info(f"Is multi-node cluster: {self.cluster_info['is_multinode']}")

        except Exception as e:
            logging.getLogger("murmura").warning(f"Could not gather cluster info: {e}")
            self.cluster_info = {
                "total_nodes": 1,
                "cluster_resources": {},
                "available_resources": {},
                "nodes": [],
                "is_multinode": False
            }

    def _calculate_resource_allocation(self) -> Tuple[Dict[str, Any], int]:
        """Calculate optimal resource allocation for actors"""
        cluster_resources = self.cluster_info.get("cluster_resources", {})
        total_cpus = cluster_resources.get("CPU", 1)
        total_gpus = cluster_resources.get("GPU", 0)
        total_nodes = self.cluster_info["total_nodes"]

        # Calculate actors per node
        if self.config.resources.actors_per_node is not None:
            actors_per_node = self.config.resources.actors_per_node
            # Ensure we don't exceed total requested actors
            total_possible_actors = actors_per_node * total_nodes
            if total_possible_actors < self.config.num_actors:
                logging.getLogger("murmura").warning(
                    f"Requested {self.config.num_actors} actors but can only create "
                    f"{total_possible_actors} with {actors_per_node} actors per node"
                )
        else:
            # Distribute evenly across nodes
            actors_per_node = max(1, self.config.num_actors // total_nodes)

        # Calculate resource allocation per actor
        resource_requirements = {}

        # CPU allocation
        if self.config.resources.cpus_per_actor is not None:
            resource_requirements["num_cpus"] = self.config.resources.cpus_per_actor
        else:
            # Auto-calculate based on available resources
            # Use a more conservative approach for multi-node
            cpus_per_actor = max(0.5, total_cpus / (self.config.num_actors * 1.2))  # Leave some headroom
            resource_requirements["num_cpus"] = min(cpus_per_actor, 2.0)  # Cap at 2 CPUs per actor

        # GPU allocation
        if self.config.resources.gpus_per_actor is not None:
            resource_requirements["num_gpus"] = self.config.resources.gpus_per_actor
        elif total_gpus > 0:
            # Auto-calculate GPU allocation
            gpus_per_actor = total_gpus / self.config.num_actors
            resource_requirements["num_gpus"] = gpus_per_actor

        # Memory allocation
        if self.config.resources.memory_per_actor is not None:
            resource_requirements["memory"] = self.config.resources.memory_per_actor * 1024 * 1024

        logging.getLogger("murmura").info(
            f"Resource allocation: {resource_requirements} per actor, "
            f"{actors_per_node} actors per node"
        )

        return resource_requirements, actors_per_node

    def create_actors(self, num_actors: int, topology: TopologyConfig) -> List[Any]:
        """
        Create pool of virtual client actors with simplified multi-node placement
        """
        self.topology_manager = TopologyManager(num_actors, topology)

        # Calculate resource allocation
        resource_requirements, actors_per_node = self._calculate_resource_allocation()

        logging.getLogger("murmura").info(
            f"Creating {num_actors} virtual clients across {self.cluster_info['total_nodes']} nodes"
        )

        self.actors = []
        for i in range(num_actors):
            actor_options = {}

            # Add resource requirements
            actor_options.update(resource_requirements)

            try:
                actor = VirtualClientActor.options(**actor_options).remote(f"client_{i}")
                self.actors.append(actor)

                # Log actor creation with node information
                if i % 10 == 0 or i == num_actors - 1:  # Log every 10th actor and the last one
                    logging.getLogger("murmura").info(f"Created actors {i+1}/{num_actors}")

            except Exception as e:
                logging.getLogger("murmura").error(f"Failed to create actor {i}: {e}")
                raise

        self._apply_topology()

        if self.aggregation_strategy and self.topology_manager:
            self._initialize_coordinator()

        # Log final actor distribution
        self._log_actor_distribution()

        return self.actors

    def _log_actor_distribution(self) -> None:
        """Log how actors are distributed across nodes"""
        try:
            # Sample a few actors to check distribution without overwhelming the system
            if len(self.actors) > 0:
                logging.getLogger("murmura").info(
                    f"Successfully created {len(self.actors)} actors across cluster"
                )

        except Exception as e:
            logging.getLogger("murmura").debug(f"Could not determine actor distribution: {e}")

    def set_aggregation_strategy(self, aggregation_config: AggregationConfig) -> None:
        """Set the aggregation strategy for the cluster"""
        if self.topology_manager:
            self.aggregation_strategy = AggregationStrategyFactory.create(
                aggregation_config, self.topology_manager.config
            )
            self._initialize_coordinator()
        else:
            self.aggregation_strategy = AggregationStrategyFactory.create(
                aggregation_config
            )

    def distribute_data(
            self,
            data_partitions: List[List[int]],
            metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Distribute data partitions to actors with improved error handling"""
        resolved_metadata = metadata or {}
        resolved_metadata.update({
            "cluster_nodes": self.cluster_info["total_nodes"],
            "is_multinode": self.cluster_info["is_multinode"]
        })

        results = []
        batch_size = 20  # Smaller batches for better stability

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i:i + batch_size]
            batch_results = []

            for j, actor in enumerate(batch_actors):
                actual_idx = i + j
                partition_idx = actual_idx % len(data_partitions)

                batch_results.append(
                    actor.receive_data.remote(
                        data_partitions[partition_idx],
                        {**resolved_metadata, "partition_idx": partition_idx, "actor_id": actual_idx},
                    )
                )

            # Wait for batch to complete
            try:
                batch_completed = ray.get(batch_results, timeout=60)  # Increased timeout
                results.extend(batch_completed)

                logging.getLogger("murmura").info(
                    f"Distributed data to actors {i+1}-{min(i+batch_size, len(self.actors))}"
                )

            except Exception as e:
                logging.getLogger("murmura").error(f"Failed to distribute data batch {i//batch_size}: {e}")
                raise

        return results

    def distribute_dataset(
            self,
            dataset: MDataset,
            feature_columns: Optional[List[str]] = None,
            label_column: Optional[str] = None,
    ) -> None:
        """Distribute dataset to all actors with batched processing"""
        batch_size = 20

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i:i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                batch_tasks.append(
                    actor.set_dataset.remote(
                        dataset, feature_columns=feature_columns, label_column=label_column
                    )
                )

            try:
                ray.get(batch_tasks, timeout=60)
                logging.getLogger("murmura").info(
                    f"Distributed dataset to actors {i+1}-{min(i+batch_size, len(self.actors))}"
                )
            except Exception as e:
                logging.getLogger("murmura").error(f"Failed to distribute dataset batch {i//batch_size}: {e}")
                raise

    def distribute_model(self, model: ModelInterface) -> None:
        """Distribute model to all actors with enhanced error handling and batching"""
        logging.getLogger("murmura").info("Distributing model to client actors...")

        # Prepare model for serialization (move to CPU)
        original_device = None
        if hasattr(model, "model") and hasattr(model.model, "to"):
            if hasattr(model, "device"):
                original_device = model.device
            model.model.to("cpu")
            if hasattr(model, "device"):
                model.device = "cpu"

        parameters = model.get_parameters()

        # Distribute in smaller batches for better stability
        batch_size = 10

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i:i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                # Set model structure
                batch_tasks.append(actor.set_model.remote(model))

            try:
                ray.get(batch_tasks, timeout=120)  # Longer timeout for model distribution

                # Now set parameters
                param_tasks = []
                for actor in batch_actors:
                    param_tasks.append(actor.set_model_parameters.remote(parameters))

                ray.get(param_tasks, timeout=120)

                logging.getLogger("murmura").info(
                    f"Distributed model to actors {i+1}-{min(i+batch_size, len(self.actors))}"
                )

            except Exception as e:
                logging.getLogger("murmura").error(f"Failed to distribute model batch {i//batch_size}: {e}")
                raise

        # Restore original device
        if (
                hasattr(model, "model")
                and hasattr(model.model, "to")
                and original_device is not None
        ):
            model.model.to(original_device)
            if hasattr(model, "device"):
                model.device = original_device

    def train_models(self, **kwargs) -> List[Dict[str, float]]:
        """Train models on all actors with improved error handling"""
        batch_size = 50
        all_results = []

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i:i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                batch_tasks.append(actor.train_model.remote(**kwargs))

            try:
                batch_results = ray.get(batch_tasks, timeout=600)  # 10 min timeout for training
                all_results.extend(batch_results)

                logging.getLogger("murmura").debug(
                    f"Completed training batch {i//batch_size + 1}/{(len(self.actors) + batch_size - 1)//batch_size}"
                )

            except Exception as e:
                logging.getLogger("murmura").error(f"Training failed for batch {i//batch_size}: {e}")
                raise

        return all_results

    def evaluate_models(self, **kwargs) -> List[Dict[str, float]]:
        """Evaluate models on all actors with improved error handling"""
        batch_size = 50
        all_results = []

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i:i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                batch_tasks.append(actor.evaluate_model.remote(**kwargs))

            try:
                batch_results = ray.get(batch_tasks, timeout=300)  # 5 min timeout for evaluation
                all_results.extend(batch_results)

            except Exception as e:
                logging.getLogger("murmura").error(f"Evaluation failed for batch {i//batch_size}: {e}")
                raise

        return all_results

    def aggregate_model_parameters(
            self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Aggregate model parameters from all actors"""
        if not self.aggregation_strategy:
            raise ValueError(
                "Aggregation strategy not set. Call set_aggregation_strategy first."
            )

        if not self.topology_coordinator:
            raise ValueError(
                "Topology coordinator not initialized."
            )

        return self.topology_coordinator.coordinate_aggregation(weights=weights)

    def update_aggregation_strategy(
            self, strategy: AggregationStrategy, topology_check: bool = True
    ) -> None:
        """Update the aggregation strategy for the cluster"""
        if topology_check and self.topology_manager:
            strategy_class = strategy.__class__
            topology_type = self.topology_manager.config.topology_type

            if not TopologyCompatibilityManager.is_compatible(
                    strategy_class, topology_type
            ):
                compatible_topologies = (
                    TopologyCompatibilityManager.get_compatible_topologies(
                        strategy_class
                    )
                )
                raise ValueError(
                    f"New strategy {strategy_class.__name__} is not compatible with the current topology {topology_type.value}."
                    f"Compatible topologies: {compatible_topologies}"
                )

        self.aggregation_strategy = strategy

        if self.topology_manager:
            self._initialize_coordinator()

    def update_models(self, parameters: Dict[str, Any]) -> None:
        """Update model parameters on all actors with batching"""
        batch_size = 20

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i:i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                batch_tasks.append(actor.set_model_parameters.remote(parameters))

            try:
                ray.get(batch_tasks, timeout=120)
            except Exception as e:
                logging.getLogger("murmura").error(f"Model update failed for batch {i//batch_size}: {e}")
                raise

    def _apply_topology(self) -> None:
        """Set neighbour relationships based on topology config"""
        if not self.topology_manager:
            return

        adjacency = self.topology_manager.adjacency_list
        tasks = []

        for node, neighbours in adjacency.items():
            neighbour_actors = [self.actors[n] for n in neighbours]
            tasks.append(self.actors[node].set_neighbours.remote(neighbour_actors))

        try:
            ray.get(tasks, timeout=60)
        except Exception as e:
            logging.getLogger("murmura").error(f"Failed to apply topology: {e}")
            raise

    def get_topology_information(self) -> Dict[str, Any]:
        """Get information about the current topology and cluster"""
        base_info = {
            "initialized": bool(self.topology_manager),
            "cluster_info": self.cluster_info
        }

        if not self.topology_manager:
            return base_info

        return {
            **base_info,
            "type": self.topology_manager.config.topology_type.value,
            "num_actors": len(self.actors),
            "hub_index": self.topology_manager.config.hub_index
            if self.topology_manager.config.topology_type == TopologyType.STAR
            else None,
            "adjacency_list": self.topology_manager.adjacency_list,
        }

    def get_compatible_strategies(self) -> List[str]:
        """Get list of strategies compatible with the current topology"""
        if not self.topology_manager:
            return []

        strategy_classes = TopologyCompatibilityManager.get_compatible_strategies(
            self.topology_manager.config.topology_type
        )

        result = []
        for cls in strategy_classes:
            found = False
            for strategy_type in AggregationStrategyType:
                if strategy_type.name.lower() == cls.__name__.lower():
                    result.append(str(strategy_type.value))
                    found = True
                    break

            if not found:
                result.append(cls.__name__.lower())

        return result

    def _initialize_coordinator(self) -> None:
        """Initialize the topology coordinator"""
        if not self.topology_manager or not self.aggregation_strategy:
            return

        self.topology_coordinator = TopologyCoordinator.create(
            self.actors, self.topology_manager, self.aggregation_strategy
        )

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get detailed cluster statistics"""
        return {
            "cluster_info": self.cluster_info,
            "num_actors": len(self.actors),
            "placement_strategy": self.config.resources.placement_strategy,
            "has_placement_group": False,  # Simplified for now
            "resource_config": self.config.resources.model_dump()
        }

    def shutdown(self) -> None:
        """Shutdown cluster resources"""
        try:
            # Clean up actors
            if self.actors:
                logging.getLogger("murmura").info(f"Cleaning up {len(self.actors)} actors")
                # Kill actors in batches to avoid overwhelming the cluster
                batch_size = 20
                for i in range(0, len(self.actors), batch_size):
                    batch_actors = self.actors[i:i + batch_size]
                    for actor in batch_actors:
                        try:
                            ray.kill(actor)
                        except:
                            pass  # Ignore errors during cleanup

            logging.getLogger("murmura").info("Cluster manager shutdown complete")

        except Exception as e:
            logging.getLogger("murmura").error(f"Error during cluster shutdown: {e}")

    @staticmethod
    def shutdown_ray() -> None:
        """Shutdown ray cluster gracefully"""
        try:
            if ray.is_initialized():
                ray.shutdown()
                logging.getLogger("murmura").info("Ray cluster shutdown complete")
        except Exception as e:
            logging.getLogger("murmura").error(f"Error shutting down Ray: {e}")
