import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import ray

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.aggregation.strategy_factory import AggregationStrategyFactory
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.data_processing.dataset import MDataset, DatasetSource
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
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add console handler if not exists
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    def _initialize_ray_cluster(self) -> None:
        """Initialize Ray cluster with multi-node support"""
        if ray.is_initialized():
            logging.getLogger("murmura").info(
                "Ray already initialized, using existing cluster"
            )
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
        if (
            self.config.ray_cluster.auto_detect_cluster
            and not self.config.ray_cluster.address
        ):
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
                "is_multinode": len([n for n in nodes if n["Alive"]]) > 1,
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
                "is_multinode": False,
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

        # CPU allocation - IMPROVED LOGIC for better distribution
        if self.config.resources.cpus_per_actor is not None:
            resource_requirements["num_cpus"] = self.config.resources.cpus_per_actor
        else:
            # Auto-calculate based on actual distribution pattern across nodes
            # Calculate actual distribution of actors across nodes
            base_actors_per_node = self.config.num_actors // total_nodes
            extra_actors = self.config.num_actors % total_nodes

            # Calculate maximum actors on any single node
            max_actors_per_node = base_actors_per_node + (1 if extra_actors > 0 else 0)

            # Calculate CPUs per node (estimate based on total CPUs)
            avg_cpus_per_node = (
                total_cpus / total_nodes if total_nodes > 0 else total_cpus
            )

            # Calculate CPU allocation based on the busiest nodes
            # Leave some headroom (20%) for system processes
            available_cpus_per_node = avg_cpus_per_node * 0.8

            if max_actors_per_node > 0:
                cpus_per_actor = available_cpus_per_node / max_actors_per_node
                # Ensure minimum allocation and cap at 4 CPUs per actor
                cpus_per_actor = max(0.5, min(cpus_per_actor, 4.0))
            else:
                cpus_per_actor = 1.0

            resource_requirements["num_cpus"] = cpus_per_actor

        # GPU allocation - IMPROVED LOGIC for better distribution
        if self.config.resources.gpus_per_actor is not None:
            resource_requirements["num_gpus"] = self.config.resources.gpus_per_actor
        elif total_gpus > 0:
            # IMPROVED: Calculate based on actual distribution pattern across nodes

            # Get GPU information per node to make informed decisions
            nodes = self.cluster_info.get("nodes", [])
            gpus_per_node = []

            for node in nodes:
                if node.get("Alive", False):
                    node_resources = node.get("Resources", {})
                    node_gpus = node_resources.get("GPU", 0)
                    gpus_per_node.append(node_gpus)

            if gpus_per_node:
                # Use the typical (median) GPU count per node as the basis
                median_gpus_per_node = sorted(gpus_per_node)[len(gpus_per_node) // 2]
                max_gpus_per_node = max(gpus_per_node)

                # Calculate actual distribution of actors across nodes
                # For example: 15 actors across 10 nodes means 5 nodes get 2 actors, 5 nodes get 1 actor
                base_actors_per_node = self.config.num_actors // total_nodes
                extra_actors = self.config.num_actors % total_nodes

                # Calculate maximum actors on any single node
                max_actors_per_node = base_actors_per_node + (
                    1 if extra_actors > 0 else 0
                )

                # Calculate GPU requirements based on the busiest nodes
                # This ensures even the busiest nodes have sufficient GPU allocation
                if max_actors_per_node <= median_gpus_per_node:
                    # We can give each actor close to 1 GPU on busiest nodes
                    gpus_per_actor = min(
                        1.0, median_gpus_per_node / max_actors_per_node
                    )
                else:
                    # More actors than GPUs per node - share GPUs
                    # Use median GPUs to ensure compatibility across all nodes
                    gpus_per_actor = median_gpus_per_node / max_actors_per_node

                # Ensure we don't exceed what any single node can provide
                gpus_per_actor = min(gpus_per_actor, max_gpus_per_node)

                # Only allocate if meaningful amount (at least 0.1 GPU)
                if gpus_per_actor >= 0.1:
                    resource_requirements["num_gpus"] = gpus_per_actor

                logging.getLogger("murmura").info(
                    f"GPU allocation: {gpus_per_actor:.2f} GPUs per actor "
                    f"(median {median_gpus_per_node} GPUs/node, "
                    f"max {max_actors_per_node} actors/node, "
                    f"distribution: {base_actors_per_node}+{extra_actors} across {total_nodes} nodes)"
                )
            else:
                # Fallback: simple division but cap at 1.0 GPU per actor
                gpus_per_actor = min(1.0, total_gpus / self.config.num_actors)
                if gpus_per_actor >= 0.1:
                    resource_requirements["num_gpus"] = gpus_per_actor

        # Memory allocation
        if self.config.resources.memory_per_actor is not None:
            resource_requirements["memory"] = (
                self.config.resources.memory_per_actor * 1024 * 1024
            )

        # Log detailed resource allocation information
        logger = logging.getLogger("murmura")
        logger.info(f"Resource allocation per actor: {resource_requirements}")
        logger.info(f"Actors per node: {actors_per_node}")

        # Log CPU allocation details if auto-calculated
        if self.config.resources.cpus_per_actor is None:
            base_actors_per_node = self.config.num_actors // total_nodes
            extra_actors = self.config.num_actors % total_nodes
            max_actors_per_node = base_actors_per_node + (1 if extra_actors > 0 else 0)
            avg_cpus_per_node = (
                total_cpus / total_nodes if total_nodes > 0 else total_cpus
            )

            logger.info(
                f"CPU allocation: {resource_requirements.get('num_cpus', 'N/A'):.2f} CPUs per actor "
                f"(avg {avg_cpus_per_node:.1f} CPUs/node, "
                f"max {max_actors_per_node} actors/node, "
                f"distribution: {base_actors_per_node}+{extra_actors} across {total_nodes} nodes)"
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
            try:
                if resource_requirements:
                    actor_ref = VirtualClientActor.options(  # type: ignore[attr-defined]
                        **resource_requirements
                    ).remote(f"client_{i}")
                else:
                    actor_ref = VirtualClientActor.remote(f"client_{i}")  # type: ignore[attr-defined]
                self.actors.append(actor_ref)

                # Log actor creation with node information
                if (
                    i % 10 == 0 or i == num_actors - 1
                ):  # Log every 10th actor and the last one
                    logging.getLogger("murmura").info(
                        f"Created actors {i + 1}/{num_actors}"
                    )

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
            logging.getLogger("murmura").debug(
                f"Could not determine actor distribution: {e}"
            )

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
        resolved_metadata.update(
            {
                "cluster_nodes": self.cluster_info["total_nodes"],
                "is_multinode": self.cluster_info["is_multinode"],
            }
        )

        results = []
        batch_size = 20  # Smaller batches for better stability

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i : i + batch_size]
            batch_results = []

            for j, actor in enumerate(batch_actors):
                actual_idx = i + j
                partition_idx = actual_idx % len(data_partitions)

                batch_results.append(
                    actor.receive_data.remote(
                        data_partitions[partition_idx],
                        {
                            **resolved_metadata,
                            "partition_idx": partition_idx,
                            "actor_id": actual_idx,
                        },
                    )
                )

            # Wait for batch to complete
            try:
                batch_completed = ray.get(
                    batch_results, timeout=18000
                )  # Increased timeout
                results.extend(batch_completed)

                logging.getLogger("murmura").info(
                    f"Distributed data to actors {i + 1}-{min(i + batch_size, len(self.actors))}"
                )

            except Exception as e:
                logging.getLogger("murmura").error(
                    f"Failed to distribute data batch {i // batch_size}: {e}"
                )
                raise

        return results

    def _is_large_dataset(self, dataset: MDataset) -> bool:
        """
        Determine if dataset should use lazy loading based on size and type.

        Returns:
            True if dataset should use lazy loading
        """
        try:
            # Check total number of samples across all splits
            total_samples = 0
            for split_name in dataset.available_splits:
                split_data = dataset.get_split(split_name)
                total_samples += len(split_data)

            # Heuristics for when to use lazy loading:
            # 1. More than 5000 total samples (likely large images)
            # 2. Multi-node cluster (network transfer overhead)
            # 3. HuggingFace dataset (can be reloaded from cache)

            is_large_by_size = total_samples > 5000
            is_multinode = self.cluster_info.get("is_multinode", False)
            is_hf_dataset = (
                dataset.dataset_metadata.get("source") == DatasetSource.HUGGING_FACE
            )

            # Use lazy loading if it's large AND (multinode OR huggingface)
            should_use_lazy = is_large_by_size and (is_multinode or is_hf_dataset)

            logger = logging.getLogger("murmura")
            logger.info(
                f"Dataset analysis - Samples: {total_samples}, "
                f"Large: {is_large_by_size}, Multi-node: {is_multinode}, "
                f"HF: {is_hf_dataset}, Using lazy: {should_use_lazy}"
            )

            return should_use_lazy

        except Exception as e:
            logging.getLogger("murmura").warning(f"Could not analyze dataset size: {e}")
            return False

    def _distribute_dataset_metadata_only(
        self,
        dataset: MDataset,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Distribute only dataset metadata for lazy loading with preprocessing information.
        """
        logger = logging.getLogger("murmura")
        logger.info("Using lazy dataset distribution (metadata only)")

        # Validate inputs
        if feature_columns is None or label_column is None:
            raise ValueError(
                "feature_columns and label_column must be provided for lazy dataset distribution"
            )

        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError("feature_columns must be a non-empty list")

        if not isinstance(label_column, str) or not label_column:
            raise ValueError("label_column must be a non-empty string")

        # Validate dataset has required metadata
        if not dataset.dataset_metadata:
            raise ValueError("Dataset must have metadata for lazy loading")

        # CRITICAL: Extract preprocessing information from the dataset
        preprocessing_info = self._extract_preprocessing_info(
            dataset, feature_columns, label_column
        )

        # Create comprehensive metadata package with preprocessing info
        dataset_metadata = {
            # Core dataset info - with validation
            "dataset_source": dataset.dataset_metadata.get("source"),
            "dataset_name": dataset.dataset_metadata.get("dataset_name"),
            "dataset_kwargs": dataset.dataset_metadata.get("kwargs", {}),
            # Structure info - with validation
            "available_splits": dataset.available_splits,
            "partitions": dataset.partitions,
            # Column information - explicitly included
            "feature_columns": feature_columns.copy(),
            "label_column": label_column,
            # CRITICAL: Include preprocessing information
            "preprocessing_info": preprocessing_info,
            # Reconstruction strategy
            "lazy_loading": True,
            "reconstruction_strategy": "on_demand",
        }

        # Validate the metadata we're about to send
        required_fields = ["dataset_source", "dataset_name", "available_splits"]
        missing_fields = [
            field for field in required_fields if not dataset_metadata.get(field)
        ]
        if missing_fields:
            raise ValueError(
                f"Dataset metadata missing required fields: {missing_fields}"
            )

        # Ensure we have partitions
        if not dataset_metadata["partitions"]:
            raise ValueError("Dataset must have partitions for lazy loading")

        # Ensure we have valid splits
        if not dataset_metadata["available_splits"]:
            raise ValueError("Dataset must have available_splits for lazy loading")

        logger.info(f"Distributing lazy metadata to {len(self.actors)} actors")
        logger.info(f"Dataset: {dataset_metadata['dataset_name']}")
        logger.info(f"Splits: {dataset_metadata['available_splits']}")
        logger.info(
            f"Partitions: {len(dataset_metadata['partitions'])} splits partitioned"
        )
        logger.info(f"Feature columns: {feature_columns}")
        logger.info(f"Label column: {label_column}")

        if preprocessing_info:
            logger.info(
                f"Preprocessing info included: {list(preprocessing_info.keys())}"
            )

        # Distribute metadata with enhanced error handling
        failed_actors = []
        successful_actors = 0

        for i, actor in enumerate(self.actors):
            try:
                # Create a deep copy of metadata for each actor to avoid reference issues
                actor_metadata = {
                    k: (v.copy() if isinstance(v, (dict, list)) else v)
                    for k, v in dataset_metadata.items()
                }

                # Send metadata with explicit column information
                task = actor.set_dataset_metadata.remote(
                    actor_metadata,
                    feature_columns=feature_columns.copy(),  # Ensure we pass a copy
                    label_column=label_column,
                )

                # Wait for each actor individually with reasonable timeout
                ray.get(task, timeout=18000)  # Increased timeout for network overhead

                successful_actors += 1

                if (i + 1) % 5 == 0 or i == len(
                    self.actors
                ) - 1:  # Log progress every 5 actors
                    logger.info(
                        f"Successfully distributed metadata to {successful_actors}/{i + 1} actors"
                    )

            except Exception as e:
                error_msg = f"Failed to distribute metadata to actor {i}: {e}"
                logger.error(error_msg)
                logger.error(f"  - Actor index: {i}")
                logger.error(f"  - Feature columns: {feature_columns}")
                logger.error(f"  - Label column: {label_column}")
                logger.error(
                    f"  - Dataset name: {dataset_metadata.get('dataset_name')}"
                )
                failed_actors.append((i, str(e)))

        # Report results_phase1
        if failed_actors:
            failure_details = [f"Actor {idx}: {error}" for idx, error in failed_actors]
            error_msg = (
                f"Dataset metadata distribution failed for {len(failed_actors)}/{len(self.actors)} actors.\n"
                f"Successful: {successful_actors}, Failed: {len(failed_actors)}\n"
                f"Failures: {failure_details[:3]}..."  # Show first 3 failures
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(
            f"Successfully distributed metadata to all {successful_actors} actors"
        )

    def _extract_preprocessing_info(
        self, dataset: MDataset, feature_columns: List[str], label_column: str
    ) -> Dict[str, Any]:
        """
        Extract preprocessing information from the dataset to recreate transformations on remote nodes.
        """
        logger = logging.getLogger("murmura")

        # PRIORITY 1: Check if preprocessing info is already stored in dataset metadata
        if (
            hasattr(dataset, "_dataset_metadata")
            and dataset._dataset_metadata
            and "preprocessing_applied" in dataset._dataset_metadata
        ):
            stored_preprocessing = dataset._dataset_metadata["preprocessing_applied"]
            logger.info(
                f"Found stored preprocessing metadata: {list(stored_preprocessing.keys())}"
            )
            return stored_preprocessing

        # PRIORITY 2: Try to detect preprocessing by examining the dataset
        # FIXED: Proper type annotation
        preprocessing_info: Dict[str, Any] = {}

        try:
            # Check if we have any split to examine
            if not dataset.available_splits:
                return preprocessing_info

            # Get a sample split to examine the data structure
            sample_split_name = dataset.available_splits[0]
            sample_split = dataset.get_split(sample_split_name)

            if len(sample_split) == 0:
                return preprocessing_info

            # Get column names and a sample
            column_names = sample_split.column_names
            sample_data = sample_split[0]

            logger.debug(
                f"Extracting preprocessing info from dataset with columns: {column_names}"
            )

            # Check for label encoding (most common case)
            # If the target label_column exists but there's also a string column that could be the source
            if label_column in column_names:
                # Look for potential source columns (common patterns)
                potential_source_columns = [
                    "dx",
                    "diagnosis",
                    "class",
                    "category",
                    "target",
                ]

                for source_col in potential_source_columns:
                    if source_col in column_names and source_col != label_column:
                        # Check if source column contains strings and target contains integers
                        source_sample = sample_data.get(source_col)
                        target_sample = sample_data.get(label_column)

                        if isinstance(source_sample, str) and isinstance(
                            target_sample, (int, np.integer)
                        ):
                            logger.info(
                                f"Detected label encoding: {source_col} -> {label_column}"
                            )

                            # Extract the full mapping by examining the entire split
                            logger.debug("Extracting label mapping from dataset...")

                            # FIXED: Proper type annotation for mapping
                            mapping: Dict[str, int] = {}

                            # Sample more data to build the mapping (don't load entire dataset)
                            sample_size = min(
                                1000, len(sample_split)
                            )  # Sample first 1000 or all if smaller
                            sample_data_list = sample_split.select(range(sample_size))

                            for example in sample_data_list:
                                source_val = example.get(source_col)
                                target_val = example.get(label_column)
                                if source_val is not None and target_val is not None:
                                    mapping[source_val] = target_val

                            if mapping:
                                preprocessing_info["label_encoding"] = {
                                    "source_column": source_col,
                                    "target_column": label_column,
                                    "mapping": mapping,
                                }
                                logger.info(
                                    f"Extracted label mapping with {len(mapping)} categories: {mapping}"
                                )
                                break

            # Add other preprocessing detection logic here as needed
            # For example: feature scaling, normalization, etc.

        except Exception as e:
            logger.warning(f"Failed to extract preprocessing info: {e}")
            # Return empty dict - preprocessing will be skipped

        return preprocessing_info

    def validate_actor_dataset_state(self) -> Dict[str, Any]:
        """
        Enhanced validation of actor dataset state with better error reporting.
        """
        logger = logging.getLogger("murmura")
        logger.info("Validating actor dataset state across cluster...")

        # FIXED: Proper type annotations for validation_results
        validation_results: Dict[str, Any] = {
            "total_actors": len(self.actors),
            "valid_actors": 0,
            "invalid_actors": 0,
            "unreachable_actors": 0,
            "errors": [],
            "actor_details": [],
        }

        # Use smaller batches for validation to avoid overwhelming the cluster
        batch_size = 10

        for batch_start in range(0, len(self.actors), batch_size):
            batch_end = min(batch_start + batch_size, len(self.actors))
            batch_actors = self.actors[batch_start:batch_end]

            # Create batch tasks
            batch_tasks = []
            for i, actor in enumerate(batch_actors):
                actual_idx = batch_start + i
                batch_tasks.append((actual_idx, actor.get_data_info.remote()))

            # Wait for batch results_phase1
            try:
                # Get all results_phase1 for this batch
                batch_results = ray.get(
                    [task for _, task in batch_tasks], timeout=18000
                )

                # Process results_phase1
                for (actual_idx, _), actor_info in zip(batch_tasks, batch_results):
                    try:
                        # Validate required fields with more specific checks
                        has_data_partition = actor_info.get("data_size", 0) > 0
                        has_feature_columns = (
                            actor_info.get("feature_columns") is not None
                            and len(actor_info.get("feature_columns", [])) > 0
                        )
                        has_label_column = (
                            actor_info.get("label_column") is not None
                            and actor_info.get("label_column") != ""
                        )
                        has_dataset_ready = actor_info.get(
                            "has_dataset", False
                        ) or actor_info.get("lazy_loading", False)

                        is_valid = all(
                            [
                                has_data_partition,
                                has_feature_columns,
                                has_label_column,
                                has_dataset_ready,
                            ]
                        )

                        actor_detail = {
                            "actor_id": actual_idx,
                            "client_id": actor_info.get("client_id"),
                            "node_id": actor_info.get("node_info", {}).get(
                                "node_id", "unknown"
                            ),
                            "is_valid": is_valid,
                            "data_partition_size": actor_info.get("data_size", 0),
                            "has_feature_columns": has_feature_columns,
                            "has_label_column": has_label_column,
                            "feature_columns": actor_info.get("feature_columns"),
                            "label_column": actor_info.get("label_column"),
                            "lazy_loading": actor_info.get("lazy_loading", False),
                            "dataset_loaded": actor_info.get("has_dataset", False),
                            "dataset_name": actor_info.get("dataset_name", "unknown"),
                        }

                        # FIXED: Direct access to known dictionary keys
                        validation_results["actor_details"].append(actor_detail)

                        if is_valid:
                            validation_results["valid_actors"] += 1
                        else:
                            validation_results["invalid_actors"] += 1

                            # Create detailed error message
                            missing_items = []
                            if not has_data_partition:
                                missing_items.append("data_partition")
                            if not has_feature_columns:
                                missing_items.append("feature_columns")
                            if not has_label_column:
                                missing_items.append("label_column")
                            if not has_dataset_ready:
                                missing_items.append("dataset")

                            error_msg = (
                                f"Actor {actual_idx} validation failed: missing {', '.join(missing_items)}. "
                                f"Feature columns: {actor_info.get('feature_columns')}, "
                                f"Label column: {actor_info.get('label_column')}"
                            )
                            validation_results["errors"].append(error_msg)

                    except Exception as detail_error:
                        validation_results["invalid_actors"] += 1
                        error_msg = f"Actor {actual_idx} detail processing error: {detail_error}"
                        validation_results["errors"].append(error_msg)
                        logger.error(error_msg)

            except Exception as batch_error:
                # Handle batch timeout or other errors
                logger.error(
                    f"Batch validation failed for actors {batch_start}-{batch_end - 1}: {batch_error}"
                )

                # Mark all actors in this batch as unreachable
                for i in range(batch_start, batch_end):
                    validation_results["unreachable_actors"] += 1
                    error_msg = f"Actor {i} unreachable: {batch_error}"
                    validation_results["errors"].append(error_msg)

        # Log comprehensive validation summary
        logger.info("Actor validation complete:")
        logger.info(
            f"  Valid actors: {validation_results['valid_actors']}/{validation_results['total_actors']}"
        )
        logger.info(f"  Invalid actors: {validation_results['invalid_actors']}")
        logger.info(f"  Unreachable actors: {validation_results['unreachable_actors']}")

        # Log first few errors for debugging
        errors_list = validation_results["errors"]
        if len(errors_list) > 0:
            logger.error("Actor validation errors found:")
            for error in errors_list[:5]:  # Show first 5 errors
                logger.error(f"  {error}")
            if len(errors_list) > 5:
                logger.error(f"  ... and {len(errors_list) - 5} more errors")

        return validation_results

    def distribute_dataset(
        self,
        dataset: MDataset,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Enhanced dataset distribution with automatic lazy loading for large datasets.
        """
        # Validate inputs
        if feature_columns is None or label_column is None:
            raise ValueError(
                "feature_columns and label_column must be provided for dataset distribution"
            )

        logger = logging.getLogger("murmura")
        is_multinode = self.cluster_info.get("is_multinode", False)

        # Decide on distribution strategy
        if self._is_large_dataset(dataset):
            logger.info("Using lazy distribution for large dataset")
            self._distribute_dataset_metadata_only(
                dataset, feature_columns, label_column
            )

        elif is_multinode:
            logger.info("Using HuggingFace metadata reconstruction for multi-node")

            # Check if dataset can be safely serialized
            if not dataset.is_serializable_for_multinode():
                logger.info("Converting dataset for multi-node compatibility...")
                dataset = dataset.prepare_for_multinode_distribution()

            # Use HuggingFace reconstruction approach
            if dataset.dataset_metadata.get("source") == DatasetSource.HUGGING_FACE:
                self._distribute_hf_dataset_metadata(
                    dataset, feature_columns, label_column
                )
            else:
                self._distribute_serializable_dataset(
                    dataset, feature_columns, label_column
                )

        else:
            logger.info("Using direct serialization for single-node")
            self._distribute_serializable_dataset(
                dataset, feature_columns, label_column
            )

    def _distribute_hf_dataset_metadata(
        self,
        dataset: MDataset,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Distribute HuggingFace dataset using metadata reconstruction approach.
        This avoids serializing large memory-mapped files across the network.
        """
        # Validate that columns are provided
        if feature_columns is None or label_column is None:
            raise ValueError(
                "feature_columns and label_column must be provided for multi-node dataset distribution"
            )

        batch_size = 5  # Smaller batches for metadata reconstruction

        # Create a lightweight metadata package
        dataset_metadata = {
            "metadata": dataset.dataset_metadata,
            "partitions": dataset.partitions,
            "available_splits": dataset.available_splits,
            # Include column information in metadata for reconstruction
            "feature_columns": feature_columns,
            "label_column": label_column,
        }

        logger = logging.getLogger("murmura")
        logger.info(
            f"Distributing dataset with feature_columns={feature_columns}, label_column={label_column}"
        )

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i : i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                # Send reconstruction instruction with explicit column information
                batch_tasks.append(
                    actor.reconstruct_and_set_dataset.remote(
                        dataset_metadata,
                        feature_columns=feature_columns,
                        label_column=label_column,
                    )
                )

            try:
                ray.get(
                    batch_tasks, timeout=1800
                )  # Increased timeout for reconstruction
                logger.info(
                    f"Distributed dataset metadata to actors {i + 1}-{min(i + batch_size, len(self.actors))}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to distribute dataset metadata batch {i // batch_size}: {e}"
                )
                # Fallback to direct distribution
                logger.info("Falling back to direct dataset distribution...")
                self._distribute_serializable_dataset(
                    dataset, feature_columns, label_column
                )
                return

    def _distribute_serializable_dataset(
        self,
        dataset: MDataset,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Distribute dataset using direct serialization (original method).
        Used for single-node clusters or already-serializable datasets.
        """
        # Validate that columns are provided
        if feature_columns is None or label_column is None:
            raise ValueError(
                "feature_columns and label_column must be provided for dataset distribution"
            )

        batch_size = 10  # Smaller batches for stability

        logger = logging.getLogger("murmura")
        logger.info(
            f"Distributing dataset with feature_columns={feature_columns}, label_column={label_column}"
        )

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i : i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                batch_tasks.append(
                    actor.set_dataset.remote(
                        dataset,
                        feature_columns=feature_columns,
                        label_column=label_column,
                    )
                )

            try:
                ray.get(batch_tasks, timeout=1800)  # Increased timeout
                logging.getLogger("murmura").info(
                    f"Distributed dataset to actors {i + 1}-{min(i + batch_size, len(self.actors))}"
                )
            except Exception as e:
                logging.getLogger("murmura").error(
                    f"Failed to distribute dataset batch {i // batch_size}: {e}"
                )
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
            batch_actors = self.actors[i : i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                # Set model structure
                batch_tasks.append(actor.set_model.remote(model))

            try:
                ray.get(
                    batch_tasks, timeout=1800
                )  # Longer timeout for model distribution

                # Now set parameters
                param_tasks = []
                for actor in batch_actors:
                    param_tasks.append(actor.set_model_parameters.remote(parameters))

                ray.get(param_tasks, timeout=1800)

                logging.getLogger("murmura").info(
                    f"Distributed model to actors {i + 1}-{min(i + batch_size, len(self.actors))}"
                )

            except Exception as e:
                logging.getLogger("murmura").error(
                    f"Failed to distribute model batch {i // batch_size}: {e}"
                )
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

    def train_models(
        self,
        client_sampling_rate: float = 1.0,
        data_sampling_rate: float = 1.0,
        **kwargs,
    ) -> List[Dict[str, float]]:
        """
        Train models on all or a subset of actors with improved error handling.

        Args:
            client_sampling_rate: Fraction of clients to sample for training (1.0 = all clients)
            data_sampling_rate: Fraction of local data to sample per client (1.0 = all data)
            **kwargs: Additional training parameters

        Returns:
            List of training results_phase1 from sampled clients
        """
        # Sample clients if client_sampling_rate < 1.0
        if client_sampling_rate < 1.0:
            num_clients_to_sample = max(1, int(len(self.actors) * client_sampling_rate))
            import random

            sampled_actors = random.sample(self.actors, num_clients_to_sample)

            logging.getLogger("murmura").info(
                f"Client subsampling: selected {num_clients_to_sample}/{len(self.actors)} clients "
                f"(sampling rate: {client_sampling_rate:.2f})"
            )
        else:
            sampled_actors = self.actors

        # Add data sampling rate to kwargs
        kwargs["data_sampling_rate"] = data_sampling_rate

        batch_size = 50
        all_results = []

        for i in range(0, len(sampled_actors), batch_size):
            batch_actors = sampled_actors[i : i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                batch_tasks.append(actor.train_model.remote(**kwargs))

            try:
                batch_results = ray.get(
                    batch_tasks, timeout=18000
                )  # 30 min timeout for training
                all_results.extend(batch_results)

                logging.getLogger("murmura").debug(
                    f"Completed training batch {i // batch_size + 1}/{(len(sampled_actors) + batch_size - 1) // batch_size}"
                )

            except Exception as e:
                logging.getLogger("murmura").error(
                    f"Training failed for batch {i // batch_size}: {e}"
                )
                raise

        return all_results

    def evaluate_models(self, **kwargs) -> List[Dict[str, float]]:
        """Evaluate models on all actors with improved error handling"""
        batch_size = 50
        all_results = []

        for i in range(0, len(self.actors), batch_size):
            batch_actors = self.actors[i : i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                batch_tasks.append(actor.evaluate_model.remote(**kwargs))

            try:
                batch_results = ray.get(
                    batch_tasks, timeout=1800
                )  # 15 min timeout for evaluation
                all_results.extend(batch_results)

            except Exception as e:
                logging.getLogger("murmura").error(
                    f"Evaluation failed for batch {i // batch_size}: {e}"
                )
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
            raise ValueError("Topology coordinator not initialized.")

        return self.topology_coordinator.coordinate_aggregation(weights=weights)

    def perform_decentralized_aggregation(
        self, weights: Optional[List[float]] = None
    ) -> None:
        """
        Perform decentralized aggregation where each node aggregates with its neighbors.
        Unlike centralized aggregation, no global model is maintained or distributed.
        Each node maintains its own local model after aggregating with neighbors.
        
        This leverages the existing topology coordinator but skips global combination.
        Works with any aggregation strategy (GossipAvg, future strategies).
        """
        if not self.aggregation_strategy:
            raise ValueError(
                "Aggregation strategy not set. Call set_aggregation_strategy first."
            )

        if not self.topology_coordinator:
            raise ValueError("Topology coordinator not initialized.")

        # Get topology information
        topology_info = self.get_topology_information()
        adjacency_list = topology_info.get("adjacency_list", {})

        # Each node performs local aggregation with its neighbors
        # This is similar to what topology coordinator does, but we don't combine results globally
        update_tasks = []
        
        for node_idx, actor in enumerate(self.actors):
            neighbors = adjacency_list.get(node_idx, [])
            if neighbors:
                # Collect parameters from this node and its neighbors
                neighbor_params = []
                neighbor_weights = []
                
                # Include the node's own parameters
                own_params = ray.get(actor.get_model_parameters.remote(), timeout=1800)
                neighbor_params.append(own_params)
                neighbor_weights.append(weights[node_idx] if weights else 1.0)
                
                # Add neighbor parameters
                for neighbor_idx in neighbors:
                    if neighbor_idx < len(self.actors):
                        neighbor_param = ray.get(
                            self.actors[neighbor_idx].get_model_parameters.remote(), 
                            timeout=1800
                        )
                        neighbor_params.append(neighbor_param)
                        neighbor_weights.append(weights[neighbor_idx] if weights else 1.0)
                
                # Perform local aggregation using the configured strategy
                if len(neighbor_params) > 1:
                    # Normalize weights for this local aggregation
                    total_weight = sum(neighbor_weights)
                    normalized_weights = [w / total_weight for w in neighbor_weights] if total_weight > 0 else neighbor_weights
                    
                    aggregated_params = self.aggregation_strategy.aggregate(
                        neighbor_params, normalized_weights
                    )
                    # Schedule asynchronous update of the node's model
                    update_tasks.append(actor.set_model_parameters.remote(aggregated_params))
        
        # Wait for all updates to complete
        if update_tasks:
            ray.get(update_tasks, timeout=1800)

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
            batch_actors = self.actors[i : i + batch_size]
            batch_tasks = []

            for actor in batch_actors:
                batch_tasks.append(actor.set_model_parameters.remote(parameters))

            try:
                ray.get(batch_tasks, timeout=1800)
            except Exception as e:
                logging.getLogger("murmura").error(
                    f"Model update failed for batch {i // batch_size}: {e}"
                )
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
            # Use very large timeout to prevent timeout issues with large configurations
            timeout = 3600  # 1 hour timeout
            logging.getLogger("murmura").info(
                f"Applying topology for {len(self.actors)} actors with {timeout}s timeout"
            )
            ray.get(tasks, timeout=timeout)
        except Exception as e:
            logging.getLogger("murmura").error(f"Failed to apply topology: {e}")
            raise

    def get_topology_information(self) -> Dict[str, Any]:
        """Get information about the current topology and cluster"""
        base_info = {
            "initialized": bool(self.topology_manager),
            "cluster_info": self.cluster_info,
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
            "resource_config": self.config.resources.model_dump(),
        }

    def shutdown(self) -> None:
        """Shutdown cluster resources"""
        try:
            # Clean up actors
            if self.actors:
                logging.getLogger("murmura").info(
                    f"Cleaning up {len(self.actors)} actors"
                )
                # Kill actors in batches to avoid overwhelming the cluster
                batch_size = 20
                for i in range(0, len(self.actors), batch_size):
                    batch_actors = self.actors[i : i + batch_size]
                    for actor in batch_actors:
                        try:
                            ray.kill(actor)
                        except Exception:
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
