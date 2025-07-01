import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import ray
import torch

from murmura.data_processing.dataset import MDataset
from murmura.model.model_interface import ModelInterface


def get_node_id() -> str:
    """Get the node ID this actor is running on - standalone function for serialization"""
    try:
        return ray.get_runtime_context().get_node_id()[:8]
    except Exception:
        return "unknown"


def get_node_info() -> Dict[str, Any]:
    """Get detailed node information - standalone function for serialization"""
    try:
        runtime_context = ray.get_runtime_context()
        node_id = runtime_context.get_node_id()[:8]

        worker_id = "unknown"
        try:
            worker_info = ray.runtime_context.get_runtime_context().get_worker_id()
            if worker_info:
                worker_id = str(worker_info)[:8]
        except (AttributeError, Exception):
            pass

        return {
            "node_id": node_id,
            "worker_id": worker_id,
            "hostname": os.environ.get("HOSTNAME", "unknown"),
            "ip": os.environ.get("RAY_NODE_IP", "unknown"),
        }
    except Exception as e:
        return {
            "node_id": "unknown",
            "worker_id": "unknown",
            "hostname": "unknown",
            "ip": "unknown",
            "error": str(e),
        }


@ray.remote
class VirtualClientActor:
    """Ray remote actor representing a virtual client in federated learning with multi-node support."""

    def __init__(self, client_id: str, attack_config: Optional[Dict[str, Any]] = None) -> None:
        self.client_id = client_id
        self.data_partition: Optional[List[int]] = None
        self.metadata: Dict[str, Any] = {}
        self.neighbours: List[Any] = []
        self.model: Optional[ModelInterface] = None
        self.mdataset: Optional[MDataset] = None
        self.split: str = "train"

        # CRITICAL: Initialize these properly
        self.feature_columns: Optional[List[str]] = None
        self.label_column: Optional[str] = None

        # Lazy loading state
        self.lazy_loading: bool = False
        self.dataset_metadata: Optional[Dict[str, Any]] = None
        self.dataset_loaded: bool = False
        
        # Attack capabilities
        self.attack_config = attack_config or {}
        self.attack_enabled = (
            self.attack_config is not None and 
            (isinstance(self.attack_config, dict) and len(self.attack_config) > 0) or
            (hasattr(self.attack_config, '__dict__'))  # For AttackConfig objects
        )
        self.attack_instance = None
        self.current_round = 0
        self.attack_history: List[Dict[str, Any]] = []
        
        # Set up logging for multi-node environment
        self._setup_logging()

        if self.attack_enabled:
            try:
                # Import here to avoid circular imports
                from murmura.attacks.gradual_label_flipping import GradualLabelFlippingAttack, AttackConfig
                
                # Handle different attack config types
                if isinstance(self.attack_config, dict):
                    attack_type = self.attack_config.get("attack_type", "label_flipping")
                    
                    if attack_type == "gradual_label_flipping":
                        # Create attack config from dict
                        attack_config_obj = AttackConfig(**self.attack_config)
                        self.attack_instance = GradualLabelFlippingAttack(
                            node_id=client_id,
                            config=attack_config_obj,
                            num_classes=self.attack_config.get("num_classes", 10),
                            dataset_name=self.attack_config.get("dataset_name", "mnist")
                        )
                    else:
                        from murmura.attacks.simple_attacks import create_simple_attack
                        self.attack_instance = create_simple_attack(attack_type, self.attack_config)
                    
                elif hasattr(self.attack_config, '__dict__'):
                    # Attack config object (like AttackConfig)
                    self.attack_instance = GradualLabelFlippingAttack(
                        node_id=client_id,
                        config=self.attack_config,
                        num_classes=getattr(self.attack_config, 'num_classes', 10),
                        dataset_name=getattr(self.attack_config, 'dataset_name', 'mnist')
                    )
                    attack_type = "gradual_label_flipping"
                else:
                    raise ValueError(f"Unknown attack config type: {type(self.attack_config)}")
                
                self.logger.warning(f"Attack mode enabled for client {client_id}: {attack_type}")
            except (ImportError, ValueError) as e:
                self.logger.warning(f"Attack functionality disabled: {e}. Running in honest mode.")
                self.attack_enabled = False
                self.attack_instance = None

        # Initialize device info
        self.device_info = {
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
        }

        # Get node information using standalone function
        self.node_info = get_node_info()

        self.logger.info(
            f"Actor {client_id} initialized on node {self.node_info['node_id']} "
            f"with device: {self.device_info['device']}"
            f"{' (MALICIOUS)' if self.attack_enabled else ''}"
        )

    def _setup_logging(self) -> None:
        """Set up logging for the actor with node information"""
        self.logger = logging.getLogger(f"murmura.actor.{self.client_id}")

        if not self.logger.handlers:
            current_node_id = get_node_id()
            formatter = logging.Formatter(
                f"%(asctime)s - %(name)s - [Node:{current_node_id}] - [Actor:{self.client_id}] - %(levelname)s - %(message)s"
            )

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            log_level = os.environ.get("MURMURA_LOG_LEVEL", "INFO")
            self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    def get_node_info(self) -> Dict[str, Any]:
        """Return node information for this actor"""
        return self.node_info

    def receive_data(
        self, data_partition: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Receive a data partition and metadata dictionary."""
        self.data_partition = data_partition
        self.metadata = metadata if metadata is not None else {}

        message = f"Client {self.client_id} received {len(data_partition)} samples"
        self.logger.debug(message)
        return message

    def set_dataset(
        self,
        mdataset: MDataset,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """Set the dataset for the client actor."""
        self.mdataset = mdataset
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.dataset_loaded = True
        self.lazy_loading = False

        self.logger.debug(
            f"Dataset set with features: {feature_columns}, label: {label_column}"
        )

    def set_dataset_metadata(
        self,
        dataset_metadata: Dict[str, Any],
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Set dataset metadata for lazy loading with robust validation.
        """
        # Validate inputs
        if not dataset_metadata:
            raise ValueError("Dataset metadata cannot be empty")

        if not feature_columns:
            raise ValueError("feature_columns must be provided and non-empty")

        if not label_column:
            raise ValueError("label_column must be provided and non-empty")

        # CRITICAL: Store all information properly
        self.dataset_metadata = (
            dataset_metadata.copy()
        )  # Deep copy to avoid reference issues
        self.feature_columns = (
            feature_columns.copy()
            if isinstance(feature_columns, list)
            else feature_columns
        )
        self.label_column = label_column
        self.mdataset = None  # Will be loaded on-demand
        self.lazy_loading = True
        self.dataset_loaded = False

        # Enhanced validation of metadata structure
        required_keys = [
            "dataset_source",
            "dataset_name",
            "available_splits",
            "dataset_kwargs",
            "partitions",
        ]

        missing_keys = [
            key for key in required_keys if key not in self.dataset_metadata
        ]
        if missing_keys:
            self.logger.error(f"Missing required metadata keys: {missing_keys}")
            raise ValueError(f"Dataset metadata missing required keys: {missing_keys}")

        # Validate that we have proper splits and partitions
        if not self.dataset_metadata.get("available_splits"):
            raise ValueError("Dataset metadata must contain 'available_splits'")

        if not isinstance(self.dataset_metadata.get("partitions"), dict):
            raise ValueError(
                "Dataset metadata must contain 'partitions' as a dictionary"
            )

        self.logger.info("Set dataset metadata for lazy loading")
        self.logger.info(f"Dataset: {self.dataset_metadata.get('dataset_name')}")
        self.logger.info(f"Splits: {self.dataset_metadata.get('available_splits')}")
        self.logger.info(
            f"Features: {self.feature_columns}, Label: {self.label_column}"
        )

        # Additional debug logging
        self.logger.debug(f"Metadata keys: {list(self.dataset_metadata.keys())}")
        self.logger.debug(
            f"Dataset source: {self.dataset_metadata.get('dataset_source')}"
        )
        self.logger.debug(
            f"Dataset kwargs: {self.dataset_metadata.get('dataset_kwargs', {})}"
        )

    def reconstruct_and_set_dataset(
        self,
        dataset_info: Dict[str, Any],
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Reconstruct dataset from metadata on this node and set it.

        This method is used in multi-node environments to avoid transferring
        large dataset files across the network.

        :param dataset_info: Dataset metadata and reconstruction information
        :param feature_columns: List of feature column names
        :param label_column: Name of the label column
        """
        try:
            self.logger.info("Reconstructing dataset from metadata on this node...")

            # Extract metadata
            metadata = dataset_info["metadata"]
            partitions = dataset_info["partitions"]

            # Import MDataset here to avoid circular imports
            from murmura.data_processing.dataset import MDataset

            # Check if MDataset class has the reconstruct_from_metadata method
            if not hasattr(MDataset, "reconstruct_from_metadata"):
                self.logger.error(
                    "MDataset class does not have 'reconstruct_from_metadata' method"
                )
                raise AttributeError(
                    "type object 'MDataset' has no attribute 'reconstruct_from_metadata'"
                )

            # Reconstruct the dataset using the class method
            reconstructed_dataset = MDataset.reconstruct_from_metadata(
                metadata, partitions
            )

            # Set the reconstructed dataset
            self.mdataset = reconstructed_dataset
            if feature_columns:
                self.feature_columns = feature_columns
            if label_column:
                self.label_column = label_column

            self.logger.info(
                f"Dataset reconstructed successfully with {len(reconstructed_dataset.available_splits)} splits"
            )
            self.logger.debug(f"Features: {feature_columns}, Label: {label_column}")

        except Exception as e:
            self.logger.error(f"Failed to reconstruct dataset: {e}")
            # Create a fallback empty dataset to prevent crashes
            try:
                from murmura.data_processing.dataset import MDataset
                from datasets import DatasetDict, Dataset

                # Create a minimal empty dataset
                empty_data = {"dummy": [0]}
                empty_dataset = Dataset.from_dict(empty_data)
                empty_splits = DatasetDict({"train": empty_dataset})
                self.mdataset = MDataset(empty_splits)
                self.feature_columns = feature_columns
                self.label_column = label_column

                self.logger.warning(
                    "Created fallback empty dataset due to reconstruction failure"
                )

            except Exception as fallback_error:
                self.logger.error(
                    f"Even fallback dataset creation failed: {fallback_error}"
                )
                self.mdataset = None
                self.feature_columns = feature_columns
                self.label_column = label_column

            raise RuntimeError(
                f"Dataset reconstruction failed on node {self.node_info['node_id']}: {e}"
            )

    def set_model(self, model: ModelInterface) -> None:
        """Set the model for the client actor with proper device handling."""
        self.model = model

        if hasattr(self.model, "detect_and_set_device"):
            self.model.detect_and_set_device()

        self.logger.debug(
            f"Model set on device: {getattr(self.model, 'device', 'unknown')}"
        )

    def get_data_info(self) -> Dict[str, Any]:
        """Return comprehensive information about stored data partition."""
        info = {
            "client_id": self.client_id,
            "data_size": len(self.data_partition) if self.data_partition else 0,
            "metadata": self.metadata,
            "has_model": self.model is not None,
            "has_dataset": self.mdataset is not None,
            "lazy_loading": self.lazy_loading,
            "dataset_loaded": self.dataset_loaded,
            "node_info": self.node_info,
            "device_info": self.device_info,
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
        }

        if self.dataset_metadata:
            info["dataset_name"] = self.dataset_metadata.get("dataset_name")
            info["available_splits"] = self.dataset_metadata.get("available_splits", [])

        return info

    def _lazy_load_dataset(self) -> None:
        """
        Load the dataset on-demand with enhanced error handling, validation, and preprocessing support.
        """
        if self.mdataset is not None:
            self.logger.debug("Dataset already loaded, skipping lazy loading")
            return

        if not self.dataset_metadata:
            raise ValueError("No dataset metadata available for lazy loading")

        try:
            self.logger.info("Lazy loading dataset from metadata...")

            # Extract and validate metadata
            dataset_source = self.dataset_metadata.get("dataset_source")
            dataset_name = self.dataset_metadata.get("dataset_name")
            dataset_kwargs = self.dataset_metadata.get("dataset_kwargs", {})
            available_splits = self.dataset_metadata.get("available_splits", [])
            partitions = self.dataset_metadata.get("partitions", {})

            # NEW: Extract preprocessing information
            preprocessing_info = self.dataset_metadata.get("preprocessing_info", {})

            # Validate required fields
            if not dataset_source:
                raise ValueError("dataset_source not found in metadata")
            if not dataset_name:
                raise ValueError("dataset_name not found in metadata")
            if not available_splits:
                raise ValueError("available_splits not found in metadata")

            self.logger.info(f"Loading dataset: {dataset_name}")
            self.logger.info(f"Source: {dataset_source}")
            self.logger.info(f"Available splits: {available_splits}")

            # Check if we need to apply preprocessing
            if preprocessing_info:
                self.logger.info(
                    f"Dataset requires preprocessing: {list(preprocessing_info.keys())}"
                )

            # Import here to avoid circular imports
            from murmura.data_processing.dataset import MDataset, DatasetSource

            # Validate dataset source
            if dataset_source != DatasetSource.HUGGING_FACE:
                raise ValueError(
                    f"Lazy loading currently only supports HuggingFace datasets, got: {dataset_source}"
                )

            # Enhanced dataset loading with better error handling
            try:
                # Method 1: Try to load all splits at once
                self.logger.debug("Attempting to load all splits at once...")

                # Ensure we don't pass duplicate dataset_name in kwargs
                clean_kwargs = {
                    k: v for k, v in dataset_kwargs.items() if k != "dataset_name"
                }

                full_dataset = MDataset.load(
                    DatasetSource.HUGGING_FACE,
                    dataset_name=dataset_name,
                    split=None,  # Load all splits
                    **clean_kwargs,
                )
                self.mdataset = full_dataset
                self.logger.info("Successfully loaded all splits at once")

            except Exception as e1:
                self.logger.warning(f"Failed to load all splits at once: {e1}")
                self.logger.info("Attempting to load splits individually...")

                # Method 2: Load splits individually and merge
                loaded_datasets = []
                for split_name in available_splits:
                    try:
                        self.logger.debug(f"Loading split: {split_name}")

                        split_dataset = MDataset.load(
                            DatasetSource.HUGGING_FACE,
                            dataset_name=dataset_name,
                            split=split_name,
                            **clean_kwargs,
                        )
                        loaded_datasets.append((split_name, split_dataset))
                        self.logger.debug(f"Successfully loaded split: {split_name}")

                    except Exception as e2:
                        self.logger.error(f"Failed to load split {split_name}: {e2}")
                        # Continue trying other splits

                if not loaded_datasets:
                    raise RuntimeError("Failed to load any dataset splits")

                # Merge loaded datasets
                self.mdataset = loaded_datasets[0][1]  # Start with first dataset

                if self.mdataset is not None:
                    for split_name, dataset in loaded_datasets[1:]:
                        try:
                            self.mdataset.merge_splits(dataset)
                            self.logger.debug(f"Merged split: {split_name}")
                        except Exception as e3:
                            self.logger.error(
                                f"Failed to merge split {split_name}: {e3}"
                            )

                    self.logger.info(
                        f"Successfully loaded {len(loaded_datasets)} splits individually"
                    )

            # CRITICAL: Apply preprocessing if needed (this is the key fix!)
            if preprocessing_info:
                self.logger.info("Applying dataset preprocessing on remote node...")
                self._apply_dataset_preprocessing(preprocessing_info)

            # Restore partitions from metadata
            if (
                partitions and self.mdataset is not None
            ):  # FIXED: Check mdataset is not None
                self.logger.debug("Restoring partitions from metadata...")
                for split_name, split_partitions in partitions.items():
                    if (
                        split_name in self.mdataset.available_splits
                    ):  # Now safe to access
                        try:
                            self.mdataset.add_partitions(split_name, split_partitions)
                            self.logger.debug(
                                f"Restored partitions for split: {split_name}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Failed to restore partitions for {split_name}: {e}"
                            )
                    else:
                        self.logger.warning(
                            f"Split {split_name} not found in loaded dataset"
                        )

            # Final validation
            if not self.mdataset:
                raise RuntimeError("Dataset loading resulted in None")

            if not self.mdataset.available_splits:  # Now safe to access
                raise RuntimeError("Loaded dataset has no available splits")

            # Mark as successfully loaded
            self.dataset_loaded = True

            self.logger.info("Dataset lazy loaded successfully!")
            self.logger.info(f"Available splits: {self.mdataset.available_splits}")
            self.logger.info(f"Feature columns: {self.feature_columns}")
            self.logger.info(f"Label column: {self.label_column}")

            # Validate that the required columns exist after preprocessing
            if (
                self.data_partition is not None
                and self.mdataset is not None
                and self.split in self.mdataset.available_splits
            ):  # FIXED: All None checks
                split_dataset = self.mdataset.get_split(self.split)  # Now safe
                dataset_columns = split_dataset.column_names

                self.logger.debug(f"Dataset columns after loading: {dataset_columns}")

                # Check if label column exists
                if (
                    self.label_column is not None
                    and self.label_column not in dataset_columns
                ):  # FIXED: None check
                    self.logger.error(
                        f"Label column '{self.label_column}' not found in dataset columns: {dataset_columns}"
                    )
                    raise ValueError(
                        f"Label column '{self.label_column}' not found after dataset loading. Available columns: {dataset_columns}"
                    )

                # Check if feature columns exist
                if self.feature_columns is not None:  # FIXED: None check
                    missing_features = [
                        col
                        for col in self.feature_columns
                        if col not in dataset_columns
                    ]
                    if missing_features:
                        self.logger.error(
                            f"Feature columns {missing_features} not found in dataset columns: {dataset_columns}"
                        )
                        raise ValueError(
                            f"Feature columns {missing_features} not found after dataset loading. Available columns: {dataset_columns}"
                        )

                # Validate partition data
                actual_size = len(split_dataset)
                partition_size = len(self.data_partition)
                max_idx = max(self.data_partition) if self.data_partition else -1

                self.logger.debug(
                    f"Partition validation - Split size: {actual_size}, Partition size: {partition_size}, Max index: {max_idx}"
                )

                if max_idx >= actual_size:
                    raise ValueError(
                        f"Partition index {max_idx} exceeds dataset size {actual_size}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to lazy load dataset: {e}")
            self.logger.error(f"Dataset metadata: {self.dataset_metadata}")
            self.logger.error(f"Node ID: {self.node_info.get('node_id', 'unknown')}")

            # Set failure state
            self.dataset_loaded = False
            self.mdataset = None

            raise RuntimeError(
                f"Lazy loading failed on node {self.node_info['node_id']}: {e}"
            )

    def _apply_dataset_preprocessing(self, preprocessing_info: Dict[str, Any]) -> None:
        """
        Apply dataset preprocessing on the remote node to recreate transformations
        that were applied on the main node.
        """
        self.logger.info("Applying dataset preprocessing transformations...")

        try:
            # Handle label encoding (most common case)
            if "label_encoding" in preprocessing_info:
                label_info = preprocessing_info["label_encoding"]
                source_column = label_info.get("source_column", "dx")
                target_column = label_info.get("target_column", "label")
                label_mapping = label_info.get("mapping", {})

                if not label_mapping:
                    self.logger.error("Label mapping is empty in preprocessing info")
                    raise ValueError("Label mapping is required for label encoding")

                self.logger.info(
                    f"Applying label encoding: {source_column} -> {target_column}"
                )
                self.logger.info(f"Label mapping: {label_mapping}")

                # Define the mapping function
                def add_label_column(example):
                    if source_column in example:
                        source_value = example[source_column]
                        if source_value in label_mapping:
                            example[target_column] = label_mapping[source_value]
                        else:
                            # Handle unknown labels gracefully
                            self.logger.warning(
                                f"Unknown label value: {source_value}, using 0"
                            )
                            example[target_column] = 0
                    else:
                        self.logger.warning(
                            f"Source column {source_column} not found in example"
                        )
                        example[target_column] = 0
                    return example

                # Apply to all splits - with proper None check
                if self.mdataset is not None:  # FIXED: None check
                    for split_name in self.mdataset.available_splits:
                        self.logger.debug(
                            f"Applying label encoding to split: {split_name}"
                        )
                        split_dataset = self.mdataset.get_split(split_name)

                        # Check if target column already exists
                        if target_column not in split_dataset.column_names:
                            processed_split = split_dataset.map(add_label_column)
                            self.mdataset._splits[split_name] = processed_split
                            self.logger.debug(
                                f"Added {target_column} column to split {split_name}"
                            )
                        else:
                            self.logger.debug(
                                f"Target column {target_column} already exists in split {split_name}"
                            )

                self.logger.info("Label encoding applied successfully")

            # Handle other preprocessing types as needed
            if "feature_transformations" in preprocessing_info:
                self.logger.info("Feature transformations found in preprocessing info")
                # Add feature transformation logic here if needed

            # Add more preprocessing types as your framework grows

        except Exception as e:
            self.logger.error(f"Failed to apply dataset preprocessing: {e}")
            raise RuntimeError(
                f"Dataset preprocessing failed on node {self.node_info['node_id']}: {e}"
            )

    def _get_partition_data(
        self, data_sampling_rate: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from the client's dataset partition with comprehensive validation.

        Args:
            data_sampling_rate: Fraction of local data to sample (1.0 = use all data)
        """
        # Step 1: Validate all required components
        validation_errors = []

        if self.data_partition is None:
            validation_errors.append("data_partition not set")

        if self.feature_columns is None:
            validation_errors.append("feature_columns not set")

        if self.label_column is None:
            validation_errors.append("label_column not set")

        # Step 2: Handle dataset loading (lazy or direct)
        if self.mdataset is None:
            if self.lazy_loading and self.dataset_metadata:
                self.logger.info("Dataset not loaded, triggering lazy loading...")
                try:
                    self._lazy_load_dataset()
                    if self.mdataset is None:
                        validation_errors.append(
                            "lazy loading completed but dataset is still None"
                        )
                    else:
                        self.logger.info("Lazy loading completed successfully")
                except Exception as e:
                    validation_errors.append(f"lazy loading failed: {e}")
                    self.logger.error(f"Lazy loading failed: {e}")
            else:
                validation_errors.append(
                    "dataset not available and lazy loading not properly configured"
                )

        # Step 3: Final validation
        if validation_errors:
            error_msg = "Cannot extract partition data: " + ", ".join(validation_errors)

            # Comprehensive debug logging
            self.logger.error("Actor state validation failed:")
            self.logger.error(
                f"  - data_partition: {self.data_partition is not None} (size: {len(self.data_partition) if self.data_partition else 0})"
            )
            self.logger.error(f"  - feature_columns: {self.feature_columns}")
            self.logger.error(f"  - label_column: {self.label_column}")
            self.logger.error(f"  - mdataset: {self.mdataset is not None}")
            self.logger.error(f"  - lazy_loading: {self.lazy_loading}")
            self.logger.error(f"  - dataset_loaded: {self.dataset_loaded}")
            self.logger.error(
                f"  - has_dataset_metadata: {self.dataset_metadata is not None}"
            )
            self.logger.error(
                f"  - node_id: {self.node_info.get('node_id', 'unknown')}"
            )

            if self.dataset_metadata:
                self.logger.error(
                    f"  - dataset_metadata keys: {list(self.dataset_metadata.keys())}"
                )
                self.logger.error(
                    f"  - dataset_name: {self.dataset_metadata.get('dataset_name')}"
                )
                self.logger.error(
                    f"  - available_splits: {self.dataset_metadata.get('available_splits')}"
                )

            raise ValueError(error_msg)

        # Step 4: Extract data - NOW ALL VARIABLES ARE GUARANTEED NON-NULL
        try:
            # These are now guaranteed to be non-None due to validation above
            assert self.mdataset is not None
            assert self.data_partition is not None
            assert self.feature_columns is not None
            assert self.label_column is not None

            split_dataset = self.mdataset.get_split(self.split)
            partition_dataset = split_dataset.select(self.data_partition)

            # Apply data subsampling if requested
            if data_sampling_rate < 1.0:
                original_size = len(partition_dataset)
                num_samples = max(1, int(original_size * data_sampling_rate))

                # Create random indices for subsampling
                import random

                sampled_indices = random.sample(range(original_size), num_samples)
                partition_dataset = partition_dataset.select(sampled_indices)

                self.logger.info(
                    f"Data subsampling: using {num_samples}/{original_size} samples "
                    f"(sampling rate: {data_sampling_rate:.2f})"
                )

            # Extract features
            if len(self.feature_columns) == 1:
                feature_data = partition_dataset[self.feature_columns[0]]

                # Use model's preprocessor if available
                if (
                    self.model is not None  # FIXED: None check
                    and hasattr(self.model, "data_preprocessor")
                ):
                    if self.model.data_preprocessor is not None:
                        try:
                            data_list = (
                                list(feature_data)
                                if not isinstance(feature_data, list)
                                else feature_data
                            )
                            features = self.model.data_preprocessor.preprocess_features(
                                data_list
                            )
                            self.logger.debug(
                                "Used model's preprocessor for feature extraction"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Model preprocessor failed, using fallback: {e}"
                            )
                            features = np.array(feature_data, dtype=np.float32)
                    else:
                        features = np.array(feature_data, dtype=np.float32)
                else:
                    features = np.array(feature_data, dtype=np.float32)
            else:
                # Multiple feature columns
                processed_columns = []
                for col in self.feature_columns:
                    col_data = partition_dataset[col]
                    if (
                        self.model is not None  # FIXED: None check
                        and hasattr(self.model, "data_preprocessor")
                    ):
                        if self.model.data_preprocessor is not None:
                            try:
                                col_features = (
                                    self.model.data_preprocessor.preprocess_features(
                                        list(col_data)
                                    )
                                )
                                processed_columns.append(col_features)
                            except Exception:
                                processed_columns.append(
                                    np.array(col_data, dtype=np.float32)
                                )
                        else:
                            processed_columns.append(
                                np.array(col_data, dtype=np.float32)
                            )
                    else:
                        processed_columns.append(np.array(col_data, dtype=np.float32))

                features = np.column_stack(processed_columns)

            # Extract labels
            label_data = partition_dataset[self.label_column]

            # Handle string labels with proper error messaging
            if len(label_data) > 0 and isinstance(label_data[0], str):
                self.logger.error(
                    f"Found string labels in column '{self.label_column}'. "
                    "Labels should be converted to integers in the example script before training."
                )
                # Emergency fallback
                unique_labels = sorted(set(label_data))
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                # FIXED: Proper numpy array creation
                labels_list = [label_map[label] for label in label_data]
                labels = np.array(labels_list, dtype=np.int64)
                self.logger.warning(f"Emergency label encoding applied: {label_map}")
            else:
                labels = np.array(label_data, dtype=np.int64)

            self.logger.debug(
                f"Successfully extracted features shape: {features.shape}, labels shape: {labels.shape}"
            )

            # Log label distribution for debugging
            unique_labels_arr, counts = np.unique(labels, return_counts=True)
            self.logger.debug(
                f"Label distribution: {dict(zip(unique_labels_arr, counts))}"
            )

            return features, labels

        except Exception as e:
            self.logger.error(f"Error during data extraction: {e}")
            self.logger.error(f"  - split: {self.split}")
            if self.mdataset is not None:  # FIXED: None check
                self.logger.error(
                    f"  - available splits: {self.mdataset.available_splits}"
                )
            else:
                self.logger.error("  - mdataset is None")
            if self.data_partition is not None:  # FIXED: None check
                self.logger.error(f"  - partition size: {len(self.data_partition)}")
            else:
                self.logger.error("  - data_partition is None")
            raise RuntimeError(
                f"Data extraction failed on node {self.node_info['node_id']}: {e}"
            )

    def train_model(self, **kwargs) -> Dict[str, float]:
        """
        Train the model on the client's dataset partition.

        :param kwargs: Additional parameters for training
        :return: Training metrics
        """
        if self.model is None:
            raise ValueError("Model is not set")

        try:
            self.logger.debug(
                f"Starting training with {len(self.data_partition) if self.data_partition else 0} samples"
            )

            # Extract callback if provided
            client_id = self.client_id
            orig_verbose = kwargs.get("verbose", False)

            # Create a callback for epoch logging that includes node info
            def log_epoch_callback(epoch, total_epochs, metrics):
                if orig_verbose:
                    self.logger.info(
                        f"Epoch [{epoch}/{total_epochs}], "
                        f"Loss: {metrics['loss']:.4f}, "
                        f"Accuracy: {metrics['accuracy']:.4f}"
                    )

            # Add callback to kwargs
            kwargs["log_epoch_callback"] = log_epoch_callback

            # Get data sampling rate from kwargs or use default
            data_sampling_rate = kwargs.pop("data_sampling_rate", 1.0)
            features, labels = self._get_partition_data(data_sampling_rate)
            original_features, original_labels = features.copy(), labels.copy()

            # Apply attack if enabled
            attack_info = {}
            if self.attack_enabled and self.attack_instance:
                # Check if this is a gradual attack
                is_gradual_attack = (
                    (isinstance(self.attack_config, dict) and 
                     self.attack_config.get("attack_type") == "gradual_label_flipping") or
                    hasattr(self.attack_config, '__dict__')  # AttackConfig object
                )
                
                if is_gradual_attack:
                    # Update attack for current round
                    self.attack_instance.update_round(self.current_round)
                    
                    # Apply gradual label flipping
                    features, labels, attack_info = self.attack_instance.poison_labels(
                        features, labels
                    )
                else:
                    # Get current model parameters for parameter-based attacks
                    model_params = None
                    if hasattr(self.model, 'get_parameters'):
                        model_params = self.model.get_parameters()
                    
                    # Apply simple attack to training data
                    features, labels, attack_info = self.attack_instance.apply_attack(
                        features, labels, model_params
                    )
                
                # Log attack information
                if attack_info.get("attack_applied", False):
                    attack_type = (
                        self.attack_config.get('attack_type') if isinstance(self.attack_config, dict)
                        else 'gradual_label_flipping'
                    )
                    self.logger.warning(
                        f"Round {self.current_round}: Applied {attack_type} "
                        f"with intensity {attack_info.get('intensity', 0):.3f}"
                    )
                
                # Store attack history
                self.attack_history.append({
                    "round": self.current_round,
                    "timestamp": time.time(),
                    **attack_info
                })

            # Ensure model is on correct device for multi-node environment
            if hasattr(self.model, "detect_and_set_device"):
                self.model.detect_and_set_device()

            # Train with potentially modified data
            result = self.model.train(features, labels, **kwargs)
            
            # Apply post-training parameter manipulation if needed
            if (self.attack_enabled and self.attack_instance and 
                attack_info.get("manipulated_params")):
                # Replace model parameters with manipulated ones
                manipulated_params = attack_info["manipulated_params"]
                self.model.set_parameters(manipulated_params)
                
                self.logger.debug(
                    f"Applied parameter manipulation to {len(manipulated_params)} parameters"
                )
            
            # Potentially manipulate reported metrics to appear normal
            stealth_mode = (
                self.attack_config.get("stealth_mode", True) if isinstance(self.attack_config, dict)
                else getattr(self.attack_config, 'stealth_mode', True)
            )
            if self.attack_enabled and stealth_mode:
                result = self._manipulate_metrics(result, attack_info)

            self.logger.debug(
                f"Training completed - Loss: {result.get('loss', 'N/A'):.4f}, "
                f"Accuracy: {result.get('accuracy', 'N/A'):.4f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(
                f"Features type: {type(features) if 'features' in locals() else 'N/A'}"
            )
            self.logger.error(
                f"Labels type: {type(labels) if 'labels' in locals() else 'N/A'}"
            )
            if "features" in locals():
                self.logger.error(
                    f"Features shape: {getattr(features, 'shape', 'N/A')}"
                )
                self.logger.error(
                    f"Features dtype: {getattr(features, 'dtype', 'N/A')}"
                )
            raise

    def evaluate_model(self, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the client's dataset partition.

        :param kwargs: Additional parameters for evaluation
        :return: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model is not set")

        try:
            self.logger.debug(
                f"Starting evaluation with {len(self.data_partition) if self.data_partition else 0} samples"
            )

            features, labels = self._get_partition_data()

            # Ensure model is on correct device
            if hasattr(self.model, "detect_and_set_device"):
                self.model.detect_and_set_device()

            result = self.model.evaluate(features, labels, **kwargs)

            self.logger.debug(
                f"Evaluation completed - Loss: {result.get('loss', 'N/A'):.4f}, "
                f"Accuracy: {result.get('accuracy', 'N/A'):.4f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    def predict(self, data: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate predictions using the client's model.

        Args:
            data: Input data for prediction. If None, uses the client's data partition.
            **kwargs: Additional prediction parameters

        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("Model not set")

        try:
            # If no specific data is provided, use the client's partition data
            if data is None:
                if self.mdataset is None or self.data_partition is None:
                    raise ValueError(
                        "No data provided and client dataset/partition not set"
                    )
                features, _ = self._get_partition_data()
            else:
                features = data

            # Ensure model is on correct device
            if hasattr(self.model, "detect_and_set_device"):
                self.model.detect_and_set_device()

            result = self.model.predict(features, **kwargs)

            self.logger.debug(f"Prediction completed for {len(features)} samples")

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the current model parameters.

        :return: Dictionary containing model parameters.
        """
        if self.model is None:
            raise ValueError("Model is not set")

        try:
            result = self.model.get_parameters()
            self.logger.debug("Model parameters retrieved")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get model parameters: {e}")
            raise

    def set_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the model parameters.

        :param parameters: Dictionary containing model parameters to set.
        """
        if self.model is None:
            raise ValueError("Model is not set")

        try:
            self.model.set_parameters(parameters)
            self.logger.debug("Model parameters updated")
        except Exception as e:
            self.logger.error(f"Failed to set model parameters: {e}")
            raise

    def set_neighbours(self, neighbours: List[Any]) -> None:
        """
        Set neighbour actors for communication

        :param neighbours: Neighbour actors
        """
        self.neighbours = neighbours
        self.logger.debug(f"Set {len(neighbours)} neighbours")
    
    def set_actor_references(self, actors: List[Any]) -> None:
        """
        Set references to all actors for gossip aggregation.
        
        :param actors: List of all actor references
        """
        self._actor_references = actors
        self.logger.debug(f"Set references to {len(actors)} actors")

    def get_neighbours(self) -> List[str]:
        """
        Get IDs of neighbouring clients

        :return: IDs of neighbouring clients
        """
        try:
            neighbour_ids = [
                ray.get(n.get_id.remote(), timeout=300) for n in self.neighbours
            ]
            return neighbour_ids
        except Exception as e:
            self.logger.error(f"Failed to get neighbour IDs: {e}")
            return []

    def get_id(self) -> str:
        """
        Return ID of client actor

        :return: ID of client actor
        """
        return self.client_id

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information for debugging and monitoring

        :return: System information dictionary
        """
        try:
            # Memory usage
            memory_info = {}
            if torch.cuda.is_available():
                memory_info.update(
                    {
                        "gpu_memory_allocated": torch.cuda.memory_allocated(),
                        "gpu_memory_cached": torch.cuda.memory_reserved(),
                        "gpu_max_memory_allocated": torch.cuda.max_memory_allocated(),
                    }
                )

            return {
                "client_id": self.client_id,
                "node_info": self.node_info,
                "device_info": self.device_info,
                "memory_info": memory_info,
                "data_partition_size": len(self.data_partition)
                if self.data_partition
                else 0,
                "has_model": self.model is not None,
                "has_dataset": self.mdataset is not None,
                "num_neighbours": len(self.neighbours),
            }
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the actor

        :return: Health status dictionary
        """
        import time

        try:
            timestamp = int(time.time() * 1000)

            status = {
                "client_id": self.client_id,
                "node_id": self.node_info["node_id"],
                "status": "healthy",
                "timestamp": timestamp,
                "checks": {},
            }

            # Check model
            status["checks"]["model"] = "ok" if self.model is not None else "missing"

            # Check dataset
            status["checks"]["dataset"] = (
                "ok" if self.mdataset is not None else "missing"
            )

            # Check data partition
            status["checks"]["data_partition"] = (
                "ok" if self.data_partition is not None else "missing"
            )

            # Check GPU if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    status["checks"]["gpu"] = "ok"
                except Exception:
                    status["checks"]["gpu"] = "error"

            # Overall status
            failed_checks = [k for k, v in status["checks"].items() if v != "ok"]
            if failed_checks:
                status["status"] = "degraded"
                status["failed_checks"] = failed_checks

            return status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "client_id": self.client_id,
                "status": "error",
                "error": str(e),
                "timestamp": int(time.time() * 1000),
            }
    
    def is_malicious(self) -> bool:
        """
        Check if this client is malicious.
        
        Returns:
            True if client has attack capabilities enabled
        """
        return self.attack_enabled
    
    def set_round_number(self, round_num: int) -> None:
        """
        Update the current round number for attack progression.
        
        Args:
            round_num: Current federated learning round
        """
        self.current_round = round_num
        if self.attack_instance:
            self.attack_instance.update_round(round_num)
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """
        Get attack summary for this client.
        
        Returns:
            Attack summary with statistics
        """
        if not self.attack_enabled:
            return {"attack_enabled": False}
        
        total_attacks = len([h for h in self.attack_history if h.get("attack_applied", False)])
        
        summary = {
            "attack_enabled": True,
            "attack_type": (
                self.attack_config.get("attack_type") if isinstance(self.attack_config, dict)
                else getattr(self.attack_config, 'attack_type', None)
            ),
            "total_rounds": len(self.attack_history),
            "attacks_applied": total_attacks,
            "attack_rate": total_attacks / max(1, len(self.attack_history)),
            "current_intensity": getattr(self.attack_instance, "current_intensity", 0),
            "stealth_mode": (
                self.attack_config.get("stealth_mode", False) if isinstance(self.attack_config, dict)
                else getattr(self.attack_config, 'stealth_mode', False)
            ),
        }
        
        # Add attack-specific statistics
        if self.attack_history:
            last_attack = self.attack_history[-1]
            summary["last_attack_round"] = last_attack.get("round", 0)
            summary["last_attack_intensity"] = last_attack.get("intensity", 0)
        
        return summary
    
    def _manipulate_metrics(
        self, 
        original_metrics: Dict[str, float], 
        attack_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Manipulate reported training metrics to appear normal.
        
        Args:
            original_metrics: True training metrics
            attack_info: Information about applied attack
            
        Returns:
            Potentially manipulated metrics
        """
        if not attack_info.get("attack_applied", False):
            return original_metrics
        
        # Add small amount of noise to metrics to make them appear normal
        manipulated = original_metrics.copy()
        
        # Slightly improve reported accuracy to hide attack impact
        if "accuracy" in manipulated:
            noise = np.random.normal(0, 0.01)  # Small noise
            improvement = (
                self.attack_config.get("metric_manipulation", 0.02) if isinstance(self.attack_config, dict)
                else getattr(self.attack_config, 'metric_manipulation', 0.02)
            )
            manipulated["accuracy"] = min(1.0, original_metrics["accuracy"] + improvement + noise)
        
        # Slightly reduce reported loss
        if "loss" in manipulated:
            noise = np.random.normal(0, 0.01)
            reduction = (
                self.attack_config.get("loss_manipulation", -0.05) if isinstance(self.attack_config, dict)
                else getattr(self.attack_config, 'loss_manipulation', -0.05)
            )
            manipulated["loss"] = max(0.0, original_metrics["loss"] + reduction + noise)
        
        return manipulated
    
    def gossip_aggregate(
        self,
        neighbor_indices: List[int],
        mixing_parameter: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform gossip-based aggregation with neighbors.
        
        This implements true decentralized learning where each node:
        1. Gets parameters from neighbors
        2. Aggregates them with its own parameters
        3. Updates its model
        
        Args:
            neighbor_indices: Indices of neighbor actors to exchange with
            mixing_parameter: Weight for own parameters (neighbors get 1-mixing_parameter)
            
        Returns:
            Dictionary with aggregation results
        """
        try:
            if not self.model:
                return {"success": False, "error": "Model not initialized"}
            
            # Get own parameters
            own_params = self.model.get_parameters()
            
            # Collect neighbor parameters
            neighbor_params_list = []
            successful_exchanges = 0
            
            # Get reference to parent actor list (passed during topology setup)
            if not hasattr(self, '_actor_references'):
                return {
                    "success": False, 
                    "error": "Actor references not set. Topology not properly initialized."
                }
            
            for neighbor_idx in neighbor_indices:
                try:
                    # Get neighbor actor reference
                    neighbor_actor = self._actor_references[neighbor_idx]
                    
                    # Get neighbor's parameters
                    neighbor_params = ray.get(
                        neighbor_actor.get_model_parameters.remote(),
                        timeout=30
                    )
                    neighbor_params_list.append(neighbor_params)
                    successful_exchanges += 1
                    
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get parameters from neighbor {neighbor_idx}: {e}"
                    )
            
            if not neighbor_params_list:
                return {
                    "success": False,
                    "error": "No neighbor parameters received",
                    "attempted": len(neighbor_indices),
                    "successful": 0
                }
            
            # Perform weighted aggregation
            # Own weight vs. average of neighbors
            own_weight = mixing_parameter
            neighbor_weight = (1 - mixing_parameter) / len(neighbor_params_list)
            
            # Aggregate parameters
            aggregated_params = {}
            
            for param_name in own_params:
                # Start with own parameters weighted
                aggregated_params[param_name] = own_weight * own_params[param_name]
                
                # Add weighted neighbor parameters
                for neighbor_params in neighbor_params_list:
                    if param_name in neighbor_params:
                        aggregated_params[param_name] += neighbor_weight * neighbor_params[param_name]
            
            # Update own model with aggregated parameters
            self.model.set_parameters(aggregated_params)
            
            self.logger.debug(
                f"Gossip aggregation completed: {successful_exchanges}/{len(neighbor_indices)} "
                f"neighbors, mixing_parameter={mixing_parameter}"
            )
            
            return {
                "success": True,
                "neighbors_contacted": len(neighbor_indices),
                "successful_exchanges": successful_exchanges,
                "mixing_parameter": mixing_parameter
            }
            
        except Exception as e:
            self.logger.error(f"Gossip aggregation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def trust_weighted_gossip_aggregate(
        self,
        neighbor_indices: List[int],
        trust_weights: Dict[str, float],
        mixing_parameter: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform trust-weighted gossip aggregation with neighbors.
        
        This is similar to gossip_aggregate but uses trust scores to weight neighbors.
        
        Args:
            neighbor_indices: Indices of neighbor actors to exchange with
            trust_weights: Dictionary mapping neighbor index (as string) to trust weight
            mixing_parameter: Weight for own parameters
            
        Returns:
            Dictionary with aggregation results
        """
        try:
            if not self.model:
                return {"success": False, "error": "Model not initialized"}
            
            # Get own parameters
            own_params = self.model.get_parameters()
            
            # Collect neighbor parameters
            neighbor_params_list = []
            neighbor_trust_scores = []
            successful_exchanges = 0
            
            if not hasattr(self, '_actor_references'):
                return {
                    "success": False, 
                    "error": "Actor references not set"
                }
            
            for neighbor_idx in neighbor_indices:
                try:
                    # Get neighbor's parameters
                    neighbor_actor = self._actor_references[neighbor_idx]
                    neighbor_params = ray.get(
                        neighbor_actor.get_model_parameters.remote(),
                        timeout=30
                    )
                    
                    # Get trust weight for this neighbor
                    trust_score = trust_weights.get(str(neighbor_idx), 1.0)
                    
                    neighbor_params_list.append(neighbor_params)
                    neighbor_trust_scores.append(trust_score)
                    successful_exchanges += 1
                    
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get parameters from neighbor {neighbor_idx}: {e}"
                    )
            
            if not neighbor_params_list:
                return {
                    "success": False,
                    "error": "No neighbor parameters received",
                    "attempted": len(neighbor_indices),
                    "successful": 0
                }
            
            # Normalize trust weights
            total_neighbor_trust = sum(neighbor_trust_scores)
            if total_neighbor_trust > 0:
                normalized_trust_scores = [s / total_neighbor_trust for s in neighbor_trust_scores]
            else:
                # If all trust scores are 0, use equal weights
                normalized_trust_scores = [1.0 / len(neighbor_trust_scores)] * len(neighbor_trust_scores)
            
            # Calculate final weights (own weight + neighbor weights)
            own_weight = mixing_parameter
            neighbor_weight_total = 1 - mixing_parameter
            
            # Aggregate parameters
            aggregated_params = {}
            
            for param_name in own_params:
                # Start with own parameters weighted
                aggregated_params[param_name] = own_weight * own_params[param_name]
                
                # Add trust-weighted neighbor parameters
                for i, neighbor_params in enumerate(neighbor_params_list):
                    if param_name in neighbor_params:
                        neighbor_weight = neighbor_weight_total * normalized_trust_scores[i]
                        aggregated_params[param_name] += neighbor_weight * neighbor_params[param_name]
            
            # Update own model
            self.model.set_parameters(aggregated_params)
            
            self.logger.debug(
                f"Trust-weighted gossip completed: {successful_exchanges}/{len(neighbor_indices)} "
                f"neighbors, avg trust score: {np.mean(neighbor_trust_scores):.3f}"
            )
            
            return {
                "success": True,
                "neighbors_contacted": len(neighbor_indices),
                "successful_exchanges": successful_exchanges,
                "mixing_parameter": mixing_parameter,
                "average_trust_score": float(np.mean(neighbor_trust_scores))
            }
            
        except Exception as e:
            self.logger.error(f"Trust-weighted gossip failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_partition_data(self, sample_rate: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get a sample of partition data for validation purposes.
        
        Args:
            sample_rate: Fraction of data to sample (0.0 to 1.0)
            
        Returns:
            Tuple of (features, labels) or None if error
        """
        try:
            features, labels = self._get_partition_data(sample_rate)
            return (features, labels)
        except Exception as e:
            self.logger.warning(f"Failed to get partition data: {e}")
            return None
