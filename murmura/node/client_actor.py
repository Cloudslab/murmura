import logging
import os
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

    def __init__(self, client_id: str) -> None:
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

        # Set up logging for multi-node environment
        self._setup_logging()

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

    def _get_partition_data(self, data_sampling_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
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
            data_sampling_rate = kwargs.pop('data_sampling_rate', 1.0)
            features, labels = self._get_partition_data(data_sampling_rate)

            # Ensure model is on correct device for multi-node environment
            if hasattr(self.model, "detect_and_set_device"):
                self.model.detect_and_set_device()

            result = self.model.train(features, labels, **kwargs)

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

    def get_neighbours(self) -> List[str]:
        """
        Get IDs of neighbouring clients

        :return: IDs of neighbouring clients
        """
        try:
            neighbour_ids = [ray.get(n.get_id.remote()) for n in self.neighbours]
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
