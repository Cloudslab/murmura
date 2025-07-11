"""
Malicious client actor implementation for model poisoning research.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import ray
import torch

from murmura.data_processing.dataset import MDataset
from murmura.model.model_interface import ModelInterface
from murmura.node.client_actor import get_node_id, get_node_info
from murmura.attacks.attack_config import AttackConfig
from murmura.attacks.label_flipping import LabelFlippingAttack
from murmura.attacks.gradient_manipulation import GradientManipulationAttack


@ray.remote
class MaliciousClientActor:
    """
    Malicious client actor that implements the same interface as VirtualClientActor
    but with attack capabilities for model poisoning research.
    """
    
    def __init__(self, client_id: str, attack_config: AttackConfig):
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

        # Attack configuration and state
        self.attack_config = attack_config
        self.is_malicious = True
        self.attack_instances: Dict[str, Any] = {}
        
        # Set up logging for multi-node environment (FIRST!)
        self._setup_logging()
        
        # Initialize attack instances (needs logger)
        self._initialize_attacks()

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
            f"MALICIOUS Actor {client_id} initialized on node {self.node_info['node_id']} "
            f"with device: {self.device_info['device']} and attacks: {list(self.attack_instances.keys())}"
        )

    def _setup_logging(self) -> None:
        """Set up logging for the malicious actor with node information"""
        self.logger = logging.getLogger(f"murmura.malicious_actor.{self.client_id}")

        if not self.logger.handlers:
            current_node_id = get_node_id()
            formatter = logging.Formatter(
                f"%(asctime)s - %(name)s - [Node:{current_node_id}] - [MALICIOUS:{self.client_id}] - %(levelname)s - %(message)s"
            )
            
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_attacks(self):
        """Initialize attack instances based on configuration."""
        attack_dict = self.attack_config.model_dump()
        
        # Initialize label flipping attack
        if self.attack_config.attack_type in ["label_flipping", "both"]:
            self.attack_instances["label_flipping"] = LabelFlippingAttack(
                self.client_id, attack_dict
            )
        
        # Initialize gradient manipulation attack
        if self.attack_config.attack_type in ["gradient_manipulation", "both"]:
            self.attack_instances["gradient_manipulation"] = GradientManipulationAttack(
                self.client_id, attack_dict
            )

    # Copy all the interface methods from VirtualClientActor with malicious modifications
    def get_node_info(self) -> Dict[str, Any]:
        """Return node information for this actor"""
        return self.node_info

    def receive_data(
        self, data_partition: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Receive a data partition and metadata dictionary."""
        self.data_partition = data_partition
        self.metadata = metadata if metadata is not None else {}

        message = f"MALICIOUS Client {self.client_id} received {len(data_partition)} samples"
        self.logger.debug(message)
        return message

    def set_data_partition(self, partition: List[int]) -> None:
        """Set the data partition for this client."""
        self.data_partition = partition
        self.logger.debug(f"Data partition set with {len(partition)} samples")

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata for this client."""
        self.metadata = metadata
        self.logger.debug("Metadata updated")

    def set_neighbours(self, neighbours: List[Any]) -> None:
        """Set the list of neighbor actors for decentralized communication."""
        self.neighbours = neighbours
        self.logger.debug(f"Neighbours set: {len(neighbours)} neighbors")

    def set_model(self, model: ModelInterface) -> None:
        """Set the model for this client."""
        self.model = model
        self.logger.debug("Model set")

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
            f"MALICIOUS Dataset set with features: {feature_columns}, label: {label_column}"
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

        self.logger.info("MALICIOUS Set dataset metadata for lazy loading")
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
            "is_malicious": True,  # Additional info for malicious clients
            "attack_config": self.attack_config.model_dump(),
            "attack_instances": list(self.attack_instances.keys()),
        }

        if self.dataset_metadata:
            info["dataset_name"] = self.dataset_metadata.get("dataset_name")
            info["available_splits"] = self.dataset_metadata.get("available_splits", [])

        return info

    def set_mdataset(
        self,
        mdataset: MDataset,
        feature_columns: List[str],
        label_column: str,
        split: str = "train",
        lazy_loading: bool = False,
    ) -> None:
        """Set the dataset for this client."""
        self.mdataset = mdataset
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.split = split
        self.lazy_loading = lazy_loading

        if not lazy_loading:
            self._load_dataset()

        self.logger.debug(f"Dataset configured with lazy_loading={lazy_loading}")

    def _load_dataset(self) -> None:
        """Load the dataset if not already loaded."""
        if self.dataset_loaded:
            return
            
        # Try lazy loading if dataset metadata is available
        if self.lazy_loading and self.dataset_metadata and self.mdataset is None:
            self._lazy_load_dataset()
            return

        if self.mdataset is None:
            raise ValueError("MDataset is not set")

        try:
            # Dataset should already be loaded by the orchestration process
            # self.mdataset.load_data(self.split)
            self.dataset_loaded = True
            self.logger.debug("Dataset loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def _lazy_load_dataset(self) -> None:
        """Load the dataset on-demand from metadata."""
        if self.mdataset is not None:
            self.logger.debug("Dataset already loaded, skipping lazy loading")
            return

        if not self.dataset_metadata:
            raise ValueError("No dataset metadata available for lazy loading")

        try:
            self.logger.info("MALICIOUS Lazy loading dataset from metadata...")

            # Extract and validate metadata
            dataset_source = self.dataset_metadata.get("dataset_source")
            dataset_name = self.dataset_metadata.get("dataset_name")
            dataset_kwargs = self.dataset_metadata.get("dataset_kwargs", {})
            available_splits = self.dataset_metadata.get("available_splits", [])

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

            # Import here to avoid circular imports
            from murmura.data_processing.dataset import MDataset, DatasetSource

            # Validate dataset source
            if dataset_source != DatasetSource.HUGGING_FACE:
                raise ValueError(
                    f"Lazy loading currently only supports HuggingFace datasets, got: {dataset_source}"
                )

            # Ensure we don't pass duplicate dataset_name in kwargs
            clean_kwargs = {
                k: v for k, v in dataset_kwargs.items() if k != "dataset_name"
            }

            # Load dataset from HuggingFace (load all splits like VirtualClientActor does)
            self.mdataset = MDataset.load(
                dataset_source, 
                dataset_name=dataset_name,
                split=None,  # Load all splits
                **clean_kwargs
            )

            # Apply partitions if available
            partitions = self.dataset_metadata.get("partitions", {})
            if partitions:
                for split_name, partition_data in partitions.items():
                    if split_name in self.mdataset.available_splits:
                        try:
                            self.mdataset.add_partitions(split_name, partition_data)
                            self.logger.debug(f"Restored partitions for split: {split_name}")
                        except Exception as e:
                            self.logger.error(f"Failed to restore partitions for {split_name}: {e}")
                    else:
                        self.logger.warning(f"Split {split_name} not found in loaded dataset")

            self.dataset_loaded = True
            self.lazy_loading = False
            self.logger.info("MALICIOUS Dataset loaded successfully via lazy loading")
            
        except Exception as e:
            self.logger.error(f"Failed to lazy load dataset: {e}")
            raise

    def _get_partition_data(self, data_sampling_rate: float = 1.0) -> Tuple[Any, Any]:
        """Get the data partition for this client."""
        if self.lazy_loading and not self.dataset_loaded:
            self._load_dataset()

        if self.data_partition is None:
            raise ValueError("Data partition is not set")

        if self.mdataset is None:
            raise ValueError("MDataset is not set")

        # Apply data sampling if specified
        partition_to_use: List[int] = []
        if data_sampling_rate < 1.0:
            num_samples = int(len(self.data_partition) * data_sampling_rate)
            if num_samples > 0:
                sampled_indices = np.random.choice(
                    self.data_partition, num_samples, replace=False
                )
                partition_to_use = sampled_indices.tolist()  # type: ignore[assignment]
            else:
                partition_to_use = self.data_partition[:1]  # type: ignore[assignment]
        else:
            partition_to_use = self.data_partition  # type: ignore[assignment]

        try:
            # Get data from the dataset split using efficient select method (like VirtualClientActor)
            split_dataset = self.mdataset.get_split(self.split)
            partition_dataset = split_dataset.select(partition_to_use)
            
            if self.feature_columns is None or len(self.feature_columns) == 0:
                raise ValueError("Feature columns not set")
            
            # Extract features and labels efficiently
            features = partition_dataset[self.feature_columns[0]]
            labels = partition_dataset[self.label_column]
            
            return features, labels
        except Exception as e:
            self.logger.error(f"Failed to get partition data: {e}")
            raise

    def train_model(self, **kwargs) -> Dict[str, float]:
        """
        Train the model with potential poisoning attacks.
        
        Args:
            **kwargs: Training parameters including current_round and total_rounds
            
        Returns:
            Training metrics
        """
        # Extract round information from kwargs (provided by cluster manager)
        current_round = kwargs.get('current_round', 1)
        total_rounds = kwargs.get('total_rounds', 10)
        
        # Remove round params from kwargs to avoid conflicts
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['current_round', 'total_rounds']}
        
        if self.model is None:
            raise ValueError("Model is not set")

        # Check if attacks should be active
        if not self.attack_config.is_attack_active(current_round):
            self.logger.info(f"Attacks inactive for round {current_round}, training normally")
            return self._train_normally(**clean_kwargs)

        # Calculate current attack intensity
        attack_intensity = self.attack_config.get_attack_intensity(current_round, total_rounds)
        
        self.logger.info(f"Executing MALICIOUS training - Round {current_round}, Intensity: {attack_intensity:.3f}")

        # If we have label flipping attack, poison the data during training
        if "label_flipping" in self.attack_instances:
            return self._train_with_label_flipping(current_round, attack_intensity, **clean_kwargs)
        else:
            # Regular training, poisoning will happen in get_model_parameters
            return self._train_normally(**clean_kwargs)

    def _train_normally(self, **kwargs) -> Dict[str, float]:
        """Train normally without attacks."""
        try:
            self.logger.debug(
                f"Starting normal training with {len(self.data_partition) if self.data_partition else 0} samples"
            )

            # Extract callback if provided
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

            # Ensure model is on correct device for multi-node environment
            if self.model is not None and hasattr(self.model, "detect_and_set_device"):
                self.model.detect_and_set_device()

            if self.model is not None:
                result = self.model.train(features, labels, **kwargs)
            else:
                raise ValueError("Model is not set")

            self.logger.debug(
                f"Training completed - Loss: {result.get('loss', 'N/A'):.4f}, "
                f"Accuracy: {result.get('accuracy', 'N/A'):.4f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _train_with_label_flipping(self, current_round: int, attack_intensity: float, **kwargs) -> Dict[str, float]:
        """Train model with label flipping attack."""
        try:
            # Get original training data
            data_sampling_rate = kwargs.pop("data_sampling_rate", 1.0)
            features, labels = self._get_partition_data(data_sampling_rate)
            
            # Apply label flipping attack
            self.logger.info(f"Applying label flipping attack - Round {current_round}, Intensity: {attack_intensity:.3f}")
            label_attack = self.attack_instances["label_flipping"]
            poisoned_features, poisoned_labels = label_attack.poison_data(
                features, labels, current_round, attack_intensity
            )
            
            # Ensure model is on correct device
            if self.model is not None and hasattr(self.model, "detect_and_set_device"):
                self.model.detect_and_set_device()
            
            # Train with poisoned data
            if self.model is not None:
                result = self.model.train(poisoned_features, poisoned_labels, **kwargs)
            else:
                raise ValueError("Model is not set")
            
            self.logger.info(f"MALICIOUS training completed - Round {current_round}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Malicious training failed: {e}")
            raise

    def get_model_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Get model parameters with potential gradient manipulation attacks.
        
        Args:
            **kwargs: Parameters including current_round and total_rounds
            
        Returns:
            Model parameters (potentially poisoned)
        """
        if self.model is None:
            raise ValueError("Model is not set")

        try:
            # Get original parameters
            parameters = self.model.get_parameters()
            
            # Check if round information is provided and apply gradient manipulation
            if 'current_round' in kwargs and 'total_rounds' in kwargs:
                current_round = kwargs['current_round']
                total_rounds = kwargs['total_rounds']
                
                self.logger.debug(f"Round info provided - Round {current_round}/{total_rounds}, checking gradient manipulation...")
                
                # Apply gradient manipulation if active
                if (self.attack_config.is_attack_active(current_round) and 
                    "gradient_manipulation" in self.attack_instances):
                    
                    attack_intensity = self.attack_config.get_attack_intensity(current_round, total_rounds)
                    
                    self.logger.info(f"Applying gradient manipulation - Round {current_round}, Intensity: {attack_intensity:.3f}")
                    
                    grad_attack = self.attack_instances["gradient_manipulation"]
                    parameters = grad_attack.poison_gradients(parameters, current_round, attack_intensity)
                else:
                    self.logger.debug(f"Gradient manipulation not active - attack_active: {self.attack_config.is_attack_active(current_round)}, has_grad_attack: {'gradient_manipulation' in self.attack_instances}")
            else:
                self.logger.debug("No round information provided - returning clean parameters")
            
            self.logger.debug("Model parameters retrieved")
            return parameters
            
        except Exception as e:
            self.logger.error(f"Failed to get model parameters: {e}")
            raise

    def set_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set the model parameters."""
        if self.model is None:
            raise ValueError("Model is not set")

        try:
            self.model.set_parameters(parameters)
            self.logger.debug("Model parameters updated")
        except Exception as e:
            self.logger.error(f"Failed to set model parameters: {e}")
            raise

    def evaluate_model(self, **kwargs) -> Dict[str, float]:
        """Evaluate the model on the client's dataset partition."""
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

    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attack statistics."""
        stats: Dict[str, Any] = {
            "is_malicious": True,
            "client_id": self.client_id,
            "attack_config": self.attack_config.model_dump(),
            "attack_instances": {}
        }
        
        for attack_type, attack_instance in self.attack_instances.items():
            stats["attack_instances"][attack_type] = attack_instance.get_attack_statistics()
        
        return stats

    def get_privacy_spent(self) -> Optional[Dict[str, Any]]:
        """Get privacy metrics from the model wrapper if available."""
        if self.model is None:
            raise ValueError("Model is not set")

        try:
            # Check if the model has the get_privacy_spent method
            if hasattr(self.model, "get_privacy_spent"):
                privacy_metrics = self.model.get_privacy_spent()
                self.logger.debug("Privacy metrics retrieved from model")
                return privacy_metrics
            else:
                self.logger.debug("Model does not support privacy metrics")
                return None
        except Exception as e:
            self.logger.error(f"Failed to get privacy metrics: {e}")
            return None

    def get_node_info(self) -> Dict[str, Any]:
        """Get node information for this actor."""
        return self.node_info

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information for this actor."""
        return self.device_info