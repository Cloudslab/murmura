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
        # Use the new Ray API
        return ray.get_runtime_context().get_node_id()[:8]  # Shortened for readability
    except Exception:
        return "unknown"


def get_node_info() -> Dict[str, Any]:
    """Get detailed node information - standalone function for serialization"""
    try:
        # Use new Ray API methods
        runtime_context = ray.get_runtime_context()
        node_id = runtime_context.get_node_id()[:8]

        # Get worker info safely
        worker_id = "unknown"
        try:
            # Try to get worker ID if available
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
        self.feature_columns: Optional[List[str]] = None
        self.label_column: Optional[str] = None

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
        # Create logger for this actor
        self.logger = logging.getLogger(f"murmura.actor.{self.client_id}")

        # Don't add handlers if they already exist (avoid duplication)
        if not self.logger.handlers:
            # Get current node ID for logging
            current_node_id = get_node_id()

            # Create formatter that includes node and actor information
            formatter = logging.Formatter(
                f"%(asctime)s - %(name)s - [Node:{current_node_id}] - [Actor:{self.client_id}] - %(levelname)s - %(message)s"
            )

            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Set log level based on environment variable or default to INFO
            log_level = os.environ.get("MURMURA_LOG_LEVEL", "INFO")
            self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    def get_node_info(self) -> Dict[str, Any]:
        """Return node information for this actor"""
        return self.node_info

    def receive_data(
        self, data_partition: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Receive a data partition and metadata dictionary.

        :param data_partition: DataPartition instance
        :param metadata: Metadata dictionary
        """
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
        """
        Set the dataset for the client actor.

        :param mdataset: Dataset instance
        :param feature_columns: List of feature column names
        :param label_column: Name of the label column
        """
        self.mdataset = mdataset
        if feature_columns:
            self.feature_columns = feature_columns
        if label_column:
            self.label_column = label_column

        self.logger.debug(
            f"Dataset set with features: {feature_columns}, label: {label_column}"
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
        """
        Set the model for the client actor with proper device handling.

        :param model: Model instance
        """
        self.model = model

        # Ensure model uses proper device detection in multi-node environment
        if hasattr(self.model, "detect_and_set_device"):
            self.model.detect_and_set_device()

        self.logger.debug(
            f"Model set on device: {getattr(self.model, 'device', 'unknown')}"
        )

    def get_data_info(self) -> Dict[str, Any]:
        """
        Return Information about stored data partition.
        """
        info = {
            "client_id": self.client_id,
            "data_size": len(self.data_partition) if self.data_partition else 0,
            "metadata": self.metadata,
            "has_model": self.model is not None,
            "has_dataset": self.mdataset is not None,
            "node_info": self.node_info,
            "device_info": self.device_info,
        }

        self.logger.debug(f"Data info requested: {info['data_size']} samples")
        return info

    def _get_partition_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from the client's dataset partition

        :return: Tuple of features and labels as numpy arrays
        """
        if (
            self.mdataset is None
            or self.data_partition is None
            or self.feature_columns is None
            or self.label_column is None
        ):
            raise ValueError(
                "Dataset, data partition, feature columns, or label column not set"
            )

        split_dataset = self.mdataset.get_split(self.split)
        partition_dataset = split_dataset.select(self.data_partition)

        if len(self.feature_columns) == 1:
            features = np.array(partition_dataset[self.feature_columns[0]])
        else:
            features = np.column_stack(
                [np.array(partition_dataset[col]) for col in self.feature_columns]
            )

        labels = np.array(partition_dataset[self.label_column])

        return features, labels

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
                f"Starting training with {len(self.data_partition)} samples"
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

            features, labels = self._get_partition_data()

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
                f"Starting evaluation with {len(self.data_partition)} samples"
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
