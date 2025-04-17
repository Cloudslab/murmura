from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import ray

from murmura.data_processing.dataset import MDataset
from murmura.model.model_interface import ModelInterface


@ray.remote
class VirtualClientActor:
    """Ray remote actor representing a virtual client in federated learning."""

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
        return f"Client {self.client_id} received {len(data_partition)} samples"

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

    def set_model(self, model: ModelInterface) -> None:
        """
        Set the model for the client actor.

        :param model: Model instance
        """
        self.model = model

    def get_data_info(self) -> Dict[str, Any]:
        """
        Return Information about stored data partition.
        :return:
        """
        return {
            "client_id": self.client_id,
            "data_size": len(self.data_partition) if self.data_partition else 0,
            "metadata": self.metadata,
            "has_model": self.model is not None,
            "has_dataset": self.mdataset is not None,
        }

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

        features, labels = self._get_partition_data()
        return self.model.train(features, labels, **kwargs)

    def evaluate_model(self, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the client's dataset partition.

        :param kwargs: Additional parameters for evaluation
        :return: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model is not set")

        features, labels = self._get_partition_data()
        return self.model.evaluate(features, labels, **kwargs)

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

        # If no specific data is provided, use the client's partition data
        if data is None:
            if self.mdataset is None or self.data_partition is None:
                raise ValueError(
                    "No data provided and client dataset/partition not set"
                )
            features, _ = self._get_partition_data()
        else:
            features = data

        return self.model.predict(features, **kwargs)

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the current model parameters.

        :return: Dictionary containing model parameters.
        """
        if self.model is None:
            raise ValueError("Model is not set")
        return self.model.get_parameters()

    def set_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the model parameters.

        :param parameters: Dictionary containing model parameters to set.
        """
        if self.model is None:
            raise ValueError("Model is not set")
        self.model.set_parameters(parameters)

    def set_neighbours(self, neighbours: List[Any]) -> None:
        """
        Set neighbour actors for communication

        :param neighbours: Neighbour actors
        """
        self.neighbours = neighbours

    def get_neighbours(self) -> List[str]:
        """
        Get IDs of neighbouring clients

        :return: IDs of neighbouring clients
        """
        return [ray.get(n.get_id.remote()) for n in self.neighbours]

    def get_id(self) -> str:
        """
        Return ID of client actor

        :return: ID of client actor
        """
        return self.client_id
