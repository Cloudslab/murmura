from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class ModelInterface(ABC):
    """
    Abstract interface for models used in distributed learning.
    Defines the common methods that all model implementations must provide.
    """

    @abstractmethod
    def train(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Train the model on the provided data and labels.

        :param data: Input data for training.
        :param labels: Corresponding target labels for the input data.
        :param kwargs: Additional parameters for training (batch_size, epochs, etc.).

        returns: Dictionary containing training metrics (e.g., loss, accuracy).
        """
        pass

    @abstractmethod
    def evaluate(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the provided data and labels.

        :param data: Input data for evaluation.
        :param labels: Corresponding target labels for the input data.
        :param kwargs: Additional parameters for evaluation (batch_size, metrics, etc.).

        returns: Dictionary containing evaluation metrics (e.g., loss, accuracy).
        """
        pass

    @abstractmethod
    def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions on the provided data.

        :param data: Input data for prediction.
        :param kwargs: Additional parameters for prediction.

        returns: Predicted labels or values.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current model parameters.

        :return: Dictionary containing model parameters.
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the model parameters.

        :param parameters: Dictionary containing model parameters to set.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.

        :param path: Path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from the specified path.

        :param path: Path to load the model from.
        """
        pass
