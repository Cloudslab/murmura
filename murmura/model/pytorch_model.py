import os
from typing import Optional, Callable, Dict, Any, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from murmura.model.model_interface import ModelInterface


class PyTorchModel(nn.Module):
    """
    Base PyTorch model implementation
    """

    def __init__(self):
        super().__init__()


class TorchModelWrapper(ModelInterface):
    """
    PyTorch's implementation of the ModelInterface. This wrapper adapts PyTorch models to the unified interface.
    """

    def __init__(
        self,
        model: PyTorchModel,
        loss_fn: Optional[nn.Module] = None,
        optimizer_class: Optional[Callable] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Initialize the PyTorch model wrapper.

        :param model: The PyTorch model to wrap.
        :param loss_fn: Loss function to use for training.
        :param optimizer_class: Optimizer class to use for training.
        :param optimizer_kwargs: Additional arguments for the optimizer.
        :param device: Device to use for training (e.g., 'cpu', 'cuda').
        :param input_shape: Shape of the input data.
        """
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer_class = optimizer_class or optim.Adam
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.001}
        self.input_shape = input_shape
        self.requested_device = device
        if device is None:
            # We'll detect the actual device when the model is used
            self.device = "cpu"  # Initialize to CPU for safe serialization
        else:
            self.device = device

        # Initialize model on CPU first for serialization safety
        self.model.to("cpu")
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )

    def _prepare_data(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_size: int = 32,
    ) -> DataLoader:
        """
        Convert numpy arrays to PyTorch DataLoader.

        :param data: Input data as numpy array.
        :param labels: Corresponding labels as numpy array.
        :param batch_size: Batch size for DataLoader.

        :return: DataLoader object.
        """
        # Convert to tensor and handle reshaping if necessary
        tensor_data = torch.tensor(data, dtype=torch.float32)

        if self.input_shape and tensor_data.shape[1:] != self.input_shape:
            tensor_data = tensor_data.reshape(-1, *self.input_shape)

        if labels is not None:
            tensor_labels = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(tensor_data, tensor_labels)
        else:
            dataset = TensorDataset(tensor_data)

        return DataLoader(dataset, batch_size=batch_size, shuffle=(labels is not None))

    def detect_and_set_device(self) -> None:
        """
        Detect and set the best available device, respecting the requested device.
        Should be called by any method that uses the model.
        """
        # If user explicitly requested a device, use that
        if self.requested_device is not None:
            self.device = self.requested_device
        else:
            # Otherwise, auto-detect the best device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure the model and loss function are on the correct device
        self.model.to(self.device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn.to(self.device)

    def train(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Train the model on the provided data and labels.

        :param data: Input data for training.
        :param labels: Corresponding target labels for the input data.
        :param kwargs: Additional parameters for training:
                       - batch_size: Batch size for training. (default: 32)
                       - epochs: Number of epochs for training. (default: 1)
                       - verbose: Whether to print training progress. (default: False)
                       - log_interval: Interval for logging. (default: 1)

        :return: Dictionary containing training metrics (e.g., loss, accuracy).
        """
        batch_size = kwargs.get("batch_size", 32)
        epochs = kwargs.get("epochs", 1)
        verbose = kwargs.get("verbose", False)
        log_interval = kwargs.get("log_interval", 1)

        dataloader = self._prepare_data(data, labels, batch_size)

        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for epoch in range(epochs):
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = (
                    batch_data.to(self.device),
                    batch_labels.to(self.device),
                )

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.loss_fn(outputs, batch_labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item() * batch_data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += batch_labels.size(0)
                epoch_correct += cast(
                    int, torch.eq(predicted, batch_labels).sum().item()
                )

            # Accumulate metrics
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total

            if verbose and (epoch + 1) % log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / epoch_total:.4f}, "
                    f"Accuracy: {100 * epoch_correct / epoch_total:.2f}"
                )

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def evaluate(
        self, data: np.ndarray, labels: np.ndarray, **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on the provided data and labels.

        :param data: Input data for evaluation.
        :param labels: Corresponding target labels for the input data.
        :param kwargs: Additional parameters for evaluation:
                       - batch_size: Batch size for evaluation. (default: 32)

        :return: Dictionary containing evaluation metrics (e.g., loss, accuracy).
        """
        batch_size = kwargs.get("batch_size", 32)
        dataloader = self._prepare_data(data, labels, batch_size)

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = (
                    batch_data.to(self.device),
                    batch_labels.to(self.device),
                )

                # Forward pass
                outputs = self.model(batch_data)
                loss = self.loss_fn(outputs, batch_labels)

                # Track metrics
                total_loss += loss.item() * batch_data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += cast(int, torch.eq(predicted, batch_labels).sum().item())

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions on the provided data.

        :param data: Input data for prediction.
        :param kwargs: Additional parameters for prediction:
                       - batch_size: Batch size for prediction. (default: 32)
                       - return_probs: Return probabilities instead of class labels. (default: False)

        :return: Predicted labels or values.
        """
        batch_size = kwargs.get("batch_size", 32)
        return_probs = kwargs.get("return_probs", False)

        dataloader = self._prepare_data(data, batch_size=batch_size)

        self.model.eval()
        all_outputs = []

        with torch.no_grad():
            for (batch_data,) in dataloader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)

                if return_probs:
                    outputs = torch.softmax(outputs, dim=1)
                else:
                    _, outputs = torch.max(outputs, 1)

                all_outputs.append(outputs.cpu().numpy())

        return np.concatenate(all_outputs)

    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters as CPU tensors for safe serialization."""
        # Move model to CPU temporarily for serialization safety
        self.model.to("cpu")
        params = {
            name: param.cpu().numpy() for name, param in self.model.state_dict().items()
        }
        # Move back to the original device if set
        if hasattr(self, "device") and self.device != "cpu":
            self.model.to(self.device)
        return params

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set parameters safely regardless of device."""
        # Always load to CPU first
        state_dict = {name: torch.tensor(param) for name, param in parameters.items()}
        self.model.load_state_dict(state_dict)

        # Then move to the target device if needed
        if hasattr(self, "device") and self.device != "cpu":
            self.model.to(self.device)

    def save(self, path: str) -> None:
        """Save the model safely to disk."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Move to CPU for safe serialization
        self.model.to("cpu")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "input_shape": self.input_shape,
                "requested_device": self.requested_device,
            },
            path,
        )

        # Move back to the original device if needed
        if self.device != "cpu":
            self.model.to(self.device)

    def load(self, path: str) -> None:
        """Load the model with proper device mapping."""
        # Always load to CPU first for safety
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.input_shape = checkpoint.get("input_shape", self.input_shape)
        self.requested_device = checkpoint.get(
            "requested_device", self.requested_device
        )

        # Then detect and set the proper device
        self.detect_and_set_device()
