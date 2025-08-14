import os
from typing import Optional, Callable, Dict, Any, Tuple, cast, Union, List

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from murmura.data_processing.data_preprocessor import GenericDataPreprocessor
from murmura.model.model_interface import ModelInterface


class PyTorchModel(nn.Module):
    """
    Base PyTorch model implementation
    """

    def __init__(self):
        super().__init__()


class TorchModelWrapper(ModelInterface):
    """
    Generic PyTorch implementation of the ModelInterface with configurable data preprocessing.
    This wrapper adapts PyTorch models to the unified interface and handles various data formats
    through a pluggable preprocessing system.
    """

    def __init__(
        self,
        model: PyTorchModel,
        loss_fn: Optional[nn.Module] = None,
        optimizer_class: Optional[Callable] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        data_preprocessor: Optional[GenericDataPreprocessor] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the PyTorch model wrapper.

        :param model: The PyTorch model to wrap.
        :param loss_fn: Loss function to use for training.
        :param optimizer_class: Optimizer class to use for training.
        :param optimizer_kwargs: Additional arguments for the optimizer.
        :param device: Device to use for training (e.g., 'cpu', 'cuda').
        :param input_shape: Shape of the input data.
        :param data_preprocessor: Custom data preprocessor. If None, uses auto-detection.
        """
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer_class = optimizer_class or optim.Adam
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.001}
        self.input_shape = input_shape
        self.requested_device = device
        self.seed = seed if seed is not None else 42  # Default seed for reproducibility
        if device is None:
            # We'll detect the actual device when the model is used
            self.device = "cpu"  # Initialize to CPU for safe serialization
        else:
            self.device = device

        # Set up data preprocessor - fix type annotation issue
        self.data_preprocessor: Optional[GenericDataPreprocessor]
        if data_preprocessor is None:
            # Import here to avoid circular imports
            try:
                self.data_preprocessor = GenericDataPreprocessor()
            except ImportError:
                # Fallback to None - will use basic numpy conversion
                self.data_preprocessor = None
        else:
            self.data_preprocessor = data_preprocessor

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
        Enhanced data preparation with pluggable preprocessing.

        :param data: Input data as numpy array or other format.
        :param labels: Corresponding labels as numpy array.
        :param batch_size: Batch size for DataLoader.

        :return: DataLoader object.
        """
        # Use the generic preprocessor if available
        if self.data_preprocessor is not None:
            try:
                # Convert data to list if it's not already - fix type handling
                processed_data: Union[List[Any], np.ndarray]
                if isinstance(data, np.ndarray) and data.dtype == np.object_:
                    processed_data_raw = data.tolist()
                    # Ensure we always get a list, even if tolist() returns a scalar
                    if isinstance(processed_data_raw, list):
                        processed_data = processed_data_raw
                    else:
                        processed_data = [processed_data_raw]
                elif hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                    processed_data = list(data)
                else:
                    processed_data = [data]

                # Ensure we have a list for the preprocessor
                if not isinstance(processed_data, list):
                    processed_data = [processed_data]

                # Use the generic preprocessor
                processed_array = self.data_preprocessor.preprocess_features(
                    processed_data
                )
                data = processed_array

            except Exception as e:
                # Fallback to manual processing if generic preprocessor fails
                print(
                    f"Warning: Generic preprocessor failed ({e}), falling back to manual processing"
                )
                data = self.fallback_data_processing(data)
        else:
            # Fallback processing when no preprocessor is available
            data = self.fallback_data_processing(data)

        # Ensure data is a numpy array with correct dtype
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        # Convert to tensor
        tensor_data = torch.tensor(data, dtype=torch.float32)

        # Handle reshaping if necessary
        if self.input_shape and tensor_data.shape[1:] != self.input_shape:
            tensor_data = tensor_data.reshape(-1, *self.input_shape)

        # Handle labels
        if labels is not None:
            if hasattr(labels, "dtype") and labels.dtype == np.object_:
                try:
                    labels = np.array(labels.tolist())
                except Exception:
                    labels = labels.astype(np.int64)

            tensor_labels = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(tensor_data, tensor_labels)
        else:
            dataset = TensorDataset(tensor_data)

        # Create a generator with fixed seed for reproducible shuffling
        generator = None
        if labels is not None and self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(labels is not None),
            generator=generator,
            worker_init_fn=self._worker_init_fn if self.seed is not None else None
        )

    def _worker_init_fn(self, worker_id: int) -> None:
        """Initialize each worker with a unique but reproducible seed."""
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    @staticmethod
    def fallback_data_processing(data: Any) -> np.ndarray:
        """
        Fallback data processing when no generic preprocessor is available.

        :param data: Input data in any format
        :return: Processed numpy array
        """
        # Handle object dtype arrays (common with HuggingFace datasets)
        if hasattr(data, "dtype") and data.dtype == np.object_:
            try:
                # Try to convert object array to regular array
                if hasattr(data[0], "shape"):
                    # If elements are arrays, stack them
                    data = np.stack(data.tolist())
                else:
                    # Otherwise, convert to regular array
                    data = np.array(data.tolist())
            except (ValueError, AttributeError):
                # If that fails, try element-wise conversion
                processed_data = []
                for item in data:
                    if hasattr(item, "astype"):
                        processed_data.append(item.astype(np.float32))
                    else:
                        processed_data.append(np.array(item, dtype=np.float32))
                data = np.stack(processed_data)

        # Ensure data is in a supported numeric dtype
        if hasattr(data, "dtype") and (
            data.dtype == np.object_ or not np.issubdtype(data.dtype, np.number)
        ):
            try:
                # FIXED: Use intermediate variable to help type inference
                data_array: np.ndarray = np.array(data, dtype=np.float32)
                data = data_array
            except (ValueError, TypeError):
                # Last resort: convert via list
                data = np.array(data.tolist(), dtype=np.float32)

        return data

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

        # Ensure device is properly set
        self.detect_and_set_device()

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

        :return: Dictionary containing evaluation metrics (e.g., loss, accuracy, precision, recall, f1_score).
        """
        # Ensure model is on the correct device before evaluation
        self.detect_and_set_device()

        batch_size = kwargs.get("batch_size", 32)
        dataloader = self._prepare_data(data, labels, batch_size)

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        # For precision, recall, F1 calculation
        all_predictions = []
        all_labels = []

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
                
                # Store predictions and labels for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        # Calculate precision, recall, and F1 score
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Use macro averaging for multiclass scenarios
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
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

        # Ensure device is properly set
        self.detect_and_set_device()

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
