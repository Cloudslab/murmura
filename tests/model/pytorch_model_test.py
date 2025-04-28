import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np

from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper


class SimpleModel(PyTorchModel):
    """A simple PyTorch model for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def model_wrapper():
    """Create a model wrapper with a simple model for testing"""
    model = SimpleModel()
    wrapper = TorchModelWrapper(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.01},
        device="cpu",
        input_shape=(10,)
    )
    return wrapper


def test_initialization():
    """Test initialization of TorchModelWrapper"""
    model = SimpleModel()

    # Default initialization
    wrapper = TorchModelWrapper(model=model)
    assert wrapper.model == model
    assert isinstance(wrapper.loss_fn, nn.CrossEntropyLoss)
    assert isinstance(wrapper.optimizer, torch.optim.Adam)
    assert wrapper.device in ["cpu", "cuda"]

    # Custom initialization
    wrapper = TorchModelWrapper(
        model=model,
        loss_fn=nn.MSELoss(),
        optimizer_class=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.1},
        device="cpu",
        input_shape=(10,)
    )
    assert isinstance(wrapper.loss_fn, nn.MSELoss)
    assert isinstance(wrapper.optimizer, torch.optim.SGD)
    assert wrapper.device == "cpu"
    assert wrapper.input_shape == (10,)


def test_prepare_data(model_wrapper):
    """Test data preparation from numpy to DataLoader"""
    # Create sample data
    data = np.random.rand(5, 10).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0]).astype(np.int64)

    # Test with data and labels
    dataloader = model_wrapper._prepare_data(data, labels, batch_size=2)
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == 2

    # Check first batch
    batch_data, batch_labels = next(iter(dataloader))
    assert batch_data.shape == (2, 10)
    assert batch_labels.shape == (2,)


def test_get_set_parameters(model_wrapper):
    """Test getting and setting model parameters"""
    # Get initial parameters
    params = model_wrapper.get_parameters()

    # Verify parameter keys include model layer names
    assert "fc1.weight" in params
    assert "fc1.bias" in params
    assert "fc2.weight" in params
    assert "fc2.bias" in params

    # Create a copy of the initial parameters for reference
    original_params = {k: np.copy(v) for k, v in params.items()}

    # Modify parameters - use a specific value instead of multiplying
    modified_params = {}
    for key, param in params.items():
        # Set all values to 1.0 for a clear test
        modified_params[key] = np.ones_like(param)

    # Set modified parameters
    model_wrapper.set_parameters(modified_params)

    # Get parameters again and verify they were updated
    new_params = model_wrapper.get_parameters()

    # Check that parameters were updated and are different from original
    for key in params:
        # Parameters should now be all ones
        assert np.allclose(new_params[key], np.ones_like(new_params[key]), rtol=1e-5)

        # Ensure they're different from the original parameters
        # (skip this check if the original was already all ones)
        if not np.allclose(original_params[key], np.ones_like(original_params[key])):
            assert not np.allclose(new_params[key], original_params[key], rtol=1e-5)


def test_train(model_wrapper):
    """Test training functionality"""
    # Create sample data
    data = np.random.rand(10, 10).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).astype(np.int64)

    # Train for a few epochs
    metrics = model_wrapper.train(data, labels, epochs=2, batch_size=2)

    # Verify metrics format
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)
    assert 0 <= metrics["accuracy"] <= 1


def test_evaluate(model_wrapper):
    """Test evaluation functionality"""
    # Create sample data
    data = np.random.rand(10, 10).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).astype(np.int64)

    # Evaluate
    metrics = model_wrapper.evaluate(data, labels, batch_size=2)

    # Verify metrics format
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)
    assert 0 <= metrics["accuracy"] <= 1


def test_predict(model_wrapper):
    """Test prediction functionality"""
    # Create sample data
    data = np.random.rand(5, 10).astype(np.float32)

    # Test class predictions
    predictions = model_wrapper.predict(data, batch_size=2)
    assert predictions.shape == (5,)
    assert np.issubdtype(predictions.dtype, np.integer)

    # Test probability predictions
    probabilities = model_wrapper.predict(data, batch_size=2, return_probs=True)
    assert probabilities.shape == (5, 2)  # 2 classes
    assert np.issubdtype(probabilities.dtype, np.floating)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)  # Probabilities sum to 1


def test_save_load(model_wrapper):
    """Test saving and loading model"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        # Save model
        model_wrapper.save(temp_path)
        assert os.path.exists(temp_path)

        # Get original parameters
        original_params = model_wrapper.get_parameters()

        # Create a new wrapper with same model structure but different parameters
        new_model = SimpleModel()
        new_wrapper = TorchModelWrapper(
            model=new_model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01},
            device="cpu"
        )

        # Verify parameters are different
        new_params = new_wrapper.get_parameters()
        assert not np.array_equal(new_params["fc1.weight"], original_params["fc1.weight"])

        # Load saved model
        new_wrapper.load(temp_path)

        # Verify parameters are now the same
        loaded_params = new_wrapper.get_parameters()
        for key in original_params:
            assert np.array_equal(loaded_params[key], original_params[key])

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_input_shape_handling():
    """Test handling of input shapes"""
    model = SimpleModel()
    wrapper = TorchModelWrapper(
        model=model,
        input_shape=(10,)
    )

    # Create data with different shape that needs reshaping
    data = np.random.rand(5, 2, 5).astype(np.float32)  # Shape doesn't match input_shape
    labels = np.array([0, 1, 0, 1, 0]).astype(np.int64)

    # Should reshape the data to match input_shape
    dataloader = wrapper._prepare_data(data, labels, batch_size=2)
    batch_data, _ = next(iter(dataloader))

    # Verify the shape was corrected
    assert batch_data.shape[1:] == (10,)
