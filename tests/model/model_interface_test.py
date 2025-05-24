import pytest
import numpy as np
import os
import tempfile
from abc import ABC

from murmura.model.model_interface import ModelInterface


class SimpleModel(ModelInterface):
    """Simple concrete implementation of ModelInterface for testing"""

    def __init__(self):
        self.parameters = {"weight": np.array([1.0, 2.0, 3.0]), "bias": np.array([0.1])}
        self.trained = False
        self.evaluated = False
        self.predicted = False
        self.saved_path = None
        self.loaded_path = None

    def train(self, data, labels, **kwargs):
        self.trained = True
        self.train_data = data
        self.train_labels = labels
        self.train_kwargs = kwargs
        return {"loss": 0.5, "accuracy": 0.8}

    def evaluate(self, data, labels, **kwargs):
        self.evaluated = True
        self.eval_data = data
        self.eval_labels = labels
        self.eval_kwargs = kwargs
        return {"loss": 0.3, "accuracy": 0.9}

    def predict(self, data, **kwargs):
        self.predicted = True
        self.predict_data = data
        self.predict_kwargs = kwargs
        return np.array([0, 1, 0])

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters

    def save(self, path):
        self.saved_path = path
        # Simulate saving by writing a simple file
        with open(path, "w") as f:
            f.write("Test model data")

    def load(self, path):
        self.loaded_path = path
        # Simulate loading by checking if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")


class PartialModel(ABC):
    """Partial implementation of model interface for testing abstract methods"""

    def train(self, data, labels, **kwargs):
        return {"loss": 0.5}

    # Missing other methods


def test_abstract_interface_cannot_be_instantiated():
    """Test that ModelInterface cannot be instantiated directly"""
    with pytest.raises(TypeError):
        ModelInterface()


def test_partial_implementation_cannot_be_instantiated():
    """Test that partial implementation cannot be instantiated"""

    class IncompleteModel(ModelInterface):
        def train(self, data, labels, **kwargs):
            return {"loss": 0.5}

        # Missing other required methods

    with pytest.raises(TypeError):
        IncompleteModel()


def test_train_with_complex_parameters():
    """Test training with complex parameters"""
    model = SimpleModel()

    # Prepare training data
    data = np.random.randn(10, 5)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Additional training parameters
    kwargs = {
        "epochs": 10,
        "batch_size": 2,
        "learning_rate": 0.01,
        "verbose": True,
        "callbacks": [lambda x: None],
    }

    # Train model
    metrics = model.train(data, labels, **kwargs)

    # Verify function was called with correct parameters
    assert model.trained is True
    assert np.array_equal(model.train_data, data)
    assert np.array_equal(model.train_labels, labels)

    # Verify all kwargs were passed through
    for key, value in kwargs.items():
        assert key in model.train_kwargs
        assert model.train_kwargs[key] == value

    # Check metrics format
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_evaluate_with_complex_parameters():
    """Test evaluating with complex parameters"""
    model = SimpleModel()

    # Prepare evaluation data
    data = np.random.randn(5, 5)
    labels = np.array([0, 1, 0, 1, 0])

    # Additional evaluation parameters
    kwargs = {"batch_size": 1, "metrics": ["precision", "recall"], "verbose": False}

    # Evaluate model
    metrics = model.evaluate(data, labels, **kwargs)

    # Verify function was called with correct parameters
    assert model.evaluated is True
    assert np.array_equal(model.eval_data, data)
    assert np.array_equal(model.eval_labels, labels)

    # Verify all kwargs were passed through
    for key, value in kwargs.items():
        assert key in model.eval_kwargs
        assert model.eval_kwargs[key] == value

    # Check metrics format
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_predict_with_complex_parameters():
    """Test predicting with complex parameters"""
    model = SimpleModel()

    # Prepare prediction data
    data = np.random.randn(3, 5)

    # Additional prediction parameters
    kwargs = {"batch_size": 1, "return_probs": True, "threshold": 0.5}

    # Make predictions
    predictions = model.predict(data, **kwargs)

    # Verify function was called with correct parameters
    assert model.predicted is True
    assert np.array_equal(model.predict_data, data)

    # Verify all kwargs were passed through
    for key, value in kwargs.items():
        assert key in model.predict_kwargs
        assert model.predict_kwargs[key] == value

    # Check predictions format
    assert isinstance(predictions, np.ndarray)


def test_get_set_parameters():
    """Test getting and setting parameters"""
    model = SimpleModel()

    # Get initial parameters
    initial_params = model.get_parameters()
    assert "weight" in initial_params
    assert "bias" in initial_params

    # Define new parameters
    new_params = {"weight": np.array([4.0, 5.0, 6.0]), "bias": np.array([0.2])}

    # Set new parameters
    model.set_parameters(new_params)

    # Get updated parameters
    updated_params = model.get_parameters()

    # Verify parameters were updated
    assert np.array_equal(updated_params["weight"], new_params["weight"])
    assert np.array_equal(updated_params["bias"], new_params["bias"])


def test_save_and_load():
    """Test saving and loading models"""
    model = SimpleModel()

    # Create a temporary file for saving/loading
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Save the model
        model.save(temp_path)

        # Verify the model was saved
        assert model.saved_path == temp_path
        assert os.path.exists(temp_path)

        # Load the model
        model.load(temp_path)

        # Verify the model was loaded
        assert model.loaded_path == temp_path

        # Test loading from a non-existent path
        with pytest.raises(FileNotFoundError):
            model.load("/non/existent/path.model")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_model_interface_structure():
    """Test the structure of ModelInterface"""
    # Verify ModelInterface is an abstract class
    assert issubclass(ModelInterface, ABC)

    # Check if all required methods are defined as abstract methods
    abstract_methods = [
        "train",
        "evaluate",
        "predict",
        "get_parameters",
        "set_parameters",
        "save",
        "load",
    ]

    for method_name in abstract_methods:
        method = getattr(ModelInterface, method_name)
        # Abstract methods should have __isabstractmethod__ attribute
        assert hasattr(method, "__isabstractmethod__")
        assert method.__isabstractmethod__ is True


def test_model_interface_doc_strings():
    """Test documentation strings in ModelInterface"""
    # Check class docstring
    assert ModelInterface.__doc__ is not None
    assert len(ModelInterface.__doc__) > 0

    # Check method docstrings
    methods = [
        "train",
        "evaluate",
        "predict",
        "get_parameters",
        "set_parameters",
        "save",
        "load",
    ]

    for method_name in methods:
        method = getattr(ModelInterface, method_name)
        assert method.__doc__ is not None
        assert len(method.__doc__) > 0
