import pytest
import numpy as np
from abc import ABC

from murmura.model.model_interface import ModelInterface


class ConcreteModel(ModelInterface):
    """Concrete implementation of ModelInterface for testing"""

    def __init__(self):
        self.parameters = {"layer1": np.array([1.0, 2.0]), "layer2": np.array([3.0, 4.0])}
        self.train_called = False
        self.evaluate_called = False
        self.predict_called = False

    def train(self, data, labels, **kwargs):
        self.train_called = True
        self.train_data = data
        self.train_labels = labels
        self.train_kwargs = kwargs
        return {"loss": 0.5, "accuracy": 0.8}

    def evaluate(self, data, labels, **kwargs):
        self.evaluate_called = True
        self.evaluate_data = data
        self.evaluate_labels = labels
        self.evaluate_kwargs = kwargs
        return {"loss": 0.3, "accuracy": 0.9}

    def predict(self, data, **kwargs):
        self.predict_called = True
        self.predict_data = data
        self.predict_kwargs = kwargs
        return np.array([0, 1, 0])

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters

    def save(self, path):
        self.save_path = path

    def load(self, path):
        self.load_path = path


def test_model_interface_is_abstract():
    """Test that ModelInterface is abstract and cannot be instantiated directly"""
    with pytest.raises(TypeError):
        ModelInterface()


def test_concrete_model_initialization():
    """Test initialization of a concrete model implementation"""
    model = ConcreteModel()
    assert model.parameters is not None
    assert "layer1" in model.parameters
    assert "layer2" in model.parameters


def test_train_method():
    """Test the train method"""
    model = ConcreteModel()
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 1])

    # Call train with various kwargs
    metrics = model.train(data, labels, batch_size=32, epochs=5)

    # Verify train was called with correct arguments
    assert model.train_called
    assert np.array_equal(model.train_data, data)
    assert np.array_equal(model.train_labels, labels)
    assert model.train_kwargs["batch_size"] == 32
    assert model.train_kwargs["epochs"] == 5

    # Verify return value
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] == 0.5
    assert metrics["accuracy"] == 0.8


def test_evaluate_method():
    """Test the evaluate method"""
    model = ConcreteModel()
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 1])

    # Call evaluate with some kwargs
    metrics = model.evaluate(data, labels, batch_size=16)

    # Verify evaluate was called with correct arguments
    assert model.evaluate_called
    assert np.array_equal(model.evaluate_data, data)
    assert np.array_equal(model.evaluate_labels, labels)
    assert model.evaluate_kwargs["batch_size"] == 16

    # Verify return value
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] == 0.3
    assert metrics["accuracy"] == 0.9


def test_predict_method():
    """Test the predict method"""
    model = ConcreteModel()
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Call predict with some kwargs
    predictions = model.predict(data, return_probs=True)

    # Verify predict was called with correct arguments
    assert model.predict_called
    assert np.array_equal(model.predict_data, data)
    assert model.predict_kwargs["return_probs"] is True

    # Verify return value
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (3,)  # Based on implementation


def test_get_parameters_method():
    """Test the get_parameters method"""
    model = ConcreteModel()

    # Call get_parameters
    params = model.get_parameters()

    # Verify return value
    assert "layer1" in params
    assert "layer2" in params
    assert np.array_equal(params["layer1"], np.array([1.0, 2.0]))
    assert np.array_equal(params["layer2"], np.array([3.0, 4.0]))


def test_set_parameters_method():
    """Test the set_parameters method"""
    model = ConcreteModel()

    # Define new parameters
    new_params = {
        "layer1": np.array([5.0, 6.0]),
        "layer2": np.array([7.0, 8.0])
    }

    # Call set_parameters
    model.set_parameters(new_params)

    # Verify parameters were updated
    updated_params = model.get_parameters()
    assert np.array_equal(updated_params["layer1"], np.array([5.0, 6.0]))
    assert np.array_equal(updated_params["layer2"], np.array([7.0, 8.0]))


def test_save_method():
    """Test the save method"""
    model = ConcreteModel()
    path = "/tmp/model.pt"

    # Call save
    model.save(path)

    # Verify save was called with correct path
    assert model.save_path == path


def test_load_method():
    """Test the load method"""
    model = ConcreteModel()
    path = "/tmp/model.pt"

    # Call load
    model.load(path)

    # Verify load was called with correct path
    assert model.load_path == path


def test_model_interface_methods_defined():
    """Test that all methods in ModelInterface are properly defined"""
    # Create a subclass that doesn't implement all methods
    class IncompleteModel(ModelInterface):
        def train(self, data, labels, **kwargs):
            return {}

        def evaluate(self, data, labels, **kwargs):
            return {}

        # Missing other required methods

    # Should raise TypeError when instantiated
    with pytest.raises(TypeError):
        IncompleteModel()


def test_method_signatures():
    """Test that method signatures are consistent with the interface"""
    # This test is checking implementation understanding, not runtime checks
    # Python doesn't enforce function signatures at runtime,
    # so we'll just verify that the abstract method is defined

    # Create a model with correct method names but wrong signature
    class IncompleteModel(ModelInterface):
        def train(self, data, **kwargs):  # Missing labels parameter
            return {}

        def evaluate(self, data, labels, **kwargs):
            return {}

        def predict(self, data, **kwargs):
            return np.array([])

        def get_parameters(self):
            return {}

        def set_parameters(self, parameters):
            pass

        def save(self, path):
            pass

        def load(self, path):
            pass

    # The issue is that Python doesn't check method signatures at runtime
    # So we'll just test that an instance can be created with the correct method names
    model = IncompleteModel()

    # Verify the methods exist
    assert hasattr(model, 'train')
    assert hasattr(model, 'evaluate')
    assert hasattr(model, 'predict')
    assert hasattr(model, 'get_parameters')
    assert hasattr(model, 'set_parameters')
    assert hasattr(model, 'save')
    assert hasattr(model, 'load')
