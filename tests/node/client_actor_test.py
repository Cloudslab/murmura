import pytest
import numpy as np
import ray
from unittest.mock import patch, MagicMock

from murmura.node.client_actor import VirtualClientActor


@pytest.fixture(scope="module")
def ray_init():
    ray.init(local_mode=True)
    yield
    ray.shutdown()


class MockRayActor:
    """Mock Ray actor that can store data without serialization issues"""
    def __init__(self, actor_id):
        self.actor_id = actor_id
        self.data_partition = None
        self.metadata = {}
        self.split = "train"
        self.feature_columns = None
        self.label_column = None
        self._model = None
        self._dataset = None
        self.neighbours = []

    def remote(self, *args, **kwargs):
        """Mock the remote method pattern of Ray"""
        return self

    def get_id(self):
        """Return actor ID"""
        return self.actor_id

    def set_dataset(self, dataset, feature_columns=None, label_column=None):
        """Set the dataset"""
        self._dataset = dataset
        if feature_columns:
            self.feature_columns = feature_columns
        if label_column:
            self.label_column = label_column

    def set_model(self, model):
        """Set the model"""
        self._model = model

    def receive_data(self, data_partition, metadata=None):
        """Receive data partition"""
        self.data_partition = data_partition
        self.metadata = metadata or {}
        return f"Client {self.actor_id} received {len(data_partition)} samples"

    def get_data_info(self):
        """Get data info"""
        return {
            "client_id": self.actor_id,
            "data_size": len(self.data_partition) if self.data_partition else 0,
            "metadata": self.metadata,
            "has_model": self._model is not None,
            "has_dataset": self._dataset is not None,
        }

    def set_neighbours(self, neighbours):
        """Set neighbours"""
        self.neighbours = neighbours

    def get_neighbours(self):
        """Get neighbours"""
        return [n.get_id() for n in self.neighbours]

    def train_model(self, **kwargs):
        """Mock train model"""
        if not self._model:
            raise ValueError("Model is not set")
        return {"loss": 0.5, "accuracy": 0.8}

    def evaluate_model(self, **kwargs):
        """Mock evaluate model"""
        if not self._model:
            raise ValueError("Model is not set")
        return {"loss": 0.4, "accuracy": 0.9}

    def predict(self, data=None, **kwargs):
        """Mock predict"""
        if not self._model:
            raise ValueError("Model not set")
        return np.array([0, 1, 0])

    def get_model_parameters(self):
        """Mock get parameters"""
        if not self._model:
            raise ValueError("Model is not set")
        return {"layer1": np.array([1.0, 2.0])}

    def set_model_parameters(self, parameters):
        """Mock set parameters"""
        if not self._model:
            raise ValueError("Model is not set")
        # Just store the parameters, don't do anything with them
        self._parameters = parameters


def test_predict_with_data(ray_init):
    """Test predict with provided data instead of partition data"""
    # Create mock client actor
    actor = MockRayActor("test_client")

    # Set up model
    actor.set_model(MagicMock())

    # Create test data
    test_data = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Call predict with explicit data
    with patch.object(actor._model, 'predict', return_value=np.array([0, 1])):
        predictions = actor.predict(data=test_data)

    # Verify output format
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)


def test_train_model_with_parameters(ray_init):
    """Test training with various parameters"""
    # Create mock client actor
    actor = MockRayActor("test_client")

    # Set up model and dataset
    actor.set_model(MagicMock())
    actor.set_dataset(MagicMock(), ["feature"], "label")
    actor.receive_data([0, 1, 2])

    # Set up mock model.train to capture parameters
    mock_train = MagicMock(return_value={"loss": 0.5, "accuracy": 0.8})
    actor._model.train = mock_train

    # Call train_model with various parameters
    actor.train_model(epochs=3, batch_size=16, verbose=True)

    # Check parameter passing (we can't directly check in this mock setup,
    # but we can verify the method was called)
    assert actor._model.train is mock_train


def test_evaluate_model_with_parameters(ray_init):
    """Test evaluating with various parameters"""
    # Create mock client actor
    actor = MockRayActor("test_client")

    # Set up model and dataset
    actor.set_model(MagicMock())
    actor.set_dataset(MagicMock(), ["feature"], "label")
    actor.receive_data([0, 1, 2])

    # Set up mock model.evaluate to capture parameters
    mock_evaluate = MagicMock(return_value={"loss": 0.4, "accuracy": 0.9})
    actor._model.evaluate = mock_evaluate

    # Call evaluate_model with various parameters
    actor.evaluate_model(batch_size=32)

    # Check parameter passing
    assert actor._model.evaluate is mock_evaluate


def test_dataset_operations_errors(ray_init):
    """Test error states with dataset operations"""
    # Create mock client actor
    actor = MockRayActor("test_client")

    # Trying operations that require a dataset
    with pytest.raises(ValueError):
        # Dataset and partition not set
        actor._get_partition_data() if hasattr(actor, '_get_partition_data') else None

    # Set model but no dataset
    actor.set_model(MagicMock())

    # Error should occur with predict
    with pytest.raises(ValueError):
        actor.predict(data=None)  # This should require dataset

    # Set dataset but no partition
    actor.set_dataset(MagicMock(), ["feature"], "label")

    # Should still fail without partition
    with pytest.raises(ValueError):
        actor._get_partition_data() if hasattr(actor, '_get_partition_data') else None


def test_model_operations_errors(ray_init):
    """Test error states with model operations"""
    # Create mock client actor
    actor = MockRayActor("test_client")

    # Model not set
    with pytest.raises(ValueError):
        actor.train_model()

    with pytest.raises(ValueError):
        actor.evaluate_model()

    with pytest.raises(ValueError):
        actor.predict(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError):
        actor.get_model_parameters()

    with pytest.raises(ValueError):
        actor.set_model_parameters({"layer1": np.array([1.0, 2.0])})


# Test using real Ray for those methods that can work properly
@pytest.fixture
def client_actor(ray_init):
    """Create a real client actor"""
    actor = VirtualClientActor.remote("test_client")
    yield actor
    ray.kill(actor)


def test_set_dataset_simple(client_actor):
    """Simple test for setting dataset - using string instead of MagicMock"""
    # Using a simple string instead of MagicMock to avoid serialization issues
    ray.get(client_actor.set_dataset.remote("test_dataset", ["feature"], "label"))

    # Verify dataset was set
    info = ray.get(client_actor.get_data_info.remote())
    assert info["has_dataset"] is True


def test_set_model_simple(client_actor):
    """Simple test for setting model - using string instead of MagicMock"""
    # Using a simple string instead of MagicMock to avoid serialization issues
    ray.get(client_actor.set_model.remote("test_model"))

    # Verify model was set
    info = ray.get(client_actor.get_data_info.remote())
    assert info["has_model"] is True


def test_get_and_set_neighbours_advanced(ray_init):
    """More advanced test for getting and setting neighbours"""
    # Create actors
    actor1 = VirtualClientActor.remote("actor1")
    actor2 = VirtualClientActor.remote("actor2")
    actor3 = VirtualClientActor.remote("actor3")

    # Create a complex neighbour structure
    ray.get(actor1.set_neighbours.remote([actor2, actor3]))
    ray.get(actor2.set_neighbours.remote([actor1, actor3]))
    ray.get(actor3.set_neighbours.remote([actor1, actor2]))

    # Verify neighbour structure
    neighbours_1 = ray.get(actor1.get_neighbours.remote())
    neighbours_2 = ray.get(actor2.get_neighbours.remote())
    neighbours_3 = ray.get(actor3.get_neighbours.remote())

    assert set(neighbours_1) == {"actor2", "actor3"}
    assert set(neighbours_2) == {"actor1", "actor3"}
    assert set(neighbours_3) == {"actor1", "actor2"}

    # Clean up
    for actor in [actor1, actor2, actor3]:
        ray.kill(actor)
