import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from murmura.orchestration.learning_process.learning_process import LearningProcess
from murmura.aggregation.aggregation_config import AggregationConfig
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.visualization.training_observer import TrainingObserver


class MockObserver(TrainingObserver):
    """Mock observer for testing"""
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)

    def set_topology(self, topology_manager):
        self.topology_manager = topology_manager


class ConcreteLearningProcess(LearningProcess):
    """Concrete implementation of LearningProcess for testing"""
    def execute(self):
        return {"status": "executed"}


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing"""
    dataset = MagicMock()
    dataset.get_partitions.return_value = {0: [1, 2, 3], 1: [4, 5, 6]}
    return dataset


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = MagicMock()
    model.get_parameters.return_value = {"layer1": np.array([1.0, 2.0])}
    return model


@pytest.fixture
def learning_process(mock_dataset, mock_model):
    """Create a concrete learning process instance for testing"""
    config = {"key": "value"}
    return ConcreteLearningProcess(config, mock_dataset, mock_model)


def test_initialization(learning_process, mock_dataset, mock_model):
    """Test proper initialization of learning process"""
    assert learning_process.config == {"key": "value"}
    assert learning_process.dataset == mock_dataset
    assert learning_process.model == mock_model
    assert learning_process.cluster_manager is None
    assert hasattr(learning_process, "training_monitor")


def test_register_observer(learning_process):
    """Test registering observers with the training monitor"""
    observer = MockObserver()
    learning_process.register_observer(observer)

    assert observer in learning_process.training_monitor.observers


@patch("murmura.orchestration.learning_process.learning_process.ClusterManager")
def test_initialize(mock_cluster_manager_class, learning_process, mock_dataset):
    """Test initializing the learning process"""
    # Setup mocks
    mock_cluster_manager = MagicMock()
    mock_cluster_manager_class.return_value = mock_cluster_manager

    mock_topology_config = TopologyConfig(topology_type=TopologyType.STAR)
    mock_aggregation_config = AggregationConfig()
    mock_partitioner = MagicMock()
    mock_partitioner.partition.return_value = None

    # Call initialize
    learning_process.initialize(
        num_actors=3,
        topology_config=mock_topology_config,
        aggregation_config=mock_aggregation_config,
        partitioner=mock_partitioner
    )

    # Verify ClusterManager was created and configured
    mock_cluster_manager_class.assert_called_once_with({"key": "value"})

    # Verify aggregation strategy was set
    mock_cluster_manager.set_aggregation_strategy.assert_called_once_with(mock_aggregation_config)

    # Verify actors were created
    mock_cluster_manager.create_actors.assert_called_once_with(3, mock_topology_config)

    # Verify dataset was partitioned
    mock_partitioner.partition.assert_called_once()

    # Verify data was distributed
    mock_cluster_manager.distribute_data.assert_called_once()

    # Verify dataset was distributed
    mock_cluster_manager.distribute_dataset.assert_called_once()

    # Verify model was distributed
    mock_cluster_manager.distribute_model.assert_called_once()


def test_execute(learning_process):
    """Test execute method on concrete implementation"""
    result = learning_process.execute()
    assert result == {"status": "executed"}


def test_shutdown(learning_process):
    """Test shutdown method with and without cluster manager"""
    # Without cluster manager
    learning_process.shutdown()

    # With cluster manager
    mock_cluster_manager = MagicMock()
    learning_process.cluster_manager = mock_cluster_manager
    learning_process.shutdown()

    mock_cluster_manager.shutdown.assert_called_once()


def test_parameter_convergence_calculation():
    """Test the _calculate_parameter_convergence static method"""
    node_params = {
        0: {"layer": np.array([1.0, 2.0, 3.0])},
        1: {"layer": np.array([1.1, 2.1, 3.1])},
        2: {"layer": np.array([0.9, 1.9, 2.9])}
    }

    global_params = {"layer": np.array([1.0, 2.0, 3.0])}

    convergence = LearningProcess._calculate_parameter_convergence(node_params, global_params)

    # Convergence should be the average L2 norm of parameter differences
    assert isinstance(convergence, float)
    assert convergence > 0


def test_parameter_summaries():
    """Test the _create_parameter_summaries static method"""
    node_params = {
        0: {"layer": np.array([1.0, 2.0, 3.0])},
        1: {"layer": np.array([4.0, 5.0, 6.0])}
    }

    summaries = LearningProcess._create_parameter_summaries(node_params)

    assert 0 in summaries
    assert 1 in summaries

    # Check summary structure
    for node_id in [0, 1]:
        assert "norm" in summaries[node_id]
        assert "mean" in summaries[node_id]
        assert "std" in summaries[node_id]

        assert isinstance(summaries[node_id]["norm"], float)
        assert isinstance(summaries[node_id]["mean"], float)
        assert isinstance(summaries[node_id]["std"], float)


def test_parameter_summaries_empty():
    """Test parameter summaries with empty parameters"""
    node_params = {
        0: {},
        1: {"layer": np.array([1.0, 2.0])}
    }

    summaries = LearningProcess._create_parameter_summaries(node_params)

    # Node 0 should be skipped since it has no parameters
    assert 0 not in summaries
    assert 1 in summaries
