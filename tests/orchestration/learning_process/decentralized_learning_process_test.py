from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from murmura.orchestration.learning_process.decentralized_learning_process import DecentralizedLearningProcess
from murmura.visualization.training_event import (
    ParameterTransferEvent,
    AggregationEvent,
    EvaluationEvent,
)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing"""
    dataset = MagicMock()

    # Mock train split
    train_split = MagicMock()
    train_split.__getitem__.return_value = np.array([[1.0]] * 100)

    # Mock test split
    test_split = MagicMock()
    test_split.__getitem__.return_value = np.array([[1.0]] * 50)

    # Mock get_split method
    dataset.get_split.side_effect = lambda x: train_split if x == "train" else test_split

    # Mock get_partitions method
    dataset.get_partitions.return_value = {0: list(range(50)), 1: list(range(50, 100))}

    return dataset


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = MagicMock()
    model.evaluate.return_value = {"loss": 0.5, "accuracy": 0.8}
    model.get_parameters.return_value = {"layer1": np.array([1.0, 2.0])}
    return model


@pytest.fixture
def mock_topology_manager():
    """Create a mock topology manager for testing"""
    topology_manager = MagicMock()
    topology_manager.config.topology_type.value = "ring"
    topology_manager.adjacency_list = {
        0: [2, 1],
        1: [0, 2],
        2: [1, 0]
    }
    return topology_manager


@pytest.fixture
def mock_cluster_manager(mock_topology_manager):
    """Create a mock cluster manager for testing"""
    cluster_manager = MagicMock()
    cluster_manager.topology_manager = mock_topology_manager

    # Mock actors
    actor1 = MagicMock()
    actor1.get_model_parameters.return_value = {"layer1": np.array([1.0, 2.0])}

    actor2 = MagicMock()
    actor2.get_model_parameters.return_value = {"layer1": np.array([3.0, 4.0])}

    actor3 = MagicMock()
    actor3.get_model_parameters.return_value = {"layer1": np.array([5.0, 6.0])}

    cluster_manager.actors = [actor1, actor2, actor3]

    # Mock aggregation strategy
    strategy = MagicMock()
    strategy.__class__.__name__ = "GossipAvg"
    cluster_manager.aggregation_strategy = strategy

    # Mock train_models method
    cluster_manager.train_models.return_value = [
        {"loss": 0.5, "accuracy": 0.7},
        {"loss": 0.6, "accuracy": 0.8},
        {"loss": 0.4, "accuracy": 0.9}
    ]

    # Mock aggregate_model_parameters method
    cluster_manager.aggregate_model_parameters.return_value = {"layer1": np.array([2.0, 3.0])}

    # Mock get_topology_information method
    cluster_manager.get_topology_information.return_value = {
        "initialized": True,
        "type": "ring",
        "num_actors": 3,
        "adjacency_list": {
            0: [2, 1],
            1: [0, 2],
            2: [1, 0]
        }
    }

    return cluster_manager


@pytest.fixture
def decentralized_learning_process(mock_dataset, mock_model, mock_cluster_manager):
    """Create a decentralized learning process for testing"""
    config = {
        "rounds": 2,
        "epochs": 1,
        "batch_size": 32,
        "test_split": "test",
        "feature_columns": ["image"],
        "label_column": "label",
        "split": "train"
    }

    process = DecentralizedLearningProcess(config, mock_dataset, mock_model)
    process.cluster_manager = mock_cluster_manager

    # Mock training_monitor to track emitted events
    process.training_monitor = MagicMock()

    return process


@pytest.fixture(autouse=True)
def ray_patch():
    """Patch ray.get to handle MagicMock objects"""
    with patch('ray.get') as mock_ray_get:
        # Configure mock to return a fixed parameter value for any input
        mock_ray_get.return_value = {"layer1": np.array([1.0, 2.0])}
        yield mock_ray_get


def test_execute_without_initialization_raises_error():
    """Test that execute raises an error if not initialized"""
    process = DecentralizedLearningProcess({}, MagicMock(), MagicMock())

    with pytest.raises(ValueError, match="Learning process not initialized"):
        process.execute()


def test_execute_evaluation_setup(decentralized_learning_process, mock_dataset, mock_model):
    """Test the initial evaluation setup in execute method"""
    # Execute the learning process
    decentralized_learning_process.execute()

    # Verify test data was properly prepared
    mock_dataset.get_split.assert_any_call("test")

    # Verify initial model evaluation
    mock_model.evaluate.assert_called()

    # Verify evaluation event was emitted
    # Get all emitted events
    emitted_events = [call.args[0] for call in decentralized_learning_process.training_monitor.emit_event.call_args_list]
    # Find EvaluationEvent with round_num=0
    eval_events = [e for e in emitted_events if isinstance(e, EvaluationEvent) and e.round_num == 0]
    assert len(eval_events) > 0, "No initial evaluation event found"
    assert set(eval_events[0].metrics.keys()) == {"loss", "accuracy"}


def test_execute_topology_information(decentralized_learning_process, mock_cluster_manager):
    """Test that topology information is retrieved"""
    # Execute the learning process
    decentralized_learning_process.execute()

    # Verify topology information was requested
    mock_cluster_manager.get_topology_information.assert_called_once()


def test_execute_training_rounds(decentralized_learning_process, mock_cluster_manager):
    """Test the training rounds in execute method"""
    # Execute the learning process
    decentralized_learning_process.execute()

    # Verify training was performed for each round
    assert mock_cluster_manager.train_models.call_count == 2

    # Verify each training call had the correct parameters
    expected_call = call(epochs=1, batch_size=32, verbose=False)
    mock_cluster_manager.train_models.assert_has_calls([expected_call, expected_call])


def test_execute_parameter_transfer_events(decentralized_learning_process):
    """Test that parameter transfer events are emitted based on topology"""
    # Execute the learning process
    decentralized_learning_process.execute()

    # Get all parameter transfer events
    event_calls = decentralized_learning_process.training_monitor.emit_event.call_args_list
    transfer_events = [
        call.args[0] for call in event_calls
        if isinstance(call.args[0], ParameterTransferEvent)
    ]

    # In a ring topology, each node should have 2 neighbors
    # So for 3 nodes, we should have at least 3 parameter transfer events per round
    assert len(transfer_events) >= 6  # At least 3 events × 2 rounds

    # Check the structure of the first event
    event = transfer_events[0]
    assert hasattr(event, 'source_nodes')
    assert hasattr(event, 'target_nodes')
    assert hasattr(event, 'param_summary')


def test_execute_aggregation_events(decentralized_learning_process):
    """Test that aggregation events are emitted for each node"""
    # Execute the learning process
    decentralized_learning_process.execute()

    # Get all aggregation events
    event_calls = decentralized_learning_process.training_monitor.emit_event.call_args_list
    aggregation_events = [
        call.args[0] for call in event_calls
        if isinstance(call.args[0], AggregationEvent)
    ]

    # In a decentralized setting, each node should perform local aggregation
    # So for 3 nodes and 2 rounds, we should have at least 6 aggregation events
    assert len(aggregation_events) >= 6  # At least 3 nodes × 2 rounds

    # Check structure of an event
    event = aggregation_events[0]
    assert hasattr(event, 'participating_nodes')
    assert hasattr(event, 'aggregator_node')
    assert hasattr(event, 'strategy_name')
    assert event.strategy_name == "GossipAvg"


def test_execute_weighted_aggregation(decentralized_learning_process, mock_cluster_manager):
    """Test that weighted aggregation is used based on data size"""
    # Execute the learning process
    decentralized_learning_process.execute()

    # Verify aggregate_model_parameters was called with weights
    mock_cluster_manager.aggregate_model_parameters.assert_called()

    # Check that weights parameter was provided
    _, kwargs = mock_cluster_manager.aggregate_model_parameters.call_args
    assert 'weights' in kwargs
    weights = kwargs['weights']

    # Should have weights (exact number depends on mock_dataset.get_partitions)
    assert len(weights) >= 1
    # All weights should be positive floats
    assert all(isinstance(w, float) and w > 0 for w in weights)


def test_execute_model_update(decentralized_learning_process, mock_cluster_manager, mock_model):
    """Test the model update process in execute method"""
    # Set the expected return value for aggregate_model_parameters
    mock_cluster_manager.aggregate_model_parameters.return_value = {"layer1": np.array([2.0, 3.0])}

    # Execute the learning process
    decentralized_learning_process.execute()

    # Verify global model was updated
    assert mock_model.set_parameters.call_count >= 1

    # Get the actual parameters passed to set_parameters
    call_args = mock_model.set_parameters.call_args_list[-1][0][0]  # Last call, first positional arg

    # Verify the parameters
    assert "layer1" in call_args
    assert np.allclose(call_args["layer1"], np.array([2.0, 3.0]))


def test_execute_evaluation_after_rounds(decentralized_learning_process, mock_model):
    """Test that evaluation is performed after each round"""
    # Execute the learning process
    decentralized_learning_process.execute()

    # The exact number of evaluate calls depends on implementation details
    # Initial evaluation + at least one per round
    assert mock_model.evaluate.call_count >= 3

    # Check all calls are properly structure with the right arguments
    for call_args in mock_model.evaluate.call_args_list:
        # Should be called with at least test data
        assert len(call_args[0]) >= 1


def test_execute_return_value(decentralized_learning_process, mock_model):
    """Test the return value from execute method"""
    # We need enough side effects for all evaluate calls (appears to be 4 from previous test)
    mock_model.evaluate.side_effect = [
        {"loss": 0.8, "accuracy": 0.7},  # Initial
        {"loss": 0.5, "accuracy": 0.8},  # Round 1
        {"loss": 0.5, "accuracy": 0.8},  # Round 2
        {"loss": 0.3, "accuracy": 0.9},  # Final
        {"loss": 0.3, "accuracy": 0.9}   # Extra in case we need it
    ]

    # Execute the learning process
    result = decentralized_learning_process.execute()

    # Verify result structure
    assert "initial_metrics" in result
    assert "final_metrics" in result
    assert "accuracy_improvement" in result
    assert "round_metrics" in result
    assert "topology" in result

    # Verify values
    assert result["initial_metrics"]["accuracy"] == 0.7
    assert result["final_metrics"]["accuracy"] == 0.9
    assert pytest.approx(result["accuracy_improvement"]) == 0.2  # 0.9 - 0.7, using approx for floating point
    assert len(result["round_metrics"]) == 2
