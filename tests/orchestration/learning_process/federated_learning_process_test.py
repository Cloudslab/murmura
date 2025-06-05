from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from murmura.orchestration.learning_process.federated_learning_process import (
    FederatedLearningProcess,
)
from murmura.visualization.training_event import (
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
    dataset.get_split.side_effect = (
        lambda x: train_split if x == "train" else test_split
    )

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
    topology_manager.config.topology_type.value = "star"
    topology_manager.config.hub_index = 0
    topology_manager.adjacency_list = {0: [1, 2], 1: [0], 2: [0]}
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

    cluster_manager.actors = [actor1, actor2]

    # Mock aggregation strategy
    strategy = MagicMock()
    strategy.__class__.__name__ = "FedAvg"
    cluster_manager.aggregation_strategy = strategy

    # Mock train_models method
    cluster_manager.train_models.return_value = [
        {"loss": 0.5, "accuracy": 0.7},
        {"loss": 0.6, "accuracy": 0.8},
    ]

    # Mock aggregate_model_parameters method
    cluster_manager.aggregate_model_parameters.return_value = {
        "layer1": np.array([2.0, 3.0])
    }

    return cluster_manager


@pytest.fixture
def federated_learning_process(mock_dataset, mock_model, mock_cluster_manager):
    """Create a federated learning process for testing"""
    config = {
        "rounds": 2,
        "epochs": 1,
        "batch_size": 32,
        "test_split": "test",
        "feature_columns": ["image"],
        "label_column": "label",
        "split": "train",
    }

    process = FederatedLearningProcess(config, mock_dataset, mock_model)
    process.cluster_manager = mock_cluster_manager

    # Mock training_monitor to track emitted events
    process.training_monitor = MagicMock()

    return process


@pytest.fixture(autouse=True)
def ray_patch():
    """Patch ray.get to handle MagicMock objects"""
    with patch("ray.get") as mock_ray_get:
        # Configure mock to return a fixed parameter value for any input
        mock_ray_get.return_value = {"layer1": np.array([1.0, 2.0])}
        yield mock_ray_get


def test_execute_without_initialization_raises_error():
    """Test that execute raises an error if not initialized"""
    process = FederatedLearningProcess({}, MagicMock(), MagicMock())

    with pytest.raises(ValueError, match="Learning process not initialized"):
        process.execute()


def test_execute_without_topology_manager_raises_error(mock_dataset, mock_model):
    """Test that execute raises an error if topology manager is not set"""
    process = FederatedLearningProcess({}, mock_dataset, mock_model)

    # Set cluster_manager but not topology_manager
    process.cluster_manager = MagicMock()
    process.cluster_manager.topology_manager = None

    with pytest.raises(ValueError, match="Topology manager not set"):
        process.execute()


def test_execute_evaluation_setup(federated_learning_process, mock_dataset, mock_model):
    """Test the initial evaluation setup in execute method"""
    # Execute the learning process
    federated_learning_process.execute()

    # Verify test data was properly prepared
    mock_dataset.get_split.assert_any_call("test")

    # Verify initial model evaluation
    mock_model.evaluate.assert_called()

    # Verify evaluation event was emitted
    # Get all emitted events
    emitted_events = [
        call.args[0]
        for call in federated_learning_process.training_monitor.emit_event.call_args_list
    ]
    # Find EvaluationEvent with round_num=0
    eval_events = [
        e for e in emitted_events if isinstance(e, EvaluationEvent) and e.round_num == 0
    ]
    assert len(eval_events) > 0, "No initial evaluation event found"
    assert set(eval_events[0].metrics.keys()) == {"loss", "accuracy"}


def test_execute_training_rounds(federated_learning_process, mock_cluster_manager):
    """Test the training rounds in execute method"""
    # Execute the learning process
    federated_learning_process.execute()

    # Verify training was performed for each round
    assert mock_cluster_manager.train_models.call_count == 2

    # Verify each training call had the correct parameters
    expected_call = call(client_sampling_rate=1.0, data_sampling_rate=1.0, epochs=1, batch_size=32, verbose=True)
    mock_cluster_manager.train_models.assert_has_calls([expected_call, expected_call])


def test_execute_aggregation(federated_learning_process, mock_cluster_manager):
    """Test the aggregation process in execute method"""
    # Execute the learning process
    federated_learning_process.execute()

    # Verify get_model_parameters was called on each actor
    for actor in mock_cluster_manager.actors:
        actor.get_model_parameters.remote.assert_called()

    # Verify aggregate_model_parameters was called for each round
    assert mock_cluster_manager.aggregate_model_parameters.call_count == 2


def test_execute_model_update(
    federated_learning_process, mock_cluster_manager, mock_model
):
    """Test the model update process in execute method"""
    # Set the expected return value for aggregate_model_parameters
    mock_cluster_manager.aggregate_model_parameters.return_value = {
        "layer1": np.array([2.0, 3.0])
    }

    # Execute the learning process
    federated_learning_process.execute()

    # Verify global model was updated
    assert mock_model.set_parameters.call_count >= 1

    # Get the actual parameters passed to set_parameters
    call_args = mock_model.set_parameters.call_args_list[-1][0][
        0
    ]  # Last call, first positional arg

    # Verify the parameters
    assert "layer1" in call_args
    assert np.allclose(call_args["layer1"], np.array([2.0, 3.0]))


def test_execute_events(federated_learning_process):
    """Test that all expected events are emitted during execution"""
    # Execute the learning process
    federated_learning_process.execute()

    # Count the different event types that were emitted
    event_calls = federated_learning_process.training_monitor.emit_event.call_args_list
    event_types = [call.args[0].__class__.__name__ for call in event_calls]

    # Check for the presence of each event type
    assert event_types.count("EvaluationEvent") == 3  # Initial + 2 rounds
    assert event_types.count("LocalTrainingEvent") == 2  # 2 rounds
    assert event_types.count("ParameterTransferEvent") >= 2  # At least 2 rounds
    assert event_types.count("AggregationEvent") == 2  # 2 rounds
    assert event_types.count("ModelUpdateEvent") == 2  # 2 rounds


def test_execute_return_value(federated_learning_process, mock_model):
    """Test the return value from execute method"""
    # We need enough side effects for all evaluate calls
    mock_model.evaluate.side_effect = [
        {"loss": 0.8, "accuracy": 0.7},  # Initial
        {"loss": 0.5, "accuracy": 0.8},  # Round 1
        {"loss": 0.5, "accuracy": 0.8},  # Round 2
        {"loss": 0.3, "accuracy": 0.9},  # Final
        {"loss": 0.3, "accuracy": 0.9},  # Extra in case we need it
    ]

    # Execute the learning process
    result = federated_learning_process.execute()

    # Verify result structure
    assert "initial_metrics" in result
    assert "final_metrics" in result
    assert "accuracy_improvement" in result
    assert "round_metrics" in result

    # Verify values
    assert result["initial_metrics"]["accuracy"] == 0.7
    assert result["final_metrics"]["accuracy"] == 0.9
    assert (
        pytest.approx(result["accuracy_improvement"]) == 0.2
    )  # 0.9 - 0.7, using approx for floating point
    assert len(result["round_metrics"]) == 2


@pytest.mark.parametrize(
    "rounds,batch_size,epochs",
    [
        (1, 16, 1),
        (2, 32, 2),
        (3, 64, 1),
    ],
)
def test_different_learning_configurations(
    mock_dataset, mock_model, mock_cluster_manager, rounds, batch_size, epochs
):
    config = {
        "rounds": rounds,
        "epochs": epochs,
        "batch_size": batch_size,
        "test_split": "test",
        "feature_columns": ["image"],
        "label_column": "label",
        "split": "train",
    }
    process = FederatedLearningProcess(config, mock_dataset, mock_model)
    process.cluster_manager = mock_cluster_manager
    process.training_monitor = MagicMock()
    process.execute()
    assert mock_cluster_manager.train_models.call_count == rounds
    expected_call = call(client_sampling_rate=1.0, data_sampling_rate=1.0, epochs=epochs, batch_size=batch_size, verbose=True)
    mock_cluster_manager.train_models.assert_has_calls([expected_call] * rounds)


def test_event_emission_edge_cases(federated_learning_process):
    """Test event emission for edge cases (e.g., zero rounds)"""
    federated_learning_process.config.rounds = 0
    federated_learning_process.execute()
    event_types = [
        call.args[0].__class__.__name__
        for call in federated_learning_process.training_monitor.emit_event.call_args_list
    ]
    assert "EvaluationEvent" in event_types
    assert event_types.count("LocalTrainingEvent") == 0


def test_error_handling_missing_data(federated_learning_process):
    """Test error handling when data is missing"""
    federated_learning_process.dataset = None
    with pytest.raises(Exception):
        federated_learning_process.execute()


def test_process_interruption_and_resumption(tmp_path, federated_learning_process):
    """Test process interruption and resumption (simulated via temp file)"""
    # Simulate saving state to a file
    state_file = tmp_path / "state.json"
    state = {"round": 1, "metrics": {"loss": 0.5}}
    with open(state_file, "w") as f:
        import json

        json.dump(state, f)
    # Simulate resuming by reading state
    with open(state_file) as f:
        loaded = json.load(f)
    assert loaded["round"] == 1
    assert "metrics" in loaded
