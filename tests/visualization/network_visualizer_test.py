import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from murmura.visualization.network_visualizer import NetworkVisualizer
from murmura.visualization.training_event import (
    LocalTrainingEvent,
    ParameterTransferEvent,
    AggregationEvent,
    ModelUpdateEvent,
    EvaluationEvent,
    InitialStateEvent,
)


@pytest.fixture
def network_visualizer():
    """Create a NetworkVisualizer with a temporary output directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = NetworkVisualizer(output_dir=temp_dir)
        yield visualizer


@pytest.fixture
def mock_topology_manager():
    """Create a mock topology manager for testing"""
    topology_manager = MagicMock()
    topology_manager.adjacency_list = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2]
    }
    topology_manager.config.topology_type.value = "star"
    return topology_manager


def test_initialization(network_visualizer):
    """Test initialization of NetworkVisualizer"""
    assert network_visualizer.topology is None
    assert network_visualizer.network_type is None
    assert network_visualizer.frames == []
    assert network_visualizer.parameter_history == {}
    assert network_visualizer.accuracy_history == []
    assert network_visualizer.loss_history == []
    assert network_visualizer.frame_descriptions == []
    assert network_visualizer.round_metrics == {}
    assert os.path.isdir(network_visualizer.output_dir)


def test_set_topology(network_visualizer, mock_topology_manager):
    """Test setting the topology"""
    network_visualizer.set_topology(mock_topology_manager)

    assert network_visualizer.topology == mock_topology_manager.adjacency_list
    assert network_visualizer.network_type == mock_topology_manager.config.topology_type.value


def test_on_event_without_topology(network_visualizer):
    """Test on_event behavior without topology (should ignore non-initial events)"""
    # Non-initial event should be ignored
    event = LocalTrainingEvent(round_num=1, active_nodes=[0, 1], metrics={})
    network_visualizer.on_event(event)

    # No frames should be created
    assert len(network_visualizer.frames) == 0

    # Initial event should still be processed
    event = InitialStateEvent(topology_type="ring", num_nodes=5)
    network_visualizer.on_event(event)

    # Frame should be created
    assert len(network_visualizer.frames) == 1
    assert network_visualizer.network_type == "ring"


def test_on_event_initial_state(network_visualizer):
    """Test processing of InitialStateEvent"""
    event = InitialStateEvent(topology_type="star", num_nodes=4)
    network_visualizer.on_event(event)

    assert len(network_visualizer.frames) == 1
    frame = network_visualizer.frames[0]

    assert frame["round"] == 0
    assert frame["step"] == "initial_state"
    assert frame["topology_type"] == "star"
    assert frame["num_nodes"] == 4

    assert network_visualizer.frame_descriptions[0] == "Initial network state"


def test_on_event_local_training(network_visualizer, mock_topology_manager):
    """Test processing of LocalTrainingEvent"""
    network_visualizer.set_topology(mock_topology_manager)

    event = LocalTrainingEvent(
        round_num=1,
        active_nodes=[0, 1, 2],
        metrics={"loss": 0.5, "accuracy": 0.8}
    )
    network_visualizer.on_event(event)

    assert len(network_visualizer.frames) == 1
    frame = network_visualizer.frames[0]

    assert frame["round"] == 1
    assert frame["step"] == "local_training"
    assert frame["active_nodes"] == [0, 1, 2]
    assert frame["metrics"] == {"loss": 0.5, "accuracy": 0.8}

    assert "Local training" in network_visualizer.frame_descriptions[0]


def test_on_event_parameter_transfer(network_visualizer, mock_topology_manager):
    """Test processing of ParameterTransferEvent"""
    network_visualizer.set_topology(mock_topology_manager)

    param_summary = {
        0: {"norm": 1.0, "mean": 0.5, "std": 0.1},
        1: {"norm": 2.0, "mean": 0.7, "std": 0.2}
    }

    event = ParameterTransferEvent(
        round_num=1,
        source_nodes=[0],
        target_nodes=[1, 2],
        param_summary=param_summary
    )
    network_visualizer.on_event(event)

    assert len(network_visualizer.frames) == 1
    frame = network_visualizer.frames[0]

    assert frame["round"] == 1
    assert frame["step"] == "parameter_transfer"
    assert frame["active_edges"] == [(0, 1), (0, 2)]
    assert frame["node_params"] == param_summary

    # Check parameter_history was updated
    assert 0 in network_visualizer.parameter_history
    assert 1 in network_visualizer.parameter_history
    assert network_visualizer.parameter_history[0] == [1.0]
    assert network_visualizer.parameter_history[1] == [2.0]

    assert "Parameter transfer" in network_visualizer.frame_descriptions[0]


def test_on_event_aggregation(network_visualizer, mock_topology_manager):
    """Test processing of AggregationEvent"""
    network_visualizer.set_topology(mock_topology_manager)

    event = AggregationEvent(
        round_num=1,
        participating_nodes=[0, 1, 2],
        aggregator_node=0,
        strategy_name="FedAvg"
    )
    network_visualizer.on_event(event)

    assert len(network_visualizer.frames) == 1
    frame = network_visualizer.frames[0]

    assert frame["round"] == 1
    assert frame["step"] == "aggregation"
    assert frame["active_nodes"] == [0, 1, 2]
    assert frame["strategy_name"] == "FedAvg"
    assert (1, 0) in frame["active_edges"]
    assert (2, 0) in frame["active_edges"]

    assert "Aggregation at node 0" in network_visualizer.frame_descriptions[0]


def test_on_event_model_update(network_visualizer, mock_topology_manager):
    """Test processing of ModelUpdateEvent"""
    network_visualizer.set_topology(mock_topology_manager)

    event = ModelUpdateEvent(
        round_num=1,
        updated_nodes=[0, 1, 2, 3],
        param_convergence=0.01
    )
    network_visualizer.on_event(event)

    assert len(network_visualizer.frames) == 1
    frame = network_visualizer.frames[0]

    assert frame["round"] == 1
    assert frame["step"] == "model_update"
    assert frame["active_nodes"] == [0, 1, 2, 3]
    assert frame["param_convergence"] == 0.01

    assert "Model update" in network_visualizer.frame_descriptions[0]


def test_on_event_evaluation(network_visualizer, mock_topology_manager):
    """Test processing of EvaluationEvent"""
    network_visualizer.set_topology(mock_topology_manager)

    event = EvaluationEvent(
        round_num=1,
        metrics={"loss": 0.3, "accuracy": 0.95}
    )
    network_visualizer.on_event(event)

    assert len(network_visualizer.frames) == 1
    frame = network_visualizer.frames[0]

    assert frame["round"] == 1
    assert frame["step"] == "evaluation"
    assert frame["metrics"] == {"loss": 0.3, "accuracy": 0.95}

    # Check metrics tracking
    assert network_visualizer.round_metrics[1] == {"loss": 0.3, "accuracy": 0.95}
    assert network_visualizer.accuracy_history == [0.95]
    assert network_visualizer.loss_history == [0.3]

    assert "Evaluation" in network_visualizer.frame_descriptions[0]
    assert "0.95" in network_visualizer.frame_descriptions[0]


@patch("matplotlib.animation.FuncAnimation")
@patch("matplotlib.animation.FFMpegWriter")
def test_render_training_animation(mock_ffmpeg_writer, mock_func_animation, network_visualizer, mock_topology_manager):
    """Test render_training_animation method (mocked to avoid actual rendering)"""
    network_visualizer.set_topology(mock_topology_manager)

    # Add some frames
    network_visualizer.on_event(InitialStateEvent(topology_type="star", num_nodes=4))
    network_visualizer.on_event(EvaluationEvent(round_num=1, metrics={"loss": 0.5, "accuracy": 0.8}))

    # Mock animation.save
    mock_animation = MagicMock()
    mock_func_animation.return_value = mock_animation

    # Call the method
    network_visualizer.render_training_animation(filename="test_animation.mp4", fps=5)

    # Check that animation was created with correct parameters
    mock_func_animation.assert_called_once()

    # Check that FFMpegWriter was created with the right FPS
    mock_ffmpeg_writer.assert_called_once_with(
        fps=5, metadata=dict(artist="Murmura Framework"), bitrate=5000
    )

    # Check that animation.save was called
    mock_animation.save.assert_called_once()


@patch("matplotlib.pyplot.savefig")
def test_render_frame_sequence(mock_savefig, network_visualizer, mock_topology_manager):
    """Test render_frame_sequence method (mocked to avoid actual rendering)"""
    network_visualizer.set_topology(mock_topology_manager)

    # Add some frames
    network_visualizer.on_event(InitialStateEvent(topology_type="star", num_nodes=4))
    network_visualizer.on_event(EvaluationEvent(round_num=1, metrics={"loss": 0.5, "accuracy": 0.8}))

    # Call the method
    network_visualizer.render_frame_sequence(prefix="test_frame")

    # Check that savefig was called for each frame
    assert mock_savefig.call_count == 2


@patch("matplotlib.pyplot.savefig")
def test_render_summary_plot(mock_savefig, network_visualizer, mock_topology_manager):
    """Test render_summary_plot method (mocked to avoid actual rendering)"""
    network_visualizer.set_topology(mock_topology_manager)

    # Add parameter history
    network_visualizer.parameter_history = {
        0: [1.0, 1.1, 1.2],
        1: [2.0, 1.9, 1.8]
    }

    # Add metrics history
    network_visualizer.accuracy_history = [0.7, 0.8, 0.9]
    network_visualizer.loss_history = [0.5, 0.4, 0.3]

    # Add some frames (needed for render_summary_plot to work)
    network_visualizer.frames = [
        {
            "round": 1,
            "step": "training",
            "timestamp": 12345,
            "topology_type": "star",
            "active_nodes": [0, 1],
            "metrics": {"loss": 0.5, "accuracy": 0.7}
        },
        {
            "round": 2,
            "step": "aggregation",
            "timestamp": 12346,
            "topology_type": "star",
            "active_nodes": [0, 1],
            "metrics": {"loss": 0.4, "accuracy": 0.8}
        }
    ]

    # Call the method
    network_visualizer.render_summary_plot(filename="test_summary.png")

    # Check that savefig was called
    mock_savefig.assert_called_once()


def test_render_without_frames(network_visualizer):
    """Test rendering methods with no frames"""
    # All should print a message and return without error
    network_visualizer.render_training_animation()
    network_visualizer.render_frame_sequence()
    network_visualizer.render_summary_plot()
