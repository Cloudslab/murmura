import os
import tempfile
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import matplotlib.pyplot as plt

from murmura.visualization.network_visualizer import NetworkVisualizer
from murmura.visualization.training_event import (
    InitialStateEvent,
    LocalTrainingEvent,
    ParameterTransferEvent,
    AggregationEvent,
    ModelUpdateEvent,
    EvaluationEvent,
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


@pytest.fixture
def populated_visualizer(network_visualizer, mock_topology_manager):
    """Create a visualizer with populated frames and data"""
    network_visualizer.set_topology(mock_topology_manager)

    # Add initial state
    network_visualizer.on_event(InitialStateEvent(topology_type="star", num_nodes=4))

    # Add parameter transfer with parameter history
    network_visualizer.on_event(ParameterTransferEvent(
        round_num=1,
        source_nodes=[0],
        target_nodes=[1, 2],
        param_summary={
            0: {"norm": 1.0, "mean": 0.5, "std": 0.1},
            1: {"norm": 2.0, "mean": 0.7, "std": 0.2}
        }
    ))

    # Add evaluation events with metrics
    network_visualizer.on_event(EvaluationEvent(
        round_num=1,
        metrics={"loss": 0.5, "accuracy": 0.8}
    ))

    network_visualizer.on_event(EvaluationEvent(
        round_num=2,
        metrics={"loss": 0.3, "accuracy": 0.9}
    ))

    return network_visualizer


def test_parameter_tracking_across_rounds(network_visualizer, mock_topology_manager):
    """Test tracking of parameter changes across multiple rounds"""
    network_visualizer.set_topology(mock_topology_manager)

    # First round parameter transfer
    network_visualizer.on_event(ParameterTransferEvent(
        round_num=1,
        source_nodes=[0],
        target_nodes=[1, 2],
        param_summary={
            0: {"norm": 1.0, "mean": 0.5, "std": 0.1},
            1: {"norm": 2.0, "mean": 0.7, "std": 0.2}
        }
    ))

    # Second round parameter transfer with different values
    network_visualizer.on_event(ParameterTransferEvent(
        round_num=2,
        source_nodes=[0],
        target_nodes=[1, 2],
        param_summary={
            0: {"norm": 1.2, "mean": 0.6, "std": 0.15},
            1: {"norm": 1.8, "mean": 0.65, "std": 0.18}
        }
    ))

    # Check parameter history is tracked correctly
    assert network_visualizer.parameter_history[0] == [1.0, 1.2]
    assert network_visualizer.parameter_history[1] == [2.0, 1.8]

    # Check parameter update frames are recorded
    assert 0 in network_visualizer.parameter_update_frames
    assert 1 in network_visualizer.parameter_update_frames

    # Each node should have two parameter updates (one for each round)
    assert len(network_visualizer.parameter_update_frames[0]) == 2
    assert len(network_visualizer.parameter_update_frames[1]) == 2


def test_raw_parameter_summary_handling(network_visualizer, mock_topology_manager):
    """Test handling of raw parameter summaries (without norm/mean/std)"""
    network_visualizer.set_topology(mock_topology_manager)

    # Provide raw parameters instead of summaries
    network_visualizer.on_event(ParameterTransferEvent(
        round_num=1,
        source_nodes=[0],
        target_nodes=[1],
        param_summary={
            0: {"raw_param": np.array([1.0, 2.0])}
        }
    ))

    # Should still have created an entry in parameter_history
    assert 0 in network_visualizer.parameter_history
    assert len(network_visualizer.parameter_history[0]) == 1
    # The value should be a placeholder (1.0)
    assert network_visualizer.parameter_history[0][0] == 1.0


def test_multiple_metrics_tracking(network_visualizer, mock_topology_manager):
    """Test tracking of multiple metrics across rounds"""
    network_visualizer.set_topology(mock_topology_manager)

    # First round evaluation
    network_visualizer.on_event(EvaluationEvent(
        round_num=1,
        metrics={"loss": 0.5, "accuracy": 0.8, "f1_score": 0.75}
    ))

    # Second round evaluation with different values
    network_visualizer.on_event(EvaluationEvent(
        round_num=2,
        metrics={"loss": 0.3, "accuracy": 0.9, "f1_score": 0.85}
    ))

    # Check metrics history is tracked correctly
    assert network_visualizer.accuracy_history == {1: 0.8, 2: 0.9}
    assert network_visualizer.loss_history == {1: 0.5, 2: 0.3}

    # Check round_metrics contains all metrics
    assert network_visualizer.round_metrics[1]["f1_score"] == 0.75
    assert network_visualizer.round_metrics[2]["f1_score"] == 0.85

    # Check frames contain the metrics
    latest_frame = network_visualizer.frames[-1]
    assert latest_frame["all_metrics"][1]["f1_score"] == 0.75
    assert latest_frame["all_metrics"][2]["f1_score"] == 0.85


def test_skipped_round_metrics_handling(network_visualizer, mock_topology_manager):
    """Test handling of metrics when rounds are skipped"""
    network_visualizer.set_topology(mock_topology_manager)

    # First round evaluation
    network_visualizer.on_event(EvaluationEvent(
        round_num=1,
        metrics={"loss": 0.5, "accuracy": 0.8}
    ))

    # Skip to third round evaluation
    network_visualizer.on_event(EvaluationEvent(
        round_num=3,
        metrics={"loss": 0.3, "accuracy": 0.9}
    ))

    # Check metrics history has the right keys
    assert set(network_visualizer.accuracy_history.keys()) == {1, 3}
    assert set(network_visualizer.loss_history.keys()) == {1, 3}

    # Check the values
    assert network_visualizer.accuracy_history[1] == 0.8
    assert network_visualizer.accuracy_history[3] == 0.9
    assert network_visualizer.loss_history[1] == 0.5
    assert network_visualizer.loss_history[3] == 0.3


def test_nested_event_handling(network_visualizer, mock_topology_manager):
    """Test handling of events with nested data structures"""
    network_visualizer.set_topology(mock_topology_manager)

    # Event with nested structure for metrics
    network_visualizer.on_event(EvaluationEvent(
        round_num=1,
        metrics={
            "main": {"loss": 0.5, "accuracy": 0.8},
            "aux": {"precision": 0.75, "recall": 0.72}
        }
    ))

    # Verify the metrics were captured correctly
    assert 1 in network_visualizer.round_metrics
    assert "main" in network_visualizer.round_metrics[1]
    assert "aux" in network_visualizer.round_metrics[1]

    # Check frame description
    frame_desc = network_visualizer.frame_descriptions[0]
    assert "Evaluation" in frame_desc


def test_aggregation_event_with_no_aggregator(network_visualizer, mock_topology_manager):
    """Test handling of aggregation event with no specific aggregator node"""
    network_visualizer.set_topology(mock_topology_manager)

    # Aggregation event without a specific aggregator (decentralized)
    network_visualizer.on_event(AggregationEvent(
        round_num=1,
        participating_nodes=[0, 1, 2],
        aggregator_node=None,
        strategy_name="GossipAvg"
    ))

    # Check frame state
    frame = network_visualizer.frames[0]
    assert frame["active_nodes"] == [0, 1, 2]
    assert frame["strategy_name"] == "GossipAvg"

    # Check description (should mention decentralized)
    assert "Decentralized aggregation" in network_visualizer.frame_descriptions[0]


@patch("matplotlib.pyplot.figure")
@patch("networkx.draw_networkx_nodes")
@patch("networkx.draw_networkx_edges")
@patch("networkx.draw_networkx_labels")
@patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), (MagicMock(), MagicMock())))
@patch("matplotlib.pyplot.savefig")
def test_render_frame_sequence_without_metrics(
        mock_savefig, mock_subplots, mock_draw_labels, mock_draw_edges, mock_draw_nodes, mock_figure,
        network_visualizer, mock_topology_manager
):
    """Test rendering frame sequence without metrics"""
    network_visualizer.set_topology(mock_topology_manager)

    # Add event with no metrics
    network_visualizer.on_event(LocalTrainingEvent(
        round_num=1,
        active_nodes=[0, 1],
        metrics={}
    ))

    # Should render without error
    network_visualizer.render_frame_sequence(prefix="test_frame")

    # Verify savefig was called
    assert mock_savefig.called


@patch("matplotlib.animation.FuncAnimation")
def test_render_training_animation_update_frames(
        mock_func_animation, populated_visualizer
):
    """Test update_frame function in render_training_animation"""
    # Extract the update_frame function from the render method
    populated_visualizer.render_training_animation(filename="test.mp4", fps=1)

    # Get the args from the FuncAnimation call
    args, kwargs = mock_func_animation.call_args
    update_frame = kwargs.get('func') or args[1]

    # Test the update_frame function with valid frame index
    result = update_frame(0)
    assert len(result) > 0  # Should return list of artists

    # Test with out-of-bounds frame index (should return empty list)
    result = update_frame(1000)
    assert len(result) == 0


@patch("matplotlib.pyplot.savefig")
def test_render_summary_plot_with_empty_data(
        mock_savefig, network_visualizer, mock_topology_manager
):
    """Test rendering summary plot with empty data"""
    network_visualizer.set_topology(mock_topology_manager)

    # Add a single frame with no data
    network_visualizer.frames = [{"round": 1, "step": "empty"}]

    # Should render without error
    network_visualizer.render_summary_plot(filename="test_summary.png")

    # savefig should have been called
    mock_savefig.assert_called_once()


def test_event_without_topology_ignored(network_visualizer):
    """Test that non-initial events without topology are ignored"""
    # Try to add various events before setting topology
    network_visualizer.on_event(LocalTrainingEvent(
        round_num=1,
        active_nodes=[0, 1],
        metrics={}
    ))

    network_visualizer.on_event(ParameterTransferEvent(
        round_num=1,
        source_nodes=[0],
        target_nodes=[1],
        param_summary={}
    ))

    # No frames should be created
    assert len(network_visualizer.frames) == 0


def test_parameter_convergence_tracking(network_visualizer, mock_topology_manager):
    """Test tracking of parameter convergence values"""
    network_visualizer.set_topology(mock_topology_manager)

    # Add model update events with convergence values
    network_visualizer.on_event(ModelUpdateEvent(
        round_num=1,
        updated_nodes=[0, 1, 2],
        param_convergence=0.5
    ))

    network_visualizer.on_event(ModelUpdateEvent(
        round_num=2,
        updated_nodes=[0, 1, 2],
        param_convergence=0.3
    ))

    # Check frames have the convergence values
    assert network_visualizer.frames[0]["param_convergence"] == 0.5
    assert network_visualizer.frames[1]["param_convergence"] == 0.3


def test_frame_history_keys_consistency(populated_visualizer):
    """Test that all frames maintain consistent history keys"""
    frames = populated_visualizer.frames

    # The latest frame should have all history
    latest_frame = frames[-1]
    assert "accuracy_history" in latest_frame
    assert "loss_history" in latest_frame
    assert "parameter_history" in latest_frame

    # All frames should have the same keys (but possibly different values
    # as history might be updated with current state at the time of event)
    for frame in frames:
        assert "accuracy_history" in frame
        assert "loss_history" in frame
        assert "parameter_history" in frame

        # Verify history keys are dictionaries or similar structures
        assert hasattr(frame["accuracy_history"], "keys")
        assert hasattr(frame["loss_history"], "keys")
        assert hasattr(frame["parameter_history"], "keys")
