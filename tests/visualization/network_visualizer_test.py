import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

    return network_visualizer


def test_render_training_animation_with_figure_closed(populated_visualizer):
    """Test that figure is properly closed after rendering animation"""
    with patch('matplotlib.animation.FuncAnimation') as mock_animation, \
            patch('matplotlib.animation.FFMpegWriter') as mock_writer, \
            patch('matplotlib.pyplot.close') as mock_close:

        # Mock animation to return a MagicMock
        mock_animation.return_value = MagicMock()

        # Call render_training_animation
        populated_visualizer.render_training_animation(filename="test_animation.mp4")

        # Verify figure was closed
        mock_close.assert_called_once()


def test_render_frame_sequence_with_empty_frames(network_visualizer):
    """Test rendering frame sequence with empty frames list"""
    # Ensure frames list is empty
    network_visualizer.frames = []

    # Call render_frame_sequence - should not raise errors
    network_visualizer.render_frame_sequence(prefix="test_frame")

    # No frames should have been saved


def test_render_summary_plot_with_no_topology(network_visualizer):
    """Test rendering summary plot without setting topology"""
    # Set some frames but no topology
    network_visualizer.frames = [{"round": 1, "step": "test"}]

    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
            patch('matplotlib.pyplot.close') as mock_close:

        # Call render_summary_plot - should not raise errors
        network_visualizer.render_summary_plot(filename="test_summary.png")

        # Verify savefig was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


def test_update_frame_function_invalid_index(populated_visualizer):
    """Test the update_frame function with invalid frame index"""
    with patch('matplotlib.animation.FuncAnimation') as mock_animation, \
            patch('matplotlib.animation.FFMpegWriter') as mock_writer, \
            patch('matplotlib.pyplot.close'):

        # Mock animation to extract the update_frame function
        def save_update_frame(*args, **kwargs):
            test_update_frame_function_invalid_index.func = kwargs.get('func')
            return MagicMock()

        mock_animation.side_effect = save_update_frame

        # Call render_training_animation to get the update_frame function
        populated_visualizer.render_training_animation(filename="test_animation.mp4")

        # Get the update_frame function
        update_frame = test_update_frame_function_invalid_index.func

        # Test with valid index
        result_valid = update_frame(0)
        assert len(result_valid) > 0

        # Test with invalid index - should return empty list
        result_invalid = update_frame(1000)
        assert result_invalid == []


@patch('networkx.spring_layout', return_value={0: [0.1, 0.2], 1: [-0.3, 0.4], 2: [0.5, -0.6]})
def test_network_drawing_with_active_nodes(mock_spring_layout, network_visualizer, mock_topology_manager):
    """Test network drawing with active nodes and edges"""
    network_visualizer.set_topology(mock_topology_manager)

    # Add event with active nodes and edges
    network_visualizer.on_event(LocalTrainingEvent(
        round_num=1,
        active_nodes=[0, 1],
        metrics={}
    ))

    # Add event with active edges
    network_visualizer.on_event(ParameterTransferEvent(
        round_num=1,
        source_nodes=[0],
        target_nodes=[1, 2],
        param_summary={
            0: {"norm": 1.0, "mean": 0.5, "std": 0.1},
        }
    ))

    with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), (MagicMock(), MagicMock()))), \
            patch('networkx.draw_networkx_nodes') as mock_draw_nodes, \
            patch('networkx.draw_networkx_edges') as mock_draw_edges, \
            patch('matplotlib.pyplot.savefig'), \
            patch('matplotlib.pyplot.close'):

        # Call render_frame_sequence
        network_visualizer.render_frame_sequence(prefix="test_frame")

        # Verify nodes and edges were drawn
        assert mock_draw_nodes.call_count >= 2  # At least one call per frame
        assert mock_draw_edges.call_count >= 2  # At least one call per frame


def test_aggregation_event_with_star_topology(network_visualizer, mock_topology_manager):
    """Test handling of aggregation event with star topology"""
    network_visualizer.set_topology(mock_topology_manager)

    # Create aggregation event with central aggregator
    event = AggregationEvent(
        round_num=1,
        participating_nodes=[0, 1, 2, 3],
        aggregator_node=0,  # Hub node as aggregator
        strategy_name="FedAvg"
    )

    # Process the event
    network_visualizer.on_event(event)

    # Check frame data
    frame = network_visualizer.frames[0]
    assert frame["active_nodes"] == [0, 1, 2, 3]
    assert frame["strategy_name"] == "FedAvg"

    # Should have edges from each participating node to the aggregator
    edges = frame["active_edges"]
    for node in [1, 2, 3]:  # All nodes except the aggregator
        assert (node, 0) in edges  # Edge from node to aggregator

    # Check description
    assert "Aggregation at node 0" in network_visualizer.frame_descriptions[0]


def test_model_update_event_with_parameter_convergence(network_visualizer, mock_topology_manager):
    """Test handling of model update event with parameter convergence metric"""
    network_visualizer.set_topology(mock_topology_manager)

    # Create model update event with parameter convergence
    event = ModelUpdateEvent(
        round_num=1,
        updated_nodes=[0, 1, 2, 3],
        param_convergence=0.01
    )

    # Process the event
    network_visualizer.on_event(event)

    # Check frame data
    frame = network_visualizer.frames[0]
    assert frame["active_nodes"] == [0, 1, 2, 3]
    assert frame["param_convergence"] == 0.01

    # Check description
    assert "Model update" in network_visualizer.frame_descriptions[0]


def test_render_summary_plot_with_all_metrics(populated_visualizer):
    """Test rendering summary plot with all metrics"""
    # Add more rounds of metrics
    populated_visualizer.on_event(EvaluationEvent(
        round_num=2,
        metrics={"loss": 0.4, "accuracy": 0.85}
    ))
    populated_visualizer.on_event(EvaluationEvent(
        round_num=3,
        metrics={"loss": 0.3, "accuracy": 0.9}
    ))

    # Add more parameter history
    populated_visualizer.on_event(ParameterTransferEvent(
        round_num=2,
        source_nodes=[0],
        target_nodes=[1, 2],
        param_summary={
            0: {"norm": 1.2, "mean": 0.6, "std": 0.08},
            1: {"norm": 1.8, "mean": 0.65, "std": 0.15}
        }
    ))

    with patch('matplotlib.pyplot.figure'), \
            patch('matplotlib.pyplot.suptitle'), \
            patch('matplotlib.pyplot.tight_layout'), \
            patch('matplotlib.pyplot.savefig') as mock_savefig, \
            patch('matplotlib.pyplot.close'), \
            patch('networkx.draw_networkx'):

        # Call render_summary_plot
        populated_visualizer.render_summary_plot(filename="test_summary.png")

        # Verify savefig was called
        mock_savefig.assert_called_once()


def test_render_training_animation_update_function(populated_visualizer):
    """Test the update_frame function in render_training_animation"""
    # Create subplots for testing
    fig = plt.figure()
    ax_network = fig.add_subplot(221)
    ax_params = fig.add_subplot(222)
    ax_metrics = fig.add_subplot(223)
    ax_metrics_twin = ax_metrics.twinx()

    # Call render_training_animation to get the update_frame function
    with patch('matplotlib.animation.FuncAnimation') as mock_animation, \
            patch('matplotlib.animation.FFMpegWriter') as mock_writer, \
            patch('matplotlib.pyplot.figure', return_value=fig), \
            patch('matplotlib.pyplot.GridSpec'), \
            patch('matplotlib.figure.Figure.add_subplot', side_effect=[ax_network, ax_params, ax_metrics]), \
            patch('matplotlib.axes.Axes.twinx', return_value=ax_metrics_twin), \
            patch('matplotlib.pyplot.close'):

        # Mock animation to extract the update_frame function
        def save_update_frame(*args, **kwargs):
            test_render_training_animation_update_function.func = kwargs.get('func')
            return MagicMock()

        mock_animation.side_effect = save_update_frame

        # Call render_training_animation
        populated_visualizer.render_training_animation(filename="test_animation.mp4")

    # Test cleanup
    plt.close(fig)
