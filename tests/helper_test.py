from unittest.mock import Mock

import matplotlib.pyplot as plt
import pytest

from murmura.helper import visualize_network_topology
from murmura.orchestration.cluster_manager import ClusterManager


@pytest.fixture
def mock_cluster_manager():
    """Fixture that returns a mocked ClusterManager with a predefined topology."""
    mock_manager = Mock(spec=ClusterManager)
    mock_manager.topology_manager = Mock()
    mock_manager.topology_manager.adjacency_list = {0: [1, 2], 1: [0], 2: [0]}
    return mock_manager


def test_visualize_network_topology_with_mock(mock_cluster_manager, monkeypatch):
    """Test that visualization runs correctly when topology exists."""
    show_called = False

    def dummy_show():
        nonlocal show_called
        show_called = True

    monkeypatch.setattr(plt, "show", dummy_show)

    visualize_network_topology(mock_cluster_manager)

    assert show_called, "Expected plt.show() to be called."


def test_visualize_network_topology_without_topology():
    """Test that visualization raises an error when topology is not initialized."""
    mock_manager = Mock(spec=ClusterManager)
    mock_manager.topology_manager = None

    with pytest.raises(ValueError, match="Topology not initialized"):
        visualize_network_topology(mock_manager)
