import pytest
from pydantic import ValidationError

from murmura.network_management.topology import TopologyConfig, TopologyType


def test_default_topology():
    """Test that the default topology is COMPLETE and default values are set."""
    config = TopologyConfig()
    assert config.topology_type == TopologyType.COMPLETE
    assert config.hub_index == 0
    assert config.adjacency_list is None


def test_star_topology_valid_hub_index():
    """Test that a valid hub index works for STAR topology."""
    config = TopologyConfig(topology_type=TopologyType.STAR, hub_index=3)
    assert config.topology_type == TopologyType.STAR
    assert config.hub_index == 3


def test_star_topology_invalid_hub_index():
    """Test that a negative hub index raises a ValidationError for STAR topology."""
    with pytest.raises(ValidationError) as exc_info:
        TopologyConfig(topology_type=TopologyType.STAR, hub_index=-1)
    assert "Hub index cannot be negative" in str(exc_info.value)


def test_custom_topology_valid_adjacency_list():
    """Test that a valid custom adjacency list is accepted."""
    adjacency = {0: [1, 2], 1: [0], 2: [0]}
    config = TopologyConfig(topology_type=TopologyType.CUSTOM, adjacency_list=adjacency)
    assert config.topology_type == TopologyType.CUSTOM
    assert config.adjacency_list == adjacency


def test_custom_topology_missing_adjacency_list():
    """Test that missing an adjacency list for CUSTOM topology raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TopologyConfig(topology_type=TopologyType.CUSTOM, adjacency_list=None)
    assert "Adjacency list required for custom topology" in str(exc_info.value)


def test_custom_topology_empty_adjacency_list():
    """Test that an empty adjacency list for CUSTOM topology raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TopologyConfig(topology_type=TopologyType.CUSTOM, adjacency_list={})
    assert "Adjacency list required for custom topology" in str(exc_info.value)


def test_custom_topology_negative_node_in_adjacency_list():
    """Test that a negative node index in the adjacency list raises a ValidationError."""
    adjacency = {
        0: [1, -2],  # -2 is an invalid negative neighbor index
        1: [0],
    }
    with pytest.raises(ValidationError) as exc_info:
        TopologyConfig(topology_type=TopologyType.CUSTOM, adjacency_list=adjacency)
    assert "Negative node indices not allowed" in str(exc_info.value)
