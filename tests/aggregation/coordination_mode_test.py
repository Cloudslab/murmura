import pytest

from murmura.aggregation.coordination_mode import CoordinationMode


def test_coordination_mode_enum_values():
    """Test the CoordinationMode enum has the expected values"""
    assert CoordinationMode.CENTRALIZED == "centralized"
    assert CoordinationMode.DECENTRALIZED == "decentralized"


def test_coordination_mode_uniqueness():
    """Test that the enum values are unique"""
    values = [mode.value for mode in CoordinationMode]
    assert len(values) == len(set(values))


def test_coordination_mode_usage():
    """Test using the CoordinationMode enum in comparisons"""
    mode = CoordinationMode.CENTRALIZED

    assert mode == CoordinationMode.CENTRALIZED
    assert mode != CoordinationMode.DECENTRALIZED
    assert mode == "centralized"

    # Test string comparison works both ways
    assert "centralized" == CoordinationMode.CENTRALIZED
    assert "decentralized" != CoordinationMode.CENTRALIZED
