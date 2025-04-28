from enum import Enum


class CoordinationMode(str, Enum):
    """
    Enumeration defining possible coordination modes for aggregation strategies.

    This defines how the topology coordinator should collect and distribute parameters.
    """

    # Centralized mode assumes a global view is required
    # All parameters are collected in a single step
    # (suitable for fedavg, trimmed_mean, etc.)
    CENTRALIZED = "centralized"

    # Decentralized mode works with local views
    # Nodes aggregate with their neighbors without global coordination
    # (suitable for gossip protocols, etc.)
    DECENTRALIZED = "decentralized"
