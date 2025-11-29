"""
Murmura: A modular framework for decentralized federated learning.

Murmura provides a composable, config-driven approach to building and running
decentralized federated learning experiments with Byzantine-resilient aggregation.
"""

__version__ = "0.1.0"

from murmura.config import Config
from murmura.core import Network, Node
from murmura.topology import create_topology, Topology
from murmura.aggregation import (
    FedAvgAggregator,
    KrumAggregator,
    BALANCEAggregator,
    SketchguardAggregator,
    UBARAggregator,
)

__all__ = [
    "__version__",
    "Config",
    "Network",
    "Node",
    "create_topology",
    "Topology",
    "FedAvgAggregator",
    "KrumAggregator",
    "BALANCEAggregator",
    "SketchguardAggregator",
    "UBARAggregator",
]
