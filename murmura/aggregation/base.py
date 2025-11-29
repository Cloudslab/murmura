"""Base aggregator protocol and helper functions."""

from typing import Protocol, Dict, List, Any, Optional
from abc import ABC, abstractmethod
import torch
from murmura.core.types import ModelState


class Aggregator(ABC):
    """Base class for aggregation algorithms.

    Aggregators define how nodes combine model updates from their neighbors
    in a decentralized federated learning setting.
    """

    def __init__(self, **kwargs):
        """Initialize aggregator with configuration parameters."""
        self.config = kwargs

    @abstractmethod
    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Aggregate own model state with neighbor states.

        Args:
            node_id: ID of the node performing aggregation
            own_state: Node's own model state
            neighbor_states: Dictionary mapping neighbor IDs to their model states
            round_num: Current training round number
            **kwargs: Additional context (e.g., train_loader for UBAR)

        Returns:
            Aggregated model state
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics for monitoring.

        Returns:
            Dictionary of statistics (e.g., acceptance rates, computation time)
        """
        return {}


# Helper functions for model state manipulation

def get_model_state(model: torch.nn.Module) -> ModelState:
    """Extract state dictionary from a model.

    Args:
        model: PyTorch model

    Returns:
        State dictionary mapping parameter names to tensors
    """
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def set_model_state(model: torch.nn.Module, state: ModelState) -> None:
    """Load state dictionary into a model.

    Args:
        model: PyTorch model
        state: State dictionary to load
    """
    model.load_state_dict(state)


def average_states(
    states: List[ModelState],
    weights: Optional[List[float]] = None
) -> ModelState:
    """Average multiple model states with optional weights.

    Args:
        states: List of model state dictionaries
        weights: Optional weights for each state (must sum to 1)

    Returns:
        Averaged model state
    """
    if not states:
        raise ValueError("Cannot average empty list of states")

    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    else:
        if len(weights) != len(states):
            raise ValueError(f"weights length ({len(weights)}) != states length ({len(states)})")
        if not abs(sum(weights) - 1.0) < 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")

    # Initialize with zeros
    averaged = {}
    for key in states[0].keys():
        averaged[key] = torch.zeros_like(states[0][key])

    # Weighted sum
    for state, weight in zip(states, weights):
        for key in averaged.keys():
            averaged[key] += weight * state[key]

    return averaged


def compute_model_distance(state1: ModelState, state2: ModelState) -> float:
    """Compute L2 distance between two model states.

    Args:
        state1: First model state
        state2: Second model state

    Returns:
        L2 distance as a float
    """
    distance = 0.0
    for key in state1.keys():
        distance += torch.norm(state1[key] - state2[key]).item() ** 2
    return distance ** 0.5


def flatten_model_state(state: ModelState) -> torch.Tensor:
    """Flatten a model state dictionary into a 1D tensor.

    Args:
        state: Model state dictionary

    Returns:
        Flattened 1D tensor
    """
    return torch.cat([param.flatten() for param in state.values()])


def calculate_model_dimension(model: torch.nn.Module) -> int:
    """Calculate total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())
