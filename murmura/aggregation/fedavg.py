"""Decentralized FedAvg aggregator."""

from typing import Dict
from murmura.aggregation.base import Aggregator, average_states
from murmura.core.types import ModelState


class FedAvgAggregator(Aggregator):
    """Decentralized Federated Averaging.

    Simple averaging of own model with all neighbor models.
    Each model (including own) receives equal weight.
    """

    def __init__(self, **kwargs):
        """Initialize FedAvg aggregator."""
        super().__init__(**kwargs)

    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Aggregate by simple averaging.

        Args:
            node_id: Node performing aggregation
            own_state: Own model state
            neighbor_states: Neighbor model states
            round_num: Current round

        Returns:
            Averaged model state (own + all neighbors)
        """
        # Collect all states: own + neighbors
        all_states = [own_state] + list(neighbor_states.values())

        # Equal-weight averaging
        return average_states(all_states)
