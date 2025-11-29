"""Multi-Krum aggregator for Byzantine-resilient learning."""

from typing import Dict, List
from murmura.aggregation.base import Aggregator, compute_model_distance
from murmura.core.types import ModelState


class KrumAggregator(Aggregator):
    """Multi-Krum Byzantine-resilient aggregation.

    Krum selects the most representative model based on distances to neighbors,
    filtering out potential Byzantine updates.
    """

    def __init__(self, num_compromised: int = 0, **kwargs):
        """Initialize Krum aggregator.

        Args:
            num_compromised: Expected number of Byzantine nodes
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.num_compromised = num_compromised

    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Aggregate using Krum selection.

        Args:
            node_id: Node performing aggregation
            own_state: Own model state
            neighbor_states: Neighbor model states
            round_num: Current round

        Returns:
            Selected model state (Krum winner)
        """
        # Collect all states: own + neighbors
        all_states = [own_state] + list(neighbor_states.values())
        m = len(all_states)
        c = self.num_compromised

        # Krum requires c < (m-2)/2
        if c >= (m - 2) / 2:
            # Fallback to own state if constraint not satisfied
            return own_state

        # Compute pairwise distances
        distances = []
        for i in range(m):
            model_distances = []
            for j in range(m):
                if i != j:
                    dist = compute_model_distance(all_states[i], all_states[j])
                    model_distances.append(dist)
            distances.append(model_distances)

        # Compute Krum scores (sum of distances to m - c - 2 closest models)
        scores = []
        for i in range(m):
            sorted_distances = sorted(distances[i])
            num_closest = max(1, m - c - 2)
            closest_distances = sorted_distances[:num_closest]
            score = sum(closest_distances)
            scores.append(score)

        # Select model with minimum score
        selected_idx = scores.index(min(scores))
        return all_states[selected_idx]
