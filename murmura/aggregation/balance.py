"""BALANCE: Byzantine-resilient Aggregation with distance-based filtering."""

from typing import Dict
from collections import defaultdict
import time
import numpy as np
import torch

from murmura.aggregation.base import Aggregator
from murmura.core.types import ModelState


class BALANCEAggregator(Aggregator):
    """BALANCE (Byzantine-resilient Aggregation with distance-based filtering).

    BALANCE filters neighbors based on L2 distance from own model update,
    using an adaptive threshold that tightens over training rounds.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        kappa: float = 1.0,
        alpha: float = 0.5,
        min_neighbors: int = 1,
        total_rounds: int = 20,
        **kwargs
    ):
        """Initialize BALANCE aggregator.

        Args:
            gamma: Base similarity threshold multiplier
            kappa: Exponential decay rate for threshold tightening
            alpha: Weight for own update vs neighbors (0.5 = equal weight)
            min_neighbors: Minimum neighbors to accept before fallback
            total_rounds: Total training rounds (for threshold scheduling)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.kappa = kappa
        self.alpha = alpha
        self.min_neighbors = min_neighbors
        self.total_rounds = total_rounds

        # Statistics tracking
        self.acceptance_history = []
        self.threshold_history = []
        self.neighbor_distances = defaultdict(list)

        # Performance tracking
        self.filtering_computation_time = 0.0
        self.aggregation_computation_time = 0.0

    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Aggregate using BALANCE filtering.

        Args:
            node_id: Node performing aggregation
            own_state: Own model state
            neighbor_states: Neighbor model states
            round_num: Current round

        Returns:
            Filtered and aggregated model state
        """
        # Step 1: Filter neighbors based on distance
        accepted_neighbors = self._filter_neighbors(
            own_state, neighbor_states, round_num
        )

        # Step 2: Aggregate filtered neighbors
        return self._aggregate_updates(own_state, accepted_neighbors)

    def _compute_similarity_threshold(self, own_state: ModelState, current_round: int) -> float:
        """Compute time-adaptive similarity threshold."""
        lambda_t = current_round / max(1, self.total_rounds)
        threshold_factor = self.gamma * np.exp(-self.kappa * lambda_t)
        own_state_norm = self._compute_l2_norm(own_state)
        threshold = threshold_factor * own_state_norm
        self.threshold_history.append(threshold)
        return threshold

    def _compute_l2_norm(self, model_state: ModelState) -> float:
        """Compute L2 norm of model parameters."""
        total_norm_sq = 0.0
        for param in model_state.values():
            if param.numel() > 0:
                total_norm_sq += torch.sum(param * param).item()
        return np.sqrt(total_norm_sq)

    def _compute_l2_distance(self, state1: ModelState, state2: ModelState) -> float:
        """Compute L2 distance between two model states."""
        total_dist_sq = 0.0
        common_keys = set(state1.keys()) & set(state2.keys())
        for key in common_keys:
            diff = state1[key] - state2[key]
            total_dist_sq += torch.sum(diff * diff).item()
        return np.sqrt(total_dist_sq)

    def _filter_neighbors(
        self,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        current_round: int
    ) -> Dict[int, ModelState]:
        """Filter neighbor states based on similarity to own state."""
        start_time = time.time()

        threshold = self._compute_similarity_threshold(own_state, current_round)
        accepted_neighbors = {}
        distances = {}

        for neighbor_id, neighbor_state in neighbor_states.items():
            distance = self._compute_l2_distance(own_state, neighbor_state)
            distances[neighbor_id] = distance
            self.neighbor_distances[neighbor_id].append(distance)

            if distance <= threshold:
                accepted_neighbors[neighbor_id] = neighbor_state

        acceptance_rate = len(accepted_neighbors) / max(1, len(neighbor_states))
        self.acceptance_history.append(acceptance_rate)

        # Fallback: if too few neighbors accepted, accept the closest one
        if len(accepted_neighbors) < self.min_neighbors and neighbor_states:
            closest_neighbor = min(distances.items(), key=lambda x: x[1])
            accepted_neighbors[closest_neighbor[0]] = neighbor_states[closest_neighbor[0]]

        self.filtering_computation_time += time.time() - start_time
        return accepted_neighbors

    def _aggregate_updates(
        self,
        own_state: ModelState,
        accepted_neighbors: Dict[int, ModelState]
    ) -> ModelState:
        """Aggregate own state with accepted neighbor states."""
        start_time = time.time()

        if not accepted_neighbors:
            self.aggregation_computation_time += time.time() - start_time
            return own_state

        # Compute average of accepted neighbors
        neighbor_avg = {}
        num_neighbors = len(accepted_neighbors)

        for key in own_state.keys():
            neighbor_sum = torch.zeros_like(own_state[key])
            for neighbor_state in accepted_neighbors.values():
                if key in neighbor_state:
                    neighbor_param = neighbor_state[key]
                    if neighbor_param.dtype != neighbor_sum.dtype:
                        neighbor_param = neighbor_param.to(neighbor_sum.dtype)
                    neighbor_sum += neighbor_param
            neighbor_avg[key] = neighbor_sum / num_neighbors

        # Weighted combination: alpha * own + (1-alpha) * neighbor_avg
        aggregated_state = {}
        for key in own_state.keys():
            aggregated_state[key] = (
                self.alpha * own_state[key] +
                (1 - self.alpha) * neighbor_avg[key]
            )

        self.aggregation_computation_time += time.time() - start_time
        return aggregated_state

    def get_statistics(self) -> Dict:
        """Get BALANCE algorithm statistics."""
        return {
            "mean_acceptance_rate": np.mean(self.acceptance_history) if self.acceptance_history else 0.0,
            "current_threshold": self.threshold_history[-1] if self.threshold_history else 0.0,
            "total_rounds_processed": len(self.acceptance_history),
            "filtering_computation_time": self.filtering_computation_time,
            "aggregation_computation_time": self.aggregation_computation_time,
        }
