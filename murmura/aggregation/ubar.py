"""UBAR: Uniform Byzantine-resilient Aggregation Rule."""

from typing import Dict, Tuple
from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from murmura.aggregation.base import Aggregator, average_states
from murmura.core.types import ModelState


class UBARAggregator(Aggregator):
    """UBAR: Two-stage Byzantine-resilient aggregation.

    Stage 1: Distance-based filtering (shortlist candidates based on L2 distance)
    Stage 2: Performance-based selection (evaluate candidates on training sample)
    """

    def __init__(
        self,
        rho: float = 0.4,
        alpha: float = 0.5,
        min_neighbors: int = 1,
        **kwargs
    ):
        """Initialize UBAR aggregator.

        Args:
            rho: Ratio of benign neighbors (Stage 1 selection ratio)
            alpha: Weight for own vs neighbors in aggregation
            min_neighbors: Minimum neighbors for fallback
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.rho = rho
        self.alpha = alpha
        self.min_neighbors = min_neighbors

        # Statistics tracking
        self.stage1_acceptance_history = []
        self.stage2_acceptance_history = []
        self.neighbor_distances = defaultdict(list)
        self.neighbor_losses = defaultdict(list)

        # Performance tracking
        self.distance_computation_time = 0.0
        self.loss_computation_time = 0.0
        self.aggregation_computation_time = 0.0

        # Cached for evaluation
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        train_loader: DataLoader = None,
        model_template: nn.Module = None,
        device: torch.device = None,
        **kwargs
    ) -> ModelState:
        """Aggregate using two-stage UBAR.

        Args:
            node_id: Node performing aggregation
            own_state: Own model state
            neighbor_states: Neighbor model states
            round_num: Current round
            train_loader: DataLoader for loss evaluation (required for Stage 2)
            model_template: Model instance for evaluation (required for Stage 2)
            device: Device for computation

        Returns:
            Aggregated model state after two-stage filtering
        """
        if not neighbor_states:
            return own_state

        # Stage 1: Distance-based filtering
        shortlisted_states = self._stage1_distance_filtering(
            own_state, neighbor_states
        )

        # Stage 2: Performance-based selection (if train_loader provided)
        if train_loader is not None and model_template is not None and device is not None:
            final_neighbors = self._stage2_performance_filtering(
                own_state, shortlisted_states, train_loader, model_template, device
            )
        else:
            # Skip Stage 2 if no training data provided
            final_neighbors = shortlisted_states

        # Aggregation
        return self._aggregate_states(own_state, final_neighbors)

    def _compute_l2_distance(self, state1: ModelState, state2: ModelState) -> float:
        """Compute L2 distance between two model states."""
        start_time = time.time()

        total_dist_sq = 0.0
        common_keys = set(state1.keys()) & set(state2.keys())
        for key in common_keys:
            diff = state1[key] - state2[key]
            total_dist_sq += torch.sum(diff * diff).item()

        self.distance_computation_time += time.time() - start_time
        return np.sqrt(total_dist_sq)

    def _stage1_distance_filtering(
        self,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState]
    ) -> Dict[int, ModelState]:
        """Stage 1: Select candidates based on parameter distances.

        Selects rho * |N_i| neighbors with smallest distances to own parameters.
        """
        if not neighbor_states:
            return {}

        # Compute distances to all neighbors
        distances = {}
        for neighbor_id, neighbor_state in neighbor_states.items():
            distance = self._compute_l2_distance(own_state, neighbor_state)
            distances[neighbor_id] = distance
            self.neighbor_distances[neighbor_id].append(distance)

        # Select top rho * |N_i| neighbors with smallest distances
        num_neighbors = len(neighbor_states)
        num_select = max(self.min_neighbors, int(self.rho * num_neighbors))

        # Sort by distance and select closest ones
        sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])
        selected_neighbor_ids = [nid for nid, _ in sorted_neighbors[:num_select]]

        # Build shortlisted neighbor states
        shortlisted_states = {
            nid: neighbor_states[nid]
            for nid in selected_neighbor_ids
        }

        stage1_acceptance_rate = len(shortlisted_states) / max(1, len(neighbor_states))
        self.stage1_acceptance_history.append(stage1_acceptance_rate)

        return shortlisted_states

    def _stage2_performance_filtering(
        self,
        own_state: ModelState,
        shortlisted_states: Dict[int, ModelState],
        train_loader: DataLoader,
        model_template: nn.Module,
        device: torch.device
    ) -> Dict[int, ModelState]:
        """Stage 2: Select final neighbors based on loss performance.

        Chooses neighbors whose loss <= own loss on a training sample.
        """
        if not shortlisted_states:
            return {}

        # Get a batch from training data for evaluation
        try:
            sample_batch = next(iter(train_loader))
        except StopIteration:
            # Fallback: use all shortlisted if no training data available
            return shortlisted_states

        # Compute own loss
        model_template.load_state_dict(own_state, strict=False)
        own_loss = self._compute_loss_with_model(model_template, sample_batch, device)

        # Evaluate each shortlisted neighbor
        final_neighbors = {}
        neighbor_losses = {}

        for neighbor_id, neighbor_state in shortlisted_states.items():
            model_template.load_state_dict(neighbor_state, strict=False)
            neighbor_loss = self._compute_loss_with_model(
                model_template, sample_batch, device
            )
            neighbor_losses[neighbor_id] = neighbor_loss
            self.neighbor_losses[neighbor_id].append(neighbor_loss)

            # Accept if loss is better or equal to own loss
            if neighbor_loss <= own_loss:
                final_neighbors[neighbor_id] = neighbor_state

        # Fallback: if no neighbors pass Stage 2, select the best one from Stage 1
        if not final_neighbors and shortlisted_states:
            best_neighbor_id = min(neighbor_losses.items(), key=lambda x: x[1])[0]
            final_neighbors[best_neighbor_id] = shortlisted_states[best_neighbor_id]

        stage2_acceptance_rate = len(final_neighbors) / max(1, len(shortlisted_states))
        self.stage2_acceptance_history.append(stage2_acceptance_rate)

        return final_neighbors

    def _compute_loss_with_model(
        self,
        model: nn.Module,
        sample_batch: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> float:
        """Compute loss with a given model on a sample batch."""
        start_time = time.time()

        model.eval()
        xb, yb = sample_batch
        xb, yb = xb.to(device), yb.to(device)

        with torch.no_grad():
            logits = model(xb)
            loss = self.criterion(logits, yb)

        self.loss_computation_time += time.time() - start_time
        return loss.item()

    def _aggregate_states(
        self,
        own_state: ModelState,
        accepted_neighbors: Dict[int, ModelState]
    ) -> ModelState:
        """Aggregate own state with accepted neighbor states."""
        start_time = time.time()

        if not accepted_neighbors:
            self.aggregation_computation_time += time.time() - start_time
            return own_state

        # Average of accepted neighbor states
        neighbor_states_list = list(accepted_neighbors.values())
        neighbor_avg_state = average_states(neighbor_states_list)

        # Weighted aggregation: alpha * own + (1-alpha) * neighbor_avg
        aggregated_state = {}
        for key in own_state.keys():
            aggregated_state[key] = (
                self.alpha * own_state[key] +
                (1 - self.alpha) * neighbor_avg_state[key]
            )

        self.aggregation_computation_time += time.time() - start_time
        return aggregated_state

    def get_statistics(self) -> Dict:
        """Get UBAR algorithm statistics."""
        return {
            "algorithm": "UBAR",
            "total_rounds_processed": len(self.stage1_acceptance_history),
            "stage1_mean_acceptance_rate": (
                np.mean(self.stage1_acceptance_history)
                if self.stage1_acceptance_history else 0.0
            ),
            "stage2_mean_acceptance_rate": (
                np.mean(self.stage2_acceptance_history)
                if self.stage2_acceptance_history else 0.0
            ),
            "overall_acceptance_rate": (
                np.mean(self.stage1_acceptance_history) * np.mean(self.stage2_acceptance_history)
                if self.stage1_acceptance_history and self.stage2_acceptance_history else 0.0
            ),
            "distance_computation_time": self.distance_computation_time,
            "loss_computation_time": self.loss_computation_time,
            "aggregation_computation_time": self.aggregation_computation_time,
        }
