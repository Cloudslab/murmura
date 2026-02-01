"""Top-K aggregation: Communication-efficient FL with sparse updates.

This module implements Top-K parameter selection, where only the top K%
of parameters (by magnitude) are communicated and aggregated.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from murmura.aggregation.base import Aggregator, average_states
from murmura.core.types import ModelState


class TopKAggregator(Aggregator):
    """Top-K parameter selection aggregator.

    Each round:
    1. Compute updates (current state - previous state) for each node
    2. Average the updates
    3. Select top K% parameters by magnitude from the averaged update
    4. Apply only the selected parameters to the previous global state

    This reduces communication by only transmitting K% of parameters.

    Parameters:
        k_ratio: Fraction of parameters to select (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        k_ratio: float = 0.1,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.k_ratio = k_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Track previous global state for computing updates
        self.previous_state: Optional[ModelState] = None

        # Track key ordering for consistent operations
        self._key_order: Optional[List[str]] = None

        # Statistics tracking
        self.sparsity_ratios: List[float] = []
        self.communication_costs: List[int] = []
        self.round_count = 0

    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Aggregate using Top-K selection.

        Args:
            node_id: ID of the aggregating node
            own_state: Node's own model state
            neighbor_states: Dictionary of neighbor model states
            round_num: Current training round

        Returns:
            Aggregated model state with sparse update applied
        """
        all_states = [own_state] + list(neighbor_states.values())

        # Initialize key ordering from first state
        if self._key_order is None:
            self._key_order = [
                k for k in own_state.keys() if own_state[k].is_floating_point()
            ]

        # First round: use full average as baseline
        if self.previous_state is None:
            self.previous_state = average_states(all_states)
            total_params = self._count_params(self.previous_state)
            self.sparsity_ratios.append(1.0)  # Full update in first round
            self.communication_costs.append(total_params)
            self.round_count += 1
            return self.previous_state

        # Compute updates for each state relative to previous global
        updates = []
        for state in all_states:
            update = self._compute_update(state, self.previous_state)
            updates.append(update)

        # Average updates
        avg_update = average_states(updates)

        # Apply Top-K mask to averaged update
        masked_update, selected_count = self._topk_mask(avg_update)

        # Apply masked update to previous state
        new_state = self._apply_update(self.previous_state, masked_update)

        # Track statistics
        total_params = self._count_params(new_state)
        self.sparsity_ratios.append(selected_count / total_params)
        self.communication_costs.append(selected_count)

        self.previous_state = new_state
        self.round_count += 1
        return new_state

    def _count_params(self, state: ModelState) -> int:
        """Count total floating-point parameters in state."""
        return sum(
            p.numel() for p in state.values() if p.is_floating_point()
        )

    def _flatten_state(self, state: ModelState) -> torch.Tensor:
        """Flatten state to 1D tensor using consistent key ordering."""
        params = [state[k].flatten().float() for k in self._key_order]
        return torch.cat(params)

    def _unflatten_state(
        self,
        flat: torch.Tensor,
        reference: ModelState
    ) -> ModelState:
        """Unflatten 1D tensor back to state dict structure."""
        result = {}
        idx = 0
        for key in self._key_order:
            shape = reference[key].shape
            numel = reference[key].numel()
            result[key] = flat[idx:idx + numel].reshape(shape)
            idx += numel

        # Copy non-float tensors
        for key, param in reference.items():
            if not param.is_floating_point():
                result[key] = param.clone()

        return result

    def _compute_update(
        self,
        current: ModelState,
        previous: ModelState
    ) -> ModelState:
        """Compute update delta between states."""
        update = {}
        for key in current:
            if current[key].is_floating_point():
                update[key] = current[key] - previous[key]
            else:
                update[key] = current[key].clone()
        return update

    def _topk_mask(
        self,
        update: ModelState
    ) -> Tuple[ModelState, int]:
        """Apply Top-K mask to update, keeping only top K% by magnitude.

        Args:
            update: Model update to mask

        Returns:
            Tuple of (masked update, number of selected parameters)
        """
        # Flatten to find top-k indices globally
        flat_update = self._flatten_state(update)
        total_params = len(flat_update)

        k = max(1, int(np.ceil(self.k_ratio * total_params)))

        # Get indices of top-k by absolute magnitude
        _, topk_indices = torch.topk(torch.abs(flat_update), k)

        # Create mask
        mask = torch.zeros_like(flat_update)
        mask[topk_indices] = 1.0

        # Apply mask
        masked_flat = flat_update * mask

        # Unflatten back to state structure
        masked = self._unflatten_state(masked_flat, update)

        return masked, k

    def _apply_update(
        self,
        base: ModelState,
        update: ModelState
    ) -> ModelState:
        """Apply update to base state."""
        result = {}
        for key in base:
            if base[key].is_floating_point():
                result[key] = base[key] + update[key]
            else:
                result[key] = base[key].clone()
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregator statistics."""
        return {
            "algorithm": "TopK",
            "k_ratio": self.k_ratio,
            "rounds_processed": self.round_count,
            "avg_sparsity": (
                float(np.mean(self.sparsity_ratios))
                if self.sparsity_ratios else self.k_ratio
            ),
            "total_communication": sum(self.communication_costs),
            "avg_communication_per_round": (
                sum(self.communication_costs) / self.round_count
                if self.round_count > 0 else 0
            ),
        }

    def get_communication_savings(self) -> Dict[str, float]:
        """Calculate communication savings compared to full model transmission.

        Returns:
            Dictionary with savings metrics
        """
        if not self.communication_costs or self.round_count == 0:
            return {"savings_ratio": 1 - self.k_ratio, "note": "estimated"}

        # Estimate full communication (first round has full update)
        full_params = self.communication_costs[0] if self.communication_costs else 0
        total_sparse = sum(self.communication_costs[1:]) if len(self.communication_costs) > 1 else 0
        total_full = full_params * (self.round_count - 1)

        if total_full == 0:
            return {"savings_ratio": 0.0}

        actual_savings = 1 - (total_sparse / total_full)

        return {
            "savings_ratio": actual_savings,
            "theoretical_savings": 1 - self.k_ratio,
            "total_params_transmitted": sum(self.communication_costs),
            "full_transmission_equivalent": full_params * self.round_count,
        }
