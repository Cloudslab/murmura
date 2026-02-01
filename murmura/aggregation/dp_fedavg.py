"""DP-FedAvg: Differentially Private Federated Averaging.

This module implements FedAvg with Gaussian mechanism for differential privacy.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch

from murmura.aggregation.base import Aggregator, average_states
from murmura.core.types import ModelState


class DPFedAvgAggregator(Aggregator):
    """Differential Privacy FedAvg with Gaussian mechanism.

    Adds calibrated Gaussian noise to achieve (epsilon, delta)-DP.
    The noise scale is computed based on the privacy parameters and
    an assumed sensitivity (gradient clipping norm).

    For the Gaussian mechanism:
        sigma = sqrt(2 * ln(1.25/delta)) * C / epsilon

    where C is the L2 sensitivity (clip_norm).

    Parameters:
        epsilon: Privacy budget (default: 4.0)
        delta: Privacy failure probability (default: 1e-5)
        clip_norm: Gradient/update clipping norm C (default: 1.0)
        noise_multiplier: Alternative to epsilon - directly set noise scale.
                         If provided, overrides epsilon-based calculation.
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        epsilon: float = 4.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Compute noise scale from privacy parameters
        if noise_multiplier is not None:
            self.noise_scale = noise_multiplier * clip_norm
        else:
            # Standard Gaussian mechanism: sigma = sqrt(2 * ln(1.25/delta)) * C / epsilon
            self.noise_scale = (
                np.sqrt(2 * np.log(1.25 / delta)) * clip_norm / epsilon
            )

        # Statistics tracking
        self.noise_magnitudes: List[float] = []
        self.round_count = 0

        # Detailed per-round tracking
        self.round_history: List[Dict[str, Any]] = []
        self.cumulative_epsilon: float = 0.0
        self.cumulative_delta: float = 0.0

    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Aggregate with DP noise injection.

        Performs standard FedAvg averaging first, then adds calibrated
        Gaussian noise to the result.

        Args:
            node_id: ID of the aggregating node
            own_state: Node's own model state
            neighbor_states: Dictionary of neighbor model states
            round_num: Current training round

        Returns:
            Differentially private aggregated model state
        """
        # Standard FedAvg first
        all_states = [own_state] + list(neighbor_states.values())
        averaged = average_states(all_states)

        # Add calibrated Gaussian noise
        noised_state, noise_info = self._add_noise_tracked(averaged)

        # Update cumulative privacy budget (simple composition)
        self.cumulative_epsilon += self.epsilon
        self.cumulative_delta += self.delta

        # Record round details
        round_detail = {
            "round": round_num,
            "node_id": node_id,
            "num_states_aggregated": len(all_states),
            "epsilon_this_round": self.epsilon,
            "delta_this_round": self.delta,
            "cumulative_epsilon": self.cumulative_epsilon,
            "cumulative_delta": min(self.cumulative_delta, 1.0),
            "noise_scale": self.noise_scale,
            "noise_magnitude": noise_info["noise_magnitude"],
            "noise_per_param": noise_info["noise_per_param"],
        }
        self.round_history.append(round_detail)

        self.round_count += 1
        return noised_state

    def _add_noise(self, state: ModelState) -> ModelState:
        """Add Gaussian noise calibrated for DP guarantee.

        Args:
            state: Model state to add noise to

        Returns:
            Noised model state
        """
        noised, _ = self._add_noise_tracked(state)
        return noised

    def _add_noise_tracked(self, state: ModelState) -> tuple:
        """Add Gaussian noise with detailed tracking.

        Args:
            state: Model state to add noise to

        Returns:
            Tuple of (noised_state, noise_info_dict)
        """
        noised = {}
        total_noise_norm_sq = 0.0
        total_params = 0

        for key, param in state.items():
            if param.is_floating_point():
                # Generate Gaussian noise with the computed scale
                noise = torch.randn_like(param) * self.noise_scale
                noised[key] = param + noise
                total_noise_norm_sq += torch.sum(noise ** 2).item()
                total_params += param.numel()
            else:
                # Non-float tensors (e.g., BatchNorm's num_batches_tracked)
                noised[key] = param.clone()

        noise_magnitude = np.sqrt(total_noise_norm_sq)
        self.noise_magnitudes.append(noise_magnitude)

        noise_info = {
            "noise_magnitude": noise_magnitude,
            "noise_per_param": noise_magnitude / total_params if total_params > 0 else 0.0,
            "total_params": total_params,
        }

        return noised, noise_info

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregator statistics."""
        return {
            "algorithm": "DP-FedAvg",
            "epsilon": self.epsilon,
            "delta": self.delta,
            "clip_norm": self.clip_norm,
            "noise_scale": self.noise_scale,
            "rounds_processed": self.round_count,
            "avg_noise_magnitude": (
                float(np.mean(self.noise_magnitudes))
                if self.noise_magnitudes else 0.0
            ),
            "total_noise_magnitude": (
                float(np.sum(self.noise_magnitudes))
                if self.noise_magnitudes else 0.0
            ),
            # Privacy budget tracking
            "cumulative_epsilon": self.cumulative_epsilon,
            "cumulative_delta": min(self.cumulative_delta, 1.0),
            # Detailed per-round history
            "round_history": self.round_history,
        }

    def get_budget_history(self) -> List[Dict[str, Any]]:
        """Return per-round privacy budget consumption history.

        Returns:
            List of dicts with per-round epsilon/delta consumption
        """
        return self.round_history

    def get_privacy_spent(self, num_rounds: Optional[int] = None) -> Dict[str, float]:
        """Estimate total privacy spent after training.

        Uses simple composition (linear in rounds). For tighter bounds,
        consider using Renyi DP or moments accountant.

        Args:
            num_rounds: Number of rounds (defaults to rounds processed)

        Returns:
            Dictionary with epsilon and delta estimates
        """
        if num_rounds is None:
            num_rounds = self.round_count

        # Simple composition: epsilon grows linearly
        # For tighter bounds, use advanced composition or RDP
        total_epsilon = self.epsilon * num_rounds
        total_delta = self.delta * num_rounds

        return {
            "total_epsilon": total_epsilon,
            "total_delta": min(total_delta, 1.0),
            "composition": "linear",
        }
