"""FedRansel: Random parameter selection for privacy-preserving federated learning.

This module implements the FedRansel algorithm which provides privacy through
random parameter sampling at both local and global levels.
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch

from murmura.aggregation.base import Aggregator, flatten_model_state
from murmura.core.types import ModelState


@dataclass
class SampledParameters:
    """Container for a node's sampled parameters."""
    node_id: int
    indices: np.ndarray  # Indices of sampled parameters (sorted)
    values: torch.Tensor  # Sampled parameter values
    sampling_ratio: float  # r_i used for this node


class FedRanselAggregator(Aggregator):
    """FedRansel centralized aggregation with random parameter selection.

    Algorithm:
    1. Each node samples parameters with ratio r_i ~ U(T_l, 1)
    2. Server receives all sampled parameter sets
    3. Server computes coverage c_j for each parameter index j
    4. Server identifies common parameters C_tau = {j : c_j >= tau}
    5. Server averages covered parameters: g_j = (1/c_j) * sum
    6. Server samples final parameters G_f with ratio T_g
    7. All nodes receive identical G_f

    This aggregator is designed for centralized mode where it receives
    ALL node states in neighbor_states (not just neighbors).

    Parameters:
        T_l: Minimum local sampling ratio (default: 0.3)
        T_g: Global sampling ratio for final selection (default: 1.0)
        tau: Minimum coverage threshold (default: 1)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        T_l: float = 0.3,
        T_g: float = 1.0,
        tau: int = 1,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.T_l = T_l
        self.T_g = T_g
        self.tau = tau
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Track the key ordering for consistent flattening/unflattening
        self._key_order: Optional[List[str]] = None
        self._key_shapes: Optional[Dict[str, torch.Size]] = None
        self._key_dtypes: Optional[Dict[str, torch.dtype]] = None

        # Statistics tracking
        self.round_stats: List[Dict[str, Any]] = []
        self.communication_costs: List[int] = []  # Parameters transmitted per round
        self.exposure_counts: Dict[int, int] = {}  # Count of times each param index exposed

        # Detailed per-round per-node tracking
        self.detailed_history: List[Dict[str, Any]] = []

    def aggregate(
        self,
        node_id: int,  # In centralized mode, this is -1 (server)
        own_state: ModelState,  # Reference state for structure
        neighbor_states: Dict[int, ModelState],  # All node states in centralized mode
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Perform FedRansel aggregation.

        In centralized mode, neighbor_states contains ALL node states.
        Returns the aggregated state to be broadcast to all nodes.

        Args:
            node_id: Node ID (-1 for server in centralized mode)
            own_state: Reference state for structure
            neighbor_states: All node states (in centralized mode)
            round_num: Current training round

        Returns:
            Aggregated model state
        """
        # Initialize key ordering from reference state if not set
        if self._key_order is None:
            self._init_key_metadata(own_state)

        # Step 1: Sample parameters from each node
        sampled_params = self._sample_from_nodes(neighbor_states)

        # Track communication cost (total parameters transmitted)
        comm_cost = sum(len(sp.indices) for sp in sampled_params)
        self.communication_costs.append(comm_cost)

        # Get total parameter count
        total_params = self._get_total_params(own_state)

        # Step 2: Compute coverage c_j for each parameter index
        coverage = self._compute_coverage(sampled_params, total_params)

        # Step 3: Identify common parameters C_tau
        common_indices = self._get_common_parameters(coverage)

        if len(common_indices) == 0:
            # No parameters meet coverage threshold, return reference state
            self._record_statistics(
                round_num, sampled_params, coverage, common_indices, np.array([])
            )
            return own_state

        # Step 4: Average common parameters
        averaged_values = self._average_parameters(
            sampled_params, coverage, common_indices, total_params
        )

        # Step 5: Sample final parameters G_f
        final_indices, final_values = self._sample_final(
            common_indices, averaged_values
        )

        # Step 6: Construct output state
        output_state = self._construct_output_state(
            own_state, final_indices, final_values
        )

        # Track statistics
        self._record_statistics(
            round_num, sampled_params, coverage, common_indices, final_indices,
            common_values=averaged_values, final_values=final_values
        )

        return output_state

    def _init_key_metadata(self, state: ModelState) -> None:
        """Initialize key ordering and metadata from reference state."""
        self._key_order = []
        self._key_shapes = {}
        self._key_dtypes = {}

        for key, param in state.items():
            if param.is_floating_point():
                self._key_order.append(key)
                self._key_shapes[key] = param.shape
                self._key_dtypes[key] = param.dtype

    def _get_total_params(self, state: ModelState) -> int:
        """Get total number of floating-point parameters."""
        return sum(
            state[key].numel() for key in self._key_order
        )

    def _flatten_state(self, state: ModelState) -> torch.Tensor:
        """Flatten state to 1D tensor using consistent key ordering."""
        params = [state[key].flatten().float() for key in self._key_order]
        return torch.cat(params)

    def _sample_from_nodes(
        self,
        node_states: Dict[int, ModelState]
    ) -> List[SampledParameters]:
        """Sample parameters from each node with random ratio r_i ~ U(T_l, 1)."""
        sampled = []

        for nid, state in node_states.items():
            # Flatten state to 1D tensor
            flat_params = self._flatten_state(state)
            total_params = len(flat_params)

            # Sample ratio r_i ~ U(T_l, 1)
            r_i = self.rng.uniform(self.T_l, 1.0)
            k_i = int(np.ceil(r_i * total_params))

            # Randomly select k_i parameter indices (without replacement)
            indices = self.rng.choice(total_params, size=k_i, replace=False)
            indices = np.sort(indices)
            values = flat_params[indices].clone()

            sampled.append(SampledParameters(
                node_id=nid,
                indices=indices,
                values=values,
                sampling_ratio=r_i
            ))

        return sampled

    def _compute_coverage(
        self,
        sampled: List[SampledParameters],
        total_params: int
    ) -> np.ndarray:
        """Compute coverage c_j for each parameter index."""
        coverage = np.zeros(total_params, dtype=np.int32)

        for sp in sampled:
            coverage[sp.indices] += 1

        return coverage

    def _get_common_parameters(self, coverage: np.ndarray) -> np.ndarray:
        """Get indices where coverage >= tau."""
        return np.where(coverage >= self.tau)[0]

    def _average_parameters(
        self,
        sampled: List[SampledParameters],
        coverage: np.ndarray,
        common_indices: np.ndarray,
        total_params: int
    ) -> torch.Tensor:
        """Average parameters weighted by coverage for common indices."""
        # Accumulate values for each parameter
        accumulated = torch.zeros(total_params, dtype=torch.float32)

        for sp in sampled:
            # Create a mask for which of this node's sampled indices are in common set
            mask = np.isin(sp.indices, common_indices)
            valid_local_indices = np.where(mask)[0]

            for local_idx in valid_local_indices:
                global_idx = sp.indices[local_idx]
                accumulated[global_idx] += sp.values[local_idx].item()

        # Divide by coverage to get average (only for common indices)
        result = torch.zeros(len(common_indices), dtype=torch.float32)
        for i, idx in enumerate(common_indices):
            result[i] = accumulated[idx] / coverage[idx]

        return result

    def _sample_final(
        self,
        common_indices: np.ndarray,
        averaged_values: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample final G_f parameters from common set with ratio T_g."""
        num_common = len(common_indices)
        num_final = int(np.ceil(self.T_g * num_common))

        if num_final >= num_common:
            return common_indices, averaged_values

        # Random selection from common parameters
        selection = self.rng.choice(num_common, size=num_final, replace=False)
        selection = np.sort(selection)

        return common_indices[selection], averaged_values[selection]

    def _construct_output_state(
        self,
        reference_state: ModelState,
        final_indices: np.ndarray,
        final_values: torch.Tensor
    ) -> ModelState:
        """Construct output ModelState from sparse indices and values.

        Parameters not in final_indices are kept from reference_state.
        Parameters in final_indices are updated with final_values.
        """
        # Start with a copy of reference state
        output = {}

        # Flatten reference to get baseline values
        flat_ref = self._flatten_state(reference_state)

        # Create updated flat tensor
        updated = flat_ref.clone()
        for i, idx in enumerate(final_indices):
            updated[idx] = final_values[i]

        # Unflatten back to state dict structure
        idx = 0
        for key in self._key_order:
            shape = self._key_shapes[key]
            numel = int(np.prod(shape))
            output[key] = updated[idx:idx + numel].reshape(shape)
            idx += numel

        # Copy non-float tensors directly from reference
        for key, param in reference_state.items():
            if not param.is_floating_point():
                output[key] = param.clone()

        return output

    def _record_statistics(
        self,
        round_num: int,
        sampled: List[SampledParameters],
        coverage: np.ndarray,
        common_indices: np.ndarray,
        final_indices: np.ndarray,
        common_values: Optional[torch.Tensor] = None,
        final_values: Optional[torch.Tensor] = None
    ) -> None:
        """Record round statistics for analysis."""
        total_params = len(coverage)
        num_nodes = len(sampled)

        # Update exposure counts
        for idx in final_indices:
            idx_int = int(idx)
            self.exposure_counts[idx_int] = self.exposure_counts.get(idx_int, 0) + 1

        # Compute statistics
        stats = {
            "round": round_num,
            "num_nodes": num_nodes,
            "avg_sampling_ratio": float(np.mean([sp.sampling_ratio for sp in sampled])),
            "total_params": total_params,
            "common_count": len(common_indices),
            "common_ratio": len(common_indices) / total_params if total_params > 0 else 0,
            "final_count": len(final_indices),
            "final_ratio": len(final_indices) / total_params if total_params > 0 else 0,
            "communication_cost": self.communication_costs[-1] if self.communication_costs else 0,
            "avg_coverage": float(np.mean(coverage[common_indices])) if len(common_indices) > 0 else 0,
        }
        self.round_stats.append(stats)

        # Store detailed per-node information
        per_node_data = {}
        for sp in sampled:
            per_node_data[sp.node_id] = {
                "r_i": float(sp.sampling_ratio),
                "S_i_indices": sp.indices.tolist(),  # Parameter indices sampled
                "S_i_count": len(sp.indices),
            }

        detailed_round = {
            "round": round_num,
            "per_node": per_node_data,
            "c_j": coverage.tolist(),  # Coverage for all parameters
            "G_a_indices": common_indices.tolist(),  # Common parameter indices
            "G_a_count": len(common_indices),
            "G_f_indices": final_indices.tolist() if len(final_indices) > 0 else [],
            "G_f_count": len(final_indices),
        }

        self.detailed_history.append(detailed_round)

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregator statistics."""
        num_rounds = len(self.round_stats)

        # Compute exposure rate (fraction of rounds each param was exposed)
        exposure_rate = {}
        if num_rounds > 0:
            for idx, count in self.exposure_counts.items():
                exposure_rate[idx] = count / num_rounds

        return {
            "algorithm": "FedRansel",
            "T_l": self.T_l,
            "T_g": self.T_g,
            "tau": self.tau,
            "rounds_processed": num_rounds,
            "total_communication": sum(self.communication_costs),
            "avg_communication_per_round": (
                sum(self.communication_costs) / num_rounds if num_rounds > 0 else 0
            ),
            "avg_final_ratio": (
                np.mean([s["final_ratio"] for s in self.round_stats])
                if self.round_stats else 0
            ),
            "avg_common_ratio": (
                np.mean([s["common_ratio"] for s in self.round_stats])
                if self.round_stats else 0
            ),
            "round_stats": self.round_stats,
            "exposure_rate": exposure_rate,
            "detailed_history": self.detailed_history,  # Per-round per-node details
        }

    def get_detailed_history(self) -> List[Dict[str, Any]]:
        """Return detailed per-round per-node tracking data.

        Returns:
            List of dictionaries, one per round, containing:
            - per_node: Dict[node_id -> {r_i, S_i_indices, S_i_count}]
            - c_j: List of coverage counts for all parameters
            - G_a_indices: Common parameter indices
            - G_f_indices: Final parameter indices
        """
        return self.detailed_history

    def get_exposure_bound(self) -> float:
        """Compute theoretical privacy bound from Theorem 1.

        P(theta_j exposed) <= (1 + T_l) / 2 * P(c_j >= tau | theta_j in S_i) * T_g

        For simplicity, this returns the upper bound assuming P(c_j >= tau) = 1.
        """
        return (1 + self.T_l) / 2 * self.T_g
