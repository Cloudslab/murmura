"""Sketchguard: Compressed Approximate Robust Secure Estimation."""

from typing import Dict, List
from collections import defaultdict, deque
import time
import numpy as np
import torch

from murmura.aggregation.base import Aggregator, average_states
from murmura.core.types import ModelState


class SketchguardAggregator(Aggregator):
    """Sketchguard aggregation with Count-Sketch compression.

    Uses Count-Sketch for filtering decisions and full model parameters
    for aggregation, providing Byzantine-resilience with lower communication.
    """

    def __init__(
        self,
        model_dim: int,
        sketch_size: int = 1000,
        gamma: float = 2.0,
        kappa: float = 1.0,
        alpha: float = 0.5,
        min_neighbors: int = 1,
        network_seed: int = 42,
        attack_detection_window: int = 5,
        total_rounds: int = 20,
        **kwargs
    ):
        """Initialize Sketchguard aggregator.

        Args:
            model_dim: Total number of model parameters
            sketch_size: Compression size (lower = more compression)
            gamma: Base similarity threshold multiplier
            kappa: Exponential decay rate
            alpha: Weight for own vs neighbors
            min_neighbors: Minimum neighbors before fallback
            network_seed: Seed for hash function generation
            attack_detection_window: Window for tracking acceptance rates
            total_rounds: Total training rounds
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.sketch_size = sketch_size
        self.gamma = gamma
        self.kappa = kappa
        self.alpha = alpha
        self.min_neighbors = min_neighbors
        self.network_seed = network_seed
        self.total_rounds = total_rounds

        # Generate Count-Sketch hash tables
        self.hash_table, self.sign_table = self._generate_count_sketch_tables()

        # Statistics tracking
        self.acceptance_history = []
        self.threshold_history = []
        self.neighbor_scores = defaultdict(list)
        self.attack_history = deque(maxlen=attack_detection_window)

        # Performance tracking
        self.sketch_computation_time = 0.0
        self.filtering_computation_time = 0.0
        self.aggregation_computation_time = 0.0

    def _generate_count_sketch_tables(self):
        """Generate Count-Sketch tables as numpy arrays for speed."""
        rng = np.random.RandomState(self.network_seed)
        hash_table = rng.randint(0, self.sketch_size, size=self.model_dim)
        sign_table = rng.choice([-1, 1], size=self.model_dim)
        return hash_table, sign_table

    def _flatten_model_state(self, model_state: ModelState) -> np.ndarray:
        """Flatten model parameters into a single vector."""
        flattened_parts = []
        for param in model_state.values():
            flattened_parts.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(flattened_parts)

    def _count_sketch_compress(self, vector: np.ndarray) -> np.ndarray:
        """Count-Sketch compression using vectorized operations."""
        start_time = time.time()

        vector_len = len(vector)
        if vector_len != len(self.hash_table):
            hash_buckets = self.hash_table[:vector_len]
            signs = self.sign_table[:vector_len]
        else:
            hash_buckets = self.hash_table
            signs = self.sign_table

        # Vectorized computation using np.bincount
        signed_values = signs * vector
        sketch = np.bincount(
            hash_buckets,
            weights=signed_values,
            minlength=self.sketch_size
        )

        self.sketch_computation_time += time.time() - start_time
        return sketch

    def get_sketch(self, model_state: ModelState) -> np.ndarray:
        """Get Count-Sketch of model state for sharing with neighbors.

        Args:
            model_state: Model state to compress

        Returns:
            Compressed sketch as numpy array
        """
        flattened = self._flatten_model_state(model_state)
        return self._count_sketch_compress(flattened)

    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        neighbor_sketches: Dict[int, np.ndarray] = None,
        **kwargs
    ) -> ModelState:
        """Aggregate using Sketchguard filtering.

        Args:
            node_id: Node performing aggregation
            own_state: Own model state
            neighbor_states: Neighbor model states
            round_num: Current round
            neighbor_sketches: Pre-computed neighbor sketches (optional)

        Returns:
            Filtered and aggregated model state
        """
        # Step 1: Compute own sketch
        own_sketch = self.get_sketch(own_state)

        # Step 2: Compute or use provided neighbor sketches
        if neighbor_sketches is None:
            neighbor_sketches = {
                nid: self.get_sketch(nstate)
                for nid, nstate in neighbor_states.items()
            }

        # Step 3: Filter neighbors based on sketch distances
        accepted_neighbor_ids = self._filter_neighbors_by_sketch(
            own_sketch, neighbor_sketches, round_num
        )

        # Step 4: Get states from accepted neighbors
        accepted_neighbor_states = {
            nid: neighbor_states[nid]
            for nid in accepted_neighbor_ids
            if nid in neighbor_states
        }

        # Step 5: Aggregate model states
        return self._aggregate_states(own_state, accepted_neighbor_states)

    def _compute_sketch_distances(
        self,
        own_sketch: np.ndarray,
        neighbor_sketches: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        """Compute L2 distances between sketches."""
        start_time = time.time()

        distances = {}
        for neighbor_id, neighbor_sketch in neighbor_sketches.items():
            distance = np.linalg.norm(own_sketch - neighbor_sketch)
            distances[neighbor_id] = distance
            self.neighbor_scores[neighbor_id].append(distance)

        self.filtering_computation_time += time.time() - start_time
        return distances

    def _adaptive_threshold(self, current_round: int, own_sketch_norm: float) -> float:
        """Compute adaptive acceptance threshold."""
        # Time decay similar to BALANCE
        lambda_t = current_round / max(1, self.total_rounds)
        time_factor = self.gamma * np.exp(-self.kappa * lambda_t)

        # Attack detection: increase threshold if recent low acceptance rates
        attack_factor = 1.0
        if len(self.attack_history) >= 3:
            recent_acceptance = np.mean(list(self.attack_history)[-3:])
            if recent_acceptance < 0.3:
                attack_factor = 1.5

        threshold = time_factor * attack_factor * own_sketch_norm
        self.threshold_history.append(threshold)
        return threshold

    def _filter_neighbors_by_sketch(
        self,
        own_sketch: np.ndarray,
        neighbor_sketches: Dict[int, np.ndarray],
        current_round: int
    ) -> List[int]:
        """Filter neighbors based on sketch distances."""
        distances = self._compute_sketch_distances(own_sketch, neighbor_sketches)
        own_sketch_norm = np.linalg.norm(own_sketch)
        threshold = self._adaptive_threshold(current_round, own_sketch_norm)

        # Accept neighbors within threshold
        accepted_neighbors = []
        for neighbor_id, distance in distances.items():
            if distance <= threshold:
                accepted_neighbors.append(neighbor_id)

        # Track acceptance rate
        acceptance_rate = len(accepted_neighbors) / max(1, len(neighbor_sketches))
        self.acceptance_history.append(acceptance_rate)
        self.attack_history.append(acceptance_rate)

        # Fallback: accept closest neighbor if too few accepted
        if len(accepted_neighbors) < self.min_neighbors and neighbor_sketches:
            closest_neighbor_id = min(distances.items(), key=lambda x: x[1])[0]
            if closest_neighbor_id not in accepted_neighbors:
                accepted_neighbors.append(closest_neighbor_id)

        return accepted_neighbors

    def _aggregate_states(
        self,
        own_state: ModelState,
        accepted_neighbor_states: Dict[int, ModelState]
    ) -> ModelState:
        """Aggregate model states using BALANCE-style weighted averaging."""
        start_time = time.time()

        if not accepted_neighbor_states:
            self.aggregation_computation_time += time.time() - start_time
            return own_state

        # Compute average neighbor state
        neighbor_states_list = list(accepted_neighbor_states.values())
        neighbor_avg_state = average_states(neighbor_states_list)

        # Weighted combination
        aggregated_state = {}
        for key in own_state.keys():
            aggregated_state[key] = (
                self.alpha * own_state[key] +
                (1 - self.alpha) * neighbor_avg_state[key]
            )

        self.aggregation_computation_time += time.time() - start_time
        return aggregated_state

    def get_statistics(self) -> Dict:
        """Get Sketchguard algorithm statistics."""
        return {
            "algorithm": "Sketchguard",
            "mean_acceptance_rate": np.mean(self.acceptance_history) if self.acceptance_history else 0.0,
            "current_threshold": self.threshold_history[-1] if self.threshold_history else 0.0,
            "total_rounds_processed": len(self.acceptance_history),
            "sketch_computation_time": self.sketch_computation_time,
            "filtering_computation_time": self.filtering_computation_time,
            "aggregation_computation_time": self.aggregation_computation_time,
            "compression_ratio": self.model_dim / self.sketch_size,
        }
