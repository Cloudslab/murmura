"""Random-walk mobility model for time-varying communication topology G^t.

Each node moves by a bounded random step each round on a 2-D torus.  Two
nodes share an undirected edge iff their torus-distance is less than
comm_range.  The model is fully deterministic given a seed, so every node
process on the same machine can compute G^t independently without any
out-of-band communication.
"""

import math
from typing import Dict, List, Tuple

import numpy as np


class MobilityModel:
    """Bounded random-walk mobility on a 2-D torus.

    Args:
        num_nodes:        Number of mobile nodes.
        area_size:        Side length of the square arena.
        comm_range:       Edge (i,j) ∈ G^t iff torus-dist(r_i^t, r_j^t) < comm_range.
        max_speed:        Maximum displacement magnitude per round.
        seed:             RNG seed for initial positions and movement sequences.
        ensure_connected: If True, any node with no neighbours is connected to its
                          nearest peer to guarantee a connected G^t each round.
    """

    def __init__(
        self,
        num_nodes: int,
        area_size: float = 100.0,
        comm_range: float = 30.0,
        max_speed: float = 5.0,
        seed: int = 42,
        ensure_connected: bool = True,
    ):
        self.num_nodes       = num_nodes
        self.area_size       = area_size
        self.comm_range      = comm_range
        self.max_speed       = max_speed
        self.ensure_connected = ensure_connected

        self._rng = np.random.default_rng(seed)
        # Uniform initial positions in [0, area_size)^2
        pos0 = self._rng.uniform(0.0, area_size, size=(num_nodes, 2))
        self._round_positions: Dict[int, np.ndarray] = {0: pos0}

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def positions_at(self, round_idx: int) -> np.ndarray:
        """Return (num_nodes, 2) float64 position array at *round_idx*."""
        last = max(self._round_positions)
        for r in range(last, round_idx):
            prev  = self._round_positions[r]
            delta = self._rng.uniform(-self.max_speed, self.max_speed, size=(self.num_nodes, 2))
            nxt   = (prev + delta) % self.area_size
            self._round_positions[r + 1] = nxt
        return self._round_positions[round_idx]

    def neighbors_at(self, round_idx: int) -> Dict[int, List[int]]:
        """Return adjacency list {node_id: [neighbor_ids]} at *round_idx*."""
        pos   = self.positions_at(round_idx)
        adj: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self._torus_dist(pos[i], pos[j]) < self.comm_range:
                    adj[i].append(j)
                    adj[j].append(i)

        if self.ensure_connected:
            self._connect_isolated(adj, pos)

        return adj

    def torus_dist(self, i: int, j: int, round_idx: int) -> float:
        """Torus distance between nodes i and j at round_idx."""
        pos = self.positions_at(round_idx)
        return self._torus_dist(pos[i], pos[j])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _torus_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        dx = abs(float(a[0]) - float(b[0]))
        dy = abs(float(a[1]) - float(b[1]))
        dx = min(dx, self.area_size - dx)
        dy = min(dy, self.area_size - dy)
        return math.sqrt(dx * dx + dy * dy)

    def _connect_isolated(self, adj: Dict[int, List[int]], pos: np.ndarray) -> None:
        """Connect each isolated node to its nearest peer (modifies adj in-place)."""
        for i in range(self.num_nodes):
            if adj[i]:
                continue
            nearest, _ = min(
                ((j, self._torus_dist(pos[i], pos[j])) for j in range(self.num_nodes) if j != i),
                key=lambda t: t[1],
            )
            adj[i].append(nearest)
            adj[nearest].append(i)
