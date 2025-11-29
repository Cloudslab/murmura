"""Base topology dataclass and utilities."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Topology:
    """Network topology structure for decentralized communication.

    Attributes:
        num_nodes: Number of nodes in the network
        neighbors: Adjacency list where neighbors[i] contains node i's neighbors
        edges: List of undirected edges as (node1, node2) tuples
    """

    num_nodes: int
    neighbors: List[List[int]]
    edges: List[Tuple[int, int]]

    def __post_init__(self):
        """Validate topology after initialization."""
        assert len(self.neighbors) == self.num_nodes, \
            f"neighbors list length ({len(self.neighbors)}) != num_nodes ({self.num_nodes})"

    def degree(self, node_id: int) -> int:
        """Get the degree of a node.

        Args:
            node_id: Node index

        Returns:
            Number of neighbors
        """
        return len(self.neighbors[node_id])

    def avg_degree(self) -> float:
        """Calculate average node degree."""
        return sum(len(neighbors) for neighbors in self.neighbors) / self.num_nodes

    def is_connected(self) -> bool:
        """Check if the topology is connected using BFS.

        Returns:
            True if all nodes are reachable from node 0
        """
        if self.num_nodes == 0:
            return True

        visited = set([0])
        queue = [0]

        while queue:
            current = queue.pop(0)
            for neighbor in self.neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == self.num_nodes
