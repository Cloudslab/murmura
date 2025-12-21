"""Network topology generators."""

import random
from typing import List, Tuple, Literal
from murmura.topology.base import Topology


TopologyType = Literal["ring", "fully", "erdos", "k-regular"]


def create_topology(
    topology_type: TopologyType,
    num_nodes: int,
    **kwargs
) -> Topology:
    """Create a network topology.

    Args:
        topology_type: Type of topology ('ring', 'fully', 'erdos', 'k-regular')
        num_nodes: Number of nodes in the network
        **kwargs: Additional parameters specific to topology type:
            - p (float): Edge probability for 'erdos' (default: 0.3)
            - k (int): Degree for 'k-regular' (default: 4, must be even)
            - seed (int): Random seed for reproducibility (default: 12345)

    Returns:
        Topology object

    Raises:
        ValueError: If topology_type is unknown or parameters are invalid
    """
    topology_type = topology_type.lower()

    if topology_type == "ring":
        return _create_ring(num_nodes)
    elif topology_type in ("fully", "full"):
        return _create_fully_connected(num_nodes)
    elif topology_type in ("erdos", "er", "erdos-renyi"):
        p = kwargs.get("p", 0.3)
        seed = kwargs.get("seed", 12345)
        return _create_erdos_renyi(num_nodes, p, seed)
    elif topology_type in ("k-regular", "kregular"):
        k = kwargs.get("k", 4)
        return _create_k_regular(num_nodes, k)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")


def _create_ring(n: int) -> Topology:
    """Create a ring topology where each node connects to its immediate neighbors."""
    neighbors = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []

    for i in range(n):
        j = (i + 1) % n
        neighbors[i].append(j)
        neighbors[j].append(i)
        edges.append((min(i, j), max(i, j)))

    # Remove duplicates and sort
    neighbors = [sorted(set(ns)) for ns in neighbors]
    edges = sorted(set(edges))

    return Topology(num_nodes=n, neighbors=neighbors, edges=edges)


def _create_fully_connected(n: int) -> Topology:
    """Create a fully connected topology where all nodes connect to all others."""
    neighbors = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            neighbors[i].append(j)
            neighbors[j].append(i)
            edges.append((i, j))

    return Topology(num_nodes=n, neighbors=neighbors, edges=edges)


def _create_erdos_renyi(n: int, p: float, seed: int = 12345) -> Topology:
    """Create an Erdős-Rényi random graph with edge probability p."""
    if not 0 <= p <= 1:
        raise ValueError(f"Edge probability p must be in [0, 1], got {p}")

    rng = random.Random(seed)
    neighbors = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges.append((i, j))

    # Ensure connectivity: if any node has no neighbors, connect it to next node
    for i in range(n):
        if not neighbors[i]:
            j = (i + 1) % n
            neighbors[i].append(j)
            neighbors[j].append(i)
            edges.append((min(i, j), max(i, j)))

    neighbors = [sorted(set(ns)) for ns in neighbors]
    edges = sorted(set(edges))

    return Topology(num_nodes=n, neighbors=neighbors, edges=edges)


def _create_k_regular(n: int, k: int) -> Topology:
    """Create a k-regular ring lattice (circulant graph).

    Each node connects to k/2 predecessors and k/2 successors.
    """
    if k % 2 != 0:
        print(f"Warning: k={k} is odd, using k={k+1} for regular ring lattice")
        k = k + 1

    if k >= n:
        print(f"Warning: k={k} >= n={n}, creating fully connected graph")
        return _create_fully_connected(n)

    neighbors = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []
    half_k = k // 2

    for i in range(n):
        # Connect to k/2 successors and k/2 predecessors
        for offset in range(1, half_k + 1):
            j = (i + offset) % n
            if j not in neighbors[i]:
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges.append((min(i, j), max(i, j)))

    neighbors = [sorted(set(ns)) for ns in neighbors]
    edges = sorted(set(edges))

    return Topology(num_nodes=n, neighbors=neighbors, edges=edges)
