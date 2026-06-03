"""Contact-trace mobility model for time-varying communication topology G^t.

Loads a preprocessed contact-trace CSV and serves per-round adjacency lists
with the same interface as MobilityModel, so NodeProcess and DMTTNodeProcess
can use either without modification.

Expected CSV format (produced by scripts/prepare_sociopatterns.py):
    round,node_i,node_j
    0,0,3
    0,1,4
    ...

Edges are undirected; each (i,j) pair appears once with i < j.
If round_idx >= total rounds in the trace the index wraps modulo total,
so a short trace (e.g. one school day) can drive an arbitrarily long run.
"""

import csv
from pathlib import Path
from typing import Dict, List


class TraceBasedMobility:
    """Serves G^t from a pre-processed real-world contact trace.

    Args:
        trace_path:       Path to the preprocessed CSV file.
        num_nodes:        Number of nodes; must match topology.num_nodes.
        ensure_connected: If True, any isolated node at a given round is
                          connected to whichever peer it contacts most often
                          across the full trace (computed once at load time).
    """

    def __init__(
        self,
        trace_path: str,
        num_nodes: int,
        ensure_connected: bool = True,
    ):
        self.num_nodes       = num_nodes
        self.ensure_connected = ensure_connected
        self._rounds: List[Dict[int, List[int]]] = []
        self._fallback_peer: Dict[int, int] = {}   # node → most-frequent peer
        self._load(Path(trace_path))

    # ------------------------------------------------------------------
    # Core accessor — same signature as MobilityModel.neighbors_at()
    # ------------------------------------------------------------------

    def neighbors_at(self, round_idx: int) -> Dict[int, List[int]]:
        """Return adjacency list {node_id: [neighbor_ids]} at round_idx.

        Wraps around if round_idx exceeds the trace length.
        """
        idx = round_idx % len(self._rounds)
        adj = {i: list(nbrs) for i, nbrs in self._rounds[idx].items()}

        if self.ensure_connected:
            for node in range(self.num_nodes):
                if not adj.get(node):
                    peer = self._fallback_peer.get(node)
                    if peer is not None and peer != node:
                        adj[node].append(peer)
                        adj[peer].append(node)

        return adj

    @property
    def total_rounds(self) -> int:
        return len(self._rounds)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Trace file not found: {path}\n"
                "Run  python scripts/prepare_sociopatterns.py  first to download "
                "and preprocess the dataset."
            )

        # First pass: count contacts per (node, peer) pair for fallback computation
        contact_count: Dict[int, Dict[int, int]] = {
            i: {} for i in range(self.num_nodes)
        }
        raw: Dict[int, List[tuple]] = {}   # round → [(i, j), ...]

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                r  = int(row["round"])
                ni = int(row["node_i"])
                nj = int(row["node_j"])
                raw.setdefault(r, []).append((ni, nj))
                contact_count[ni][nj] = contact_count[ni].get(nj, 0) + 1
                contact_count[nj][ni] = contact_count[nj].get(ni, 0) + 1

        # Compute fallback peers (most-frequent contact overall)
        for node in range(self.num_nodes):
            peers = contact_count.get(node, {})
            if peers:
                self._fallback_peer[node] = max(peers, key=lambda p: peers[p])

        # Build per-round adjacency lists (all keys present, empty list if isolated)
        if not raw:
            raise ValueError(f"Trace file is empty: {path}")
        n_rounds = max(raw) + 1
        for r in range(n_rounds):
            adj: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
            for ni, nj in raw.get(r, []):
                adj[ni].append(nj)
                adj[nj].append(ni)
            self._rounds.append(adj)
