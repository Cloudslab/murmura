"""Passive metrics monitor for distributed Murmura.

The monitor replaces the old Coordinator.  It has exactly one socket — a PULL
that receives METRICS messages from nodes — and never sends anything.  Round
synchronisation is the clock's job; the monitor's only job is to aggregate
per-round accuracy and loss numbers for the training history.

Because the monitor is passive it introduces no single point of failure for
the learning protocol.  If the monitor crashes, nodes continue training
correctly; they just cannot report metrics to the caller.

Socket layout:
    PULL (bind monitor_pull endpoint) — receives METRICS from all nodes

Expected message rate:  num_nodes messages per round, rounds total.
"""

import time
from typing import Any, Dict, List, Set

import numpy as np
import zmq

from murmura.distributed.endpoints import Endpoints
from murmura.distributed.messaging import MsgType, decode, unpack_obj


class Monitor:
    """Collects per-round metrics from node processes without influencing them."""

    def __init__(
        self,
        num_nodes: int,
        endpoints: Endpoints,
        rounds: int,
        t_start: float,
        round_duration_s: float,
        compromised_nodes: Set[int],
        verbose: bool = False,
    ):
        self.num_nodes        = num_nodes
        self.endpoints        = endpoints
        self.rounds           = rounds
        self.t_start          = t_start
        self.round_duration_s = round_duration_s
        self.compromised_nodes = compromised_nodes
        self.verbose          = verbose

        self.history: Dict[str, List[Any]] = {
            "round":               [],
            "mean_accuracy":       [],
            "std_accuracy":        [],
            "mean_loss":           [],
            "honest_accuracy":     [],
            "compromised_accuracy":[],
            "mean_vacuity":        [],
            "mean_entropy":        [],
            "mean_strength":       [],
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, List[Any]]:
        """Block until metrics for all rounds have been collected. Returns history."""
        ctx  = zmq.Context()
        pull = ctx.socket(zmq.PULL)
        try:
            pull.bind(self.endpoints.monitor_pull_bind())
            self._collect(pull)
        finally:
            pull.close()
            ctx.term()
        return self.history

    # ------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------

    def _collect(self, pull: zmq.Socket) -> None:
        # Buffer all incoming metrics keyed by (round_idx, node_id).
        # Nodes run on wall-clock time so metrics for different rounds can
        # arrive slightly out of order near a round boundary.
        pending: Dict[int, Dict[int, Dict[str, Any]]] = {}

        total_expected = self.num_nodes * self.rounds
        total_received = 0

        # Give a generous deadline: last round end + 2× round_duration for stragglers.
        hard_deadline = (
            self.t_start
            + self.rounds * self.round_duration_s
            + 2 * self.round_duration_s
        )

        while total_received < total_expected:
            remaining_ms = int((hard_deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                self._log("Deadline reached; some metrics may be missing.")
                break
            if not pull.poll(timeout=max(200, min(remaining_ms, 2000))):
                continue

            frames = pull.recv_multipart()
            msg_type, node_id, payload = decode(frames)
            if msg_type != MsgType.METRICS:
                continue

            m = unpack_obj(payload)
            round_idx: int = m.pop("round_idx")

            pending.setdefault(round_idx, {})[node_id] = m
            total_received += 1

            # Flush any complete rounds in order
            while True:
                next_round = len(self.history["round"])   # 0-based index
                if next_round not in pending:
                    break
                if len(pending[next_round]) < self.num_nodes:
                    break   # wait — more nodes may still report this round
                self._record(next_round + 1, pending.pop(next_round))

        # Flush whatever remains (partial rounds from stragglers)
        for round_idx in sorted(pending):
            if pending[round_idx]:
                self._record(round_idx + 1, pending[round_idx])

    # ------------------------------------------------------------------
    # History recording
    # ------------------------------------------------------------------

    def _record(self, round_num: int, metrics: Dict[int, Dict[str, Any]]) -> None:
        accs   = [m["accuracy"] for m in metrics.values()]
        losses = [m["loss"]     for m in metrics.values()]

        honest_accs = [
            m["accuracy"] for nid, m in metrics.items()
            if nid not in self.compromised_nodes
        ]
        comp_accs = [
            m["accuracy"] for nid, m in metrics.items()
            if nid in self.compromised_nodes
        ]
        vacuities = [m["vacuity"] for m in metrics.values() if "vacuity" in m]
        entropies = [m["entropy"] for m in metrics.values() if "entropy" in m]
        strengths = [m["strength"] for m in metrics.values() if "strength" in m]

        self.history["round"].append(round_num)
        self.history["mean_accuracy"].append(float(np.mean(accs)))
        self.history["std_accuracy"].append(float(np.std(accs)))
        self.history["mean_loss"].append(float(np.mean(losses)))
        if honest_accs:
            self.history["honest_accuracy"].append(float(np.mean(honest_accs)))
        if comp_accs:
            self.history["compromised_accuracy"].append(float(np.mean(comp_accs)))
        if vacuities:
            self.history["mean_vacuity"].append(float(np.mean(vacuities)))
            self.history["mean_entropy"].append(float(np.mean(entropies)))
            self.history["mean_strength"].append(float(np.mean(strengths)))

        self._log(
            f"Round {round_num} ({len(metrics)}/{self.num_nodes} nodes): "
            f"acc={np.mean(accs):.4f} ± {np.std(accs):.4f}"
        )
        if honest_accs and comp_accs:
            self._log(
                f"  Honest: {np.mean(honest_accs):.4f}  "
                f"Compromised: {np.mean(comp_accs):.4f}"
            )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Monitor] {msg}", flush=True)
