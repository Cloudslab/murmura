"""Coordinator process for distributed Murmura.

The coordinator is the only process that crosses all node boundaries, but only
for control flow and metrics — it never holds or touches model weights.

Socket layout:
    PUB socket (bind)  — broadcasts ROUND_START and SHUTDOWN to all nodes
    PULL socket (bind) — receives READY signals and METRICS from all nodes

Round protocol:
    1. Broadcast ROUND_START with {round, local_epochs, lr, neighbors}
    2. Block on PULL until all N METRICS messages arrive (with timeout)
    3. Aggregate metrics into history dict
    4. Repeat for all rounds, then broadcast SHUTDOWN
"""

import time
from typing import Any, Dict, List, Set

import numpy as np
import zmq

from murmura.distributed.endpoints import Endpoints
from murmura.distributed.messaging import (
    COORDINATOR_ID,
    MsgType,
    decode,
    encode,
    pack_obj,
    unpack_obj,
)


class Coordinator:
    """Orchestrates training rounds without ever holding node model state."""

    def __init__(
        self,
        num_nodes: int,
        endpoints: Endpoints,
        rounds: int,
        local_epochs: int,
        lr: float,
        topology_neighbors: Dict[int, List[int]],
        compromised_nodes: Set[int],
        startup_timeout_s: float = 30.0,
        round_timeout_s: float = 300.0,
        verbose: bool = False,
    ):
        self.num_nodes = num_nodes
        self.endpoints = endpoints
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.topology_neighbors = topology_neighbors
        self.compromised_nodes = compromised_nodes
        self.startup_timeout_s = startup_timeout_s
        self.round_timeout_s = round_timeout_s
        self.verbose = verbose

        self.history: Dict[str, List[Any]] = {
            "round": [],
            "mean_accuracy": [],
            "std_accuracy": [],
            "mean_loss": [],
            "honest_accuracy": [],
            "compromised_accuracy": [],
            "mean_vacuity": [],
            "mean_entropy": [],
            "mean_strength": [],
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, List[Any]]:
        """Block until all training rounds complete. Returns history dict."""
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pull = ctx.socket(zmq.PULL)

        try:
            pub.bind(self.endpoints.coordinator_pub_bind())
            pull.bind(self.endpoints.coordinator_pull_bind())

            # Give the PUB socket time to bind before any node subscribes.
            # ZMQ's slow-joiner problem: subscribers miss messages sent before
            # they connect — so the coordinator must bind first, then wait.
            time.sleep(0.5)

            self._log(f"Waiting for {self.num_nodes} nodes to signal READY …")
            self._wait_for_ready(pull)
            self._log("All nodes ready. Starting training.")

            # Extra pause after all READY signals: ensures every node's SUB
            # socket has fully subscribed before the first ROUND_START goes out.
            time.sleep(0.3)

            for round_idx in range(self.rounds):
                self._run_round(pub, pull, round_idx)

            pub.send_multipart(encode(MsgType.SHUTDOWN, COORDINATOR_ID, pack_obj({})))
            time.sleep(0.2)  # let SHUTDOWN propagate before closing

        finally:
            pub.close()
            pull.close()
            ctx.term()

        return self.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_ready(self, pull: zmq.Socket) -> None:
        ready: Set[int] = set()
        deadline = time.monotonic() + self.startup_timeout_s

        while len(ready) < self.num_nodes:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                missing = set(range(self.num_nodes)) - ready
                raise TimeoutError(
                    f"Startup timeout ({self.startup_timeout_s}s): "
                    f"nodes {sorted(missing)} did not signal READY"
                )
            if pull.poll(timeout=min(1000, remaining_ms)):
                frames = pull.recv_multipart()
                msg_type, sender_id, _ = decode(frames)
                if msg_type == MsgType.READY and sender_id not in ready:
                    ready.add(sender_id)
                    self._log(f"  Node {sender_id} ready ({len(ready)}/{self.num_nodes})")

    def _run_round(self, pub: zmq.Socket, pull: zmq.Socket, round_idx: int) -> None:
        round_num = round_idx + 1
        self._log(f"\n=== Round {round_num}/{self.rounds} ===")

        payload = pack_obj({
            "round": round_idx,
            "local_epochs": self.local_epochs,
            "lr": self.lr,
            "neighbors": self.topology_neighbors,
        })
        pub.send_multipart(encode(MsgType.ROUND_START, COORDINATOR_ID, payload))

        metrics_by_node: Dict[int, Dict[str, Any]] = {}
        deadline = time.monotonic() + self.round_timeout_s

        while len(metrics_by_node) < self.num_nodes:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                missing = set(range(self.num_nodes)) - set(metrics_by_node)
                raise TimeoutError(
                    f"Round {round_num} timeout ({self.round_timeout_s}s): "
                    f"no metrics from nodes {sorted(missing)}"
                )
            if pull.poll(timeout=max(100, remaining_ms)):
                frames = pull.recv_multipart()
                msg_type, sender_id, payload = decode(frames)
                if msg_type == MsgType.METRICS:
                    metrics_by_node[sender_id] = unpack_obj(payload)

        self._record(round_num, metrics_by_node)

    def _record(self, round_num: int, metrics: Dict[int, Dict[str, Any]]) -> None:
        accs = [m["accuracy"] for m in metrics.values()]
        losses = [m["loss"] for m in metrics.values()]

        honest_accs = [
            m["accuracy"] for nid, m in metrics.items() if nid not in self.compromised_nodes
        ]
        comp_accs = [
            m["accuracy"] for nid, m in metrics.items() if nid in self.compromised_nodes
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

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        self._log(f"Round {round_num}: Mean Acc = {mean_acc:.4f} ± {std_acc:.4f}")
        if honest_accs and comp_accs:
            self._log(
                f"  Honest: {np.mean(honest_accs):.4f}  "
                f"Compromised: {np.mean(comp_accs):.4f}"
            )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Coordinator] {msg}", flush=True)
