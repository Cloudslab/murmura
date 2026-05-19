"""Standalone node process for distributed Murmura.

Each node runs as an independent OS process owning its model exclusively.
Round timing is governed by the system wall clock — the node sleeps until
its pre-agreed round-start time and then proceeds without waiting for any
signal from a coordinator.

Socket layout per node i:
    PULL  (bind  node_i endpoint)    — receives MODEL_STATE from neighbours
    PUSH  (connect node_j endpoint)  — sends   MODEL_STATE to each neighbour j
    PUSH  (connect monitor endpoint) — sends   METRICS to the passive monitor

Round protocol per round k:
    1. Sleep until  t_start + k * round_duration_s
    2. Local training  (honest nodes only)
    3. PUSH own (possibly attacked) model state to each current neighbour
    4. PULL model states from neighbours, deadline = round-window end
    5. Aggregate with received states (skip missing neighbours gracefully)
    6. Evaluate
    7. PUSH METRICS {accuracy, loss, round_idx, …} to monitor

Neighbour set:
    Default (static topology): read from config at startup.
    DMTT override: subclass overrides _get_current_neighbors(round_idx) to
    return the dynamically discovered trusted-feasible set F_i^t.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import zmq
from torch.utils.data import DataLoader

from murmura.config.loader import load_config
from murmura.config.schema import Config
from murmura.core.node import Node
from murmura.core.types import ModelState
from murmura.distributed.endpoints import Endpoints
from murmura.distributed.messaging import (
    MsgType,
    decode,
    encode,
    pack_obj,
    pack_state,
    unpack_obj,
    unpack_state,
)
from murmura.utils.factories import (
    build_aggregator_factory,
    build_attack,
    build_criterion,
    build_dataset_adapter,
    build_model_factory,
)
from murmura.utils.seed import set_seed


class NodeProcess:
    """Runs the FL protocol for a single node inside its own OS process."""

    def __init__(
        self,
        node_id: int,
        config: Config,
        endpoints: Endpoints,
        t_start: float,
        mobility=None,  # Optional[MobilityModel] — avoids circular import at class level
    ):
        self.node_id   = node_id
        self.config    = config
        self.endpoints = endpoints
        self.t_start   = t_start          # absolute wall-clock time for round 0
        self.mobility  = mobility         # when set, use G^t per round instead of static topology

        # ZMQ state — populated in run()
        self._ctx:          Optional[zmq.Context]           = None
        self._pull:         Optional[zmq.Socket]            = None
        self._push_socks:   Dict[int, zmq.Socket]           = {}
        self._monitor_push: Optional[zmq.Socket]            = None

        # Static neighbour set resolved from topology config
        self._static_neighbors: Optional[List[int]]         = None

    @classmethod
    def from_config_path(
        cls,
        node_id: int,
        config_path: str,
        endpoints: Endpoints,
        t_start: float,
    ) -> "NodeProcess":
        config = load_config(Path(config_path))
        mobility = None
        if config.mobility is not None:
            from murmura.utils.factories import build_mobility_model
            mobility = build_mobility_model(config)
        return cls(
            node_id=node_id,
            config=config,
            endpoints=endpoints,
            t_start=t_start,
            mobility=mobility,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main entry point — runs all training rounds then exits."""
        set_seed(self.config.experiment.seed + self.node_id)

        device = self._resolve_device()
        node   = self._build_node(device)
        attack = build_attack(self.config)

        self._ctx = zmq.Context()
        try:
            self._setup_sockets()
            self._run_all_rounds(node, attack)
        finally:
            self._teardown_sockets()

    # ------------------------------------------------------------------
    # Socket lifecycle
    # ------------------------------------------------------------------

    def _setup_sockets(self) -> None:
        ctx = self._ctx

        # PULL — binds so neighbours can push model states to us
        self._pull = ctx.socket(zmq.PULL)
        self._pull.bind(self.endpoints.node_pull_bind(self.node_id))

        # Pre-connect PUSH sockets to all static neighbours.
        # _ensure_push_sock() adds sockets lazily for dynamic neighbours (DMTT).
        for nid in self._get_static_neighbors():
            self._ensure_push_sock(nid)

        # PUSH to monitor (metrics only, fire-and-forget)
        self._monitor_push = ctx.socket(zmq.PUSH)
        self._monitor_push.connect(self.endpoints.monitor_pull_connect())

        # Brief pause to let connections establish before the first round.
        time.sleep(0.1)

    def _ensure_push_sock(self, neighbor_id: int) -> zmq.Socket:
        """Return existing PUSH socket to neighbor_id, creating one if absent."""
        if neighbor_id not in self._push_socks:
            sock = self._ctx.socket(zmq.PUSH)
            sock.connect(self.endpoints.node_pull_connect(neighbor_id))
            self._push_socks[neighbor_id] = sock
        return self._push_socks[neighbor_id]

    def _teardown_sockets(self) -> None:
        for sock in [self._pull, self._monitor_push]:
            if sock is not None:
                sock.close()
        for sock in self._push_socks.values():
            sock.close()
        self._push_socks.clear()
        if self._ctx is not None:
            self._ctx.term()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run_all_rounds(self, node: Node, attack) -> None:
        rounds = self.config.experiment.rounds
        dur    = self.config.distributed.round_duration_s

        for round_idx in range(rounds):
            round_wall_start = self.t_start + round_idx * dur
            round_wall_end   = round_wall_start + dur

            # Sleep until this round's window opens
            now = time.monotonic()
            if round_wall_start > now:
                time.sleep(round_wall_start - now)

            current_neighbors = self._get_current_neighbors(round_idx)
            self._execute_round(
                node=node,
                attack=attack,
                round_idx=round_idx,
                round_wall_end=round_wall_end,
                current_neighbors=current_neighbors,
            )

    def _execute_round(
        self,
        node: Node,
        attack,
        round_idx: int,
        round_wall_end: float,
        current_neighbors: List[int],
    ) -> None:
        is_byzantine = attack is not None and attack.is_compromised(self.node_id)
        local_epochs = self.config.training.local_epochs
        lr           = self.config.training.lr

        # 1. Local training
        if not is_byzantine:
            node.local_train(epochs=local_epochs, lr=lr, round_num=round_idx)

        # Warn if training consumed the whole round window
        if time.monotonic() >= round_wall_end:
            print(
                f"[Node {self.node_id}] WARNING: training for round {round_idx + 1} "
                f"exceeded round_duration_s={self.config.distributed.round_duration_s}s. "
                f"Model exchange will be skipped.",
                flush=True,
            )
            self._push_metrics(node, round_idx, skipped=True)
            return

        # 2. Prepare outgoing model state (attacked if Byzantine)
        state: ModelState = node.get_state()
        if is_byzantine and attack is not None:
            state = attack.apply_attack(
                node_id=self.node_id, model_state=state, round_num=round_idx
            )

        # 3. Push state to every current neighbour
        state_bytes = pack_state(state)
        for nid in current_neighbors:
            self._ensure_push_sock(nid).send_multipart(
                encode(MsgType.MODEL_STATE, self.node_id, state_bytes)
            )

        # 4. Pull states from neighbours within the round window
        neighbor_states = self._collect_neighbor_states(
            expected=current_neighbors,
            round_idx=round_idx,
            deadline=round_wall_end,
        )

        # 5. Aggregate (with however many states arrived)
        if neighbor_states:
            aggregated = node.aggregate_with_neighbors(neighbor_states, round_idx)
            node.apply_aggregated_state(aggregated)

        # 6 + 7. Evaluate and send metrics
        self._push_metrics(node, round_idx)

    def _collect_neighbor_states(
        self,
        expected: List[int],
        round_idx: int,
        deadline: float,
    ) -> Dict[int, ModelState]:
        """Pull MODEL_STATE messages until all expected neighbours respond or deadline."""
        neighbor_states: Dict[int, ModelState] = {}
        expected_set: Set[int] = set(expected)

        while len(neighbor_states) < len(expected_set):
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                missing = expected_set - set(neighbor_states)
                print(
                    f"[Node {self.node_id}] Round {round_idx + 1}: "
                    f"deadline reached, missing states from {sorted(missing)}. "
                    f"Aggregating with {len(neighbor_states)}/{len(expected_set)} neighbours.",
                    flush=True,
                )
                break
            if self._pull.poll(timeout=max(50, remaining_ms)):
                frames = self._pull.recv_multipart()
                msg_type, sender_id, payload = decode(frames)
                if msg_type == MsgType.MODEL_STATE and sender_id in expected_set:
                    neighbor_states[sender_id] = unpack_state(payload)

        return neighbor_states

    def _push_metrics(self, node: Node, round_idx: int, skipped: bool = False) -> None:
        if skipped:
            metrics: Dict = {"accuracy": 0.0, "loss": 0.0, "skipped": True}
        else:
            metrics = node.evaluate()
        metrics["round_idx"] = round_idx
        self._monitor_push.send_multipart(
            encode(MsgType.METRICS, self.node_id, pack_obj(metrics))
        )

    # ------------------------------------------------------------------
    # Neighbour resolution — override in DMTTNodeProcess for dynamic topology
    # ------------------------------------------------------------------

    def _get_static_neighbors(self) -> List[int]:
        """Resolve and cache pre-connect targets for the PULL-socket setup.

        When a mobility model is present the topology is dynamic, so we
        pre-connect to all other nodes so that lazy PUSH sockets can send
        without a per-round connection delay.
        """
        if self._static_neighbors is None:
            if self.mobility is not None:
                n = self.config.topology.num_nodes
                self._static_neighbors = [i for i in range(n) if i != self.node_id]
            else:
                from murmura.topology import create_topology
                topo = create_topology(
                    topology_type=self.config.topology.type,
                    num_nodes=self.config.topology.num_nodes,
                    p=self.config.topology.p,
                    k=self.config.topology.k,
                    seed=self.config.topology.seed,
                )
                self._static_neighbors = topo.neighbors[self.node_id]
        return self._static_neighbors

    def _get_current_neighbors(self, round_idx: int) -> List[int]:
        """Return the neighbour list for round_idx.

        When config.mobility is set, returns G^t neighbours from the mobility
        model.  DMTTNodeProcess further overrides this to apply TopB selection.
        """
        if self.mobility is not None:
            return self.mobility.neighbors_at(round_idx).get(self.node_id, [])
        return self._get_static_neighbors()

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        from murmura.utils.device import get_device
        return get_device()

    def _build_node(self, device: torch.device) -> Node:
        cfg = self.config

        dataset_adapter   = build_dataset_adapter(cfg)
        model_factory     = build_model_factory(cfg)
        aggregator_factory = build_aggregator_factory(cfg, model_factory, device)
        criterion, evidential = build_criterion(cfg)

        model        = model_factory().to(device)
        train_dataset = dataset_adapter.get_client_data(self.node_id)

        n_samples  = len(train_dataset)
        batch_size = min(cfg.training.batch_size, max(2, n_samples))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            drop_last=n_samples > batch_size,
        )
        test_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
        )

        return Node(
            node_id=self.node_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            aggregator=aggregator_factory(self.node_id),
            device=device,
            criterion=criterion,
            evidential=evidential,
        )
