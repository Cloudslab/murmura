"""DMTTNodeProcess — wall-clock node with dynamic topology and trust.

Extends NodeProcess with:
  1. Dynamic neighbour discovery via MobilityModel (G^t per round).
  2. TOPO_CLAIM message exchange: each node broadcasts its claimed local
     neighbourhood; honest nodes report truthfully, topology liars falsify it.
  3. Beta-evidence trust update (Algorithm 4): confirmations from j raise
     α_ij; contradictions raise β_ij.
  4. Collaboration scoring and TopB selection (Algorithms 1 / Section 4.4):
     q_ij = λ1·s_model + λ2·T_topo + λ3·ĉ - λ4·c_comm
     C_i^{t+1} = TopB_j q_ij^t

Topology-claim verification:
  Because the mobility model is fully deterministic from a shared seed, every
  node process can independently compute G^t.  When node j sends a TOPO_CLAIM
  saying its neighbours at round t are {k₁, k₂, …}, this node verifies each
  claim against its local mobility model:
    d += 1 (positive evidence) for each claim that matches reality
    x += 1 (negative evidence) for each claim that contradicts reality

Socket layout (inherited from NodeProcess plus TOPO_CLAIM):
  PULL    (bind)   — receives MODEL_STATE *and* TOPO_CLAIM from neighbours
  PUSH×N  (connect) — sends MODEL_STATE *and* TOPO_CLAIM to each collaborator
  PUSH    (monitor) — sends METRICS to the passive monitor
"""

import copy
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from murmura.config.loader import load_config
from murmura.config.schema import Config, DMTTConfig
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
from murmura.distributed.node_process import NodeProcess
from murmura.dmtt.state import DMTTNodeState
from murmura.topology.dynamic import MobilityModel


class DMTTNodeProcess(NodeProcess):
    """NodeProcess extended with DMTT dynamic topology and trust protocol."""

    def __init__(
        self,
        node_id: int,
        config: Config,
        endpoints: Endpoints,
        t_start: float,
        mobility: MobilityModel,
    ):
        super().__init__(node_id, config, endpoints, t_start)
        self.mobility     = mobility
        self.dmtt_cfg: DMTTConfig = config.dmtt  # type: ignore[assignment]
        self._dmtt:       Optional[DMTTNodeState] = None
        # C_i^t — collaborators selected at end of previous round.
        # Initialised to None; first round uses all G^0 neighbours.
        self._collaborators: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config_path(
        cls,
        node_id: int,
        config_path: str,
        endpoints: Endpoints,
        t_start: float,
    ) -> "DMTTNodeProcess":
        config = load_config(Path(config_path))
        if config.mobility is None:
            raise ValueError("DMTTNodeProcess requires config.mobility to be set.")
        if config.dmtt is None:
            raise ValueError("DMTTNodeProcess requires config.dmtt to be set.")
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
    # Socket setup — pre-connect to all nodes (topology is dynamic)
    # ------------------------------------------------------------------

    def _get_static_neighbors(self) -> List[int]:
        """Pre-connect PUSH sockets to all peers; actual sends are gated later."""
        return [i for i in range(self.config.topology.num_nodes) if i != self.node_id]

    # ------------------------------------------------------------------
    # Neighbour resolution — DMTT override
    # ------------------------------------------------------------------

    def _get_current_neighbors(self, round_idx: int) -> List[int]:
        """C_i^{round_idx}: collaborators selected at end of round round_idx-1.

        Round 0 falls back to all direct neighbours in G^0.
        """
        if round_idx == 0 or self._collaborators is None:
            return self.mobility.neighbors_at(0).get(self.node_id, [])
        return self._collaborators

    # ------------------------------------------------------------------
    # Round execution — extended with TOPO_CLAIM exchange
    # ------------------------------------------------------------------

    def run(self) -> None:
        from murmura.utils.seed import set_seed
        set_seed(self.config.experiment.seed + self.node_id)

        device = self._resolve_device()
        node   = self._build_node(device)

        # Build a light-weight model factory for evaluating neighbour models.
        from murmura.utils.factories import build_model_factory
        self._model_factory = build_model_factory(self.config)
        self._eval_device   = device

        attack = None
        from murmura.utils.factories import build_attack
        attack = build_attack(self.config)

        self._dmtt = DMTTNodeState(self.node_id, self.dmtt_cfg)

        import zmq
        self._ctx = zmq.Context()
        try:
            self._setup_sockets()
            self._run_all_rounds(node, attack)
        finally:
            self._teardown_sockets()

    def _execute_round(
        self,
        node,
        attack,
        round_idx: int,
        round_wall_end: float,
        current_neighbors: List[int],
    ) -> None:
        is_byzantine = attack is not None and attack.is_compromised(self.node_id)
        cfg          = self.config

        # 1. Local training
        if not is_byzantine:
            node.local_train(
                epochs=cfg.training.local_epochs,
                lr=cfg.training.lr,
                round_num=round_idx,
            )

        if time.monotonic() >= round_wall_end:
            print(
                f"[DMTT Node {self.node_id}] WARNING: round {round_idx + 1} training "
                f"exceeded budget; skipping exchange.",
                flush=True,
            )
            self._push_metrics(node, round_idx, skipped=True)
            self._update_collaborators_no_data(round_idx)
            return

        # 2. Model state (possibly poisoned)
        state = node.get_state()
        if is_byzantine and attack is not None:
            state = attack.apply_attack(
                node_id=self.node_id, model_state=state, round_num=round_idx
            )
        state_bytes = pack_state(state)

        # 3. Topology claim (honest or falsified)
        true_neighbors = self.mobility.neighbors_at(round_idx).get(self.node_id, [])
        if is_byzantine and attack is not None and hasattr(attack, "get_false_claims"):
            claimed_neighbors = attack.get_false_claims(
                node_id=self.node_id,
                true_neighbors=true_neighbors,
                round_num=round_idx,
            )
        else:
            claimed_neighbors = list(true_neighbors)
        claim_bytes = pack_obj({
            "round_idx":   round_idx,
            "neighbors":   claimed_neighbors,
        })

        # 4. Push MODEL_STATE + TOPO_CLAIM to every current collaborator
        for nid in current_neighbors:
            sock = self._ensure_push_sock(nid)
            sock.send_multipart(encode(MsgType.MODEL_STATE, self.node_id, state_bytes))
            sock.send_multipart(encode(MsgType.TOPO_CLAIM,  self.node_id, claim_bytes))

        # 5. Collect MODEL_STATE + TOPO_CLAIM from expected neighbours
        neighbor_states, topo_claims = self._collect_dmtt_messages(
            expected=current_neighbors,
            round_idx=round_idx,
            deadline=round_wall_end,
        )

        # 6. Update link reliability EMA
        assert self._dmtt is not None
        for nid in current_neighbors:
            self._dmtt.update_link_reliability(nid, nid in neighbor_states)

        # 7. Evaluate received models → model compatibility scores
        model_scores: Dict[int, float] = {}
        if neighbor_states:
            model_scores = self._score_neighbor_models(
                neighbor_states, node, round_idx
            )

        # 8. Process topology claims → update Beta trust
        self._process_topo_claims(topo_claims, round_idx)

        # 9. Aggregate with received states
        if neighbor_states:
            aggregated = node.aggregate_with_neighbors(neighbor_states, round_idx)
            node.apply_aggregated_state(aggregated)

        # 10. Update collaborators for next round (TopB selection)
        all_direct = self.mobility.neighbors_at(round_idx).get(self.node_id, [])
        self._collaborators = self._dmtt.top_b(
            candidates=all_direct,
            model_scores=model_scores,
            B=self.dmtt_cfg.budget_B,
        )
        if cfg.experiment.verbose:
            print(
                f"[DMTT Node {self.node_id}] Round {round_idx + 1}: "
                f"collaborators → {self._collaborators}",
                flush=True,
            )

        # 11. Evaluate + push metrics
        self._push_metrics(node, round_idx)

    # ------------------------------------------------------------------
    # Combined MODEL_STATE + TOPO_CLAIM collection
    # ------------------------------------------------------------------

    def _collect_dmtt_messages(
        self,
        expected: List[int],
        round_idx: int,
        deadline: float,
    ) -> Tuple[Dict[int, Any], Dict[int, dict]]:
        """Pull MODEL_STATE and TOPO_CLAIM for all expected senders until deadline."""
        neighbor_states: Dict[int, Any]  = {}
        topo_claims:     Dict[int, dict] = {}
        expected_set: Set[int]           = set(expected)

        # We wait until we have both types from every expected sender, or deadline.
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                missing_states  = expected_set - set(neighbor_states)
                missing_claims  = expected_set - set(topo_claims)
                if missing_states or missing_claims:
                    print(
                        f"[DMTT Node {self.node_id}] Round {round_idx + 1}: deadline — "
                        f"missing states from {sorted(missing_states)}, "
                        f"claims from {sorted(missing_claims)}.",
                        flush=True,
                    )
                break

            if not self._pull.poll(timeout=max(50, remaining_ms)):  # type: ignore[union-attr]
                continue

            frames   = self._pull.recv_multipart()  # type: ignore[union-attr]
            msg_type, sender_id, payload = decode(frames)

            if sender_id not in expected_set:
                continue

            if msg_type == MsgType.MODEL_STATE and sender_id not in neighbor_states:
                neighbor_states[sender_id] = unpack_state(payload)
            elif msg_type == MsgType.TOPO_CLAIM and sender_id not in topo_claims:
                topo_claims[sender_id] = unpack_obj(payload)

            # Stop early if every expected sender has provided both messages.
            if (
                expected_set <= set(neighbor_states)
                and expected_set <= set(topo_claims)
            ):
                break

        return neighbor_states, topo_claims

    # ------------------------------------------------------------------
    # Model compatibility scoring
    # ------------------------------------------------------------------

    def _score_neighbor_models(
        self,
        neighbor_states: Dict[int, Any],
        node,
        round_idx: int,
    ) -> Dict[int, float]:
        """Evaluate each neighbour's model on this node's test data → s_ij^model."""
        scores: Dict[int, float] = {}
        if node.test_loader is None:
            return {j: 0.5 for j in neighbor_states}

        for j, state in neighbor_states.items():
            try:
                # Instantiate a fresh model, load j's weights, evaluate on local test data.
                foreign_model = self._model_factory().to(self._eval_device)
                foreign_model.load_state_dict(
                    {k: v.to(self._eval_device) for k, v in state.items()}
                )
                a_ij, u_bar = self._evaluate_foreign(foreign_model, node)
                scores[j] = self._dmtt.model_score(a_ij, u_bar)  # type: ignore[union-attr]
            except Exception:
                scores[j] = 0.0
        return scores

    def _evaluate_foreign(self, model: nn.Module, node) -> Tuple[float, float]:
        """Return (accuracy, mean_uncertainty) of a foreign model on local test data."""
        model.eval()
        correct = 0
        total   = 0
        u_total = 0.0

        with torch.no_grad():
            for inputs, targets in node.test_loader:
                inputs  = inputs.to(self._eval_device)
                targets = targets.to(self._eval_device)
                outputs = model(inputs)

                if node.evidential:
                    # Dirichlet output: alpha → expected probs
                    S    = outputs.sum(dim=-1, keepdim=True)
                    probs = outputs / S
                    preds = outputs.argmax(dim=-1)
                    K     = outputs.shape[-1]
                    vacuity = (K / S.squeeze(-1)).mean().item()
                    u_total += vacuity * targets.size(0)
                else:
                    probs = torch.softmax(outputs, dim=-1)
                    preds = probs.argmax(dim=-1)

                correct += (preds == targets).sum().item()
                total   += targets.size(0)

        accuracy = correct / total if total > 0 else 0.0
        u_bar    = u_total / total if total > 0 else 0.0
        return accuracy, u_bar

    # ------------------------------------------------------------------
    # Topology claim processing → Beta trust update (Algorithms 2 / 4)
    # ------------------------------------------------------------------

    def _process_topo_claims(
        self,
        topo_claims: Dict[int, dict],
        round_idx: int,
    ) -> None:
        """Update Beta trust for each sender based on claim accuracy.

        For each node j that sent a TOPO_CLAIM about its neighbourhood:
          d += 1 for each claimed neighbour that actually appears in G^t
          x += 1 for each claimed neighbour that does NOT appear in G^t

        We verify against our local (ground-truth) mobility model.
        """
        assert self._dmtt is not None
        for j, claim in topo_claims.items():
            claimed: List[int] = claim.get("neighbors", [])
            true_j_neighbors   = set(
                self.mobility.neighbors_at(round_idx).get(j, [])
            )
            d = 0.0
            x = 0.0
            for u in claimed:
                if u in true_j_neighbors:
                    d += 1.0
                else:
                    x += 1.0
            self._dmtt.update_trust(j, d=d, x=x)

    # ------------------------------------------------------------------
    # Fallback collaborator update when a round is skipped
    # ------------------------------------------------------------------

    def _update_collaborators_no_data(self, round_idx: int) -> None:
        """Retain previous collaborators (or G^t direct neighbours) on skip."""
        if self._collaborators is None:
            self._collaborators = self.mobility.neighbors_at(round_idx).get(
                self.node_id, []
            )
