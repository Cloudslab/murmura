"""Topology-liar Byzantine attack for DMTT experiments.

Byzantine nodes under this attack send *false* TOPO_CLAIM messages that
misreport their neighbourhood — specifically, they claim that other Byzantine
nodes are within communication range even when they are not.  This attempts to
inject Byzantine models into honest nodes' collaboration sets.

The attack also optionally poisons the model state (via a wrapped model attack)
so the combined effect matches the threat model in the DMTT paper.

Protocol hook: DMTTNodeProcess calls attack.get_false_claims() before
sending TOPO_CLAIM messages.  Non-DMTT node processes do not call this.
"""

import random
from typing import List, Optional, Set

from murmura.core.types import ModelState


class TopologyLiarAttack:
    """Byzantine nodes lie about topology to corrupt honest nodes' collaboration sets.

    Args:
        num_nodes:         Total number of nodes in the network.
        attack_percentage: Fraction of nodes to compromise.
        seed:              RNG seed for selecting compromised nodes.
        model_attack:      Optional underlying model-state attack applied on top.
    """

    def __init__(
        self,
        num_nodes: int,
        attack_percentage: float,
        seed: int = 42,
        model_attack=None,
    ):
        self.num_nodes  = num_nodes
        self._model_attack = model_attack

        n_compromised  = max(1, int(num_nodes * attack_percentage))
        rng            = random.Random(seed)
        self._compromised: Set[int] = set(
            rng.sample(range(num_nodes), min(n_compromised, num_nodes))
        )

    # ------------------------------------------------------------------
    # Attack protocol (satisfies Attack protocol)
    # ------------------------------------------------------------------

    def is_compromised(self, node_id: int) -> bool:
        return node_id in self._compromised

    def get_compromised_nodes(self) -> Set[int]:
        return set(self._compromised)

    def apply_attack(
        self,
        node_id: int,
        model_state: ModelState,
        round_num: int,
        **kwargs,
    ) -> ModelState:
        """Delegate model-state poisoning to the wrapped attack (if any)."""
        if self._model_attack is not None:
            return self._model_attack.apply_attack(
                node_id=node_id,
                model_state=model_state,
                round_num=round_num,
                **kwargs,
            )
        return model_state  # topology liar: lie about topology, not model

    # ------------------------------------------------------------------
    # DMTT-specific hook
    # ------------------------------------------------------------------

    def get_false_claims(
        self,
        node_id: int,
        true_neighbors: List[int],
        round_num: int,
    ) -> List[int]:
        """Return a falsified neighbour list for a Byzantine node's TOPO_CLAIM.

        Strategy: claim all other Byzantine nodes as neighbours (whether or
        not they are actually in range) while retaining the true neighbours so
        the claim does not look completely implausible.

        Args:
            node_id:        The Byzantine node sending the claim.
            true_neighbors: Actual neighbours per the mobility model.
            round_num:      Current round (unused; available for future strategies).

        Returns:
            Falsified neighbour list to embed in the TOPO_CLAIM payload.
        """
        false_set = set(true_neighbors)
        for c in self._compromised:
            if c != node_id:
                false_set.add(c)
        return sorted(false_set)
