"""Base attack protocol for Byzantine node simulation."""

from typing import Protocol, Set, runtime_checkable
from murmura.core.types import ModelState


@runtime_checkable
class Attack(Protocol):
    """Protocol for Byzantine attacks.

    Attacks define how compromised nodes modify their model updates
    to disrupt the learning process.
    """

    def is_compromised(self, node_id: int) -> bool:
        """Check if a node is compromised.

        Args:
            node_id: Node index

        Returns:
            True if node is compromised
        """
        ...

    def get_compromised_nodes(self) -> Set[int]:
        """Get set of compromised node IDs.

        Returns:
            Set of compromised node indices
        """
        ...

    def apply_attack(
        self,
        node_id: int,
        model_state: ModelState,
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Apply attack to a model state.

        Args:
            node_id: Compromised node ID
            model_state: Original model state
            round_num: Current training round
            **kwargs: Additional context

        Returns:
            Attacked model state
        """
        ...
