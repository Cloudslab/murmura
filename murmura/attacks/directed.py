"""Directed deviation attack."""

from typing import Set
import random
import torch

from murmura.core.types import ModelState


class DirectedDeviationAttack:
    """Directed deviation attack.

    Compromised nodes scale their model parameters by a negative factor,
    effectively sending opposite gradients to disrupt learning.
    """

    def __init__(
        self,
        num_nodes: int,
        attack_percentage: float,
        lambda_param: float = -5.0,
        seed: int = 42
    ):
        """Initialize directed deviation attack.

        Args:
            num_nodes: Total number of nodes
            attack_percentage: Fraction of nodes to compromise
            lambda_param: Scaling factor (negative for opposite direction)
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.attack_percentage = attack_percentage
        self.lambda_param = lambda_param

        # Select compromised nodes
        num_compromised = int(num_nodes * attack_percentage)
        if num_compromised == 0 and attack_percentage > 0:
            num_compromised = 1

        random.seed(seed)
        self.compromised_nodes: Set[int] = set(
            random.sample(range(num_nodes), min(num_compromised, num_nodes))
        )

        print(f"Directed Deviation Attack: Compromised {len(self.compromised_nodes)}/{num_nodes} nodes")
        print(f"  Compromised nodes: {sorted(self.compromised_nodes)}")
        print(f"  Lambda (scaling factor): {lambda_param}")

    def is_compromised(self, node_id: int) -> bool:
        """Check if a node is compromised."""
        return node_id in self.compromised_nodes

    def get_compromised_nodes(self) -> Set[int]:
        """Get set of compromised node IDs."""
        return self.compromised_nodes

    def apply_attack(
        self,
        node_id: int,
        model_state: ModelState,
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Apply directed deviation by scaling parameters.

        Args:
            node_id: Compromised node ID
            model_state: Original model state
            round_num: Current round
            **kwargs: Additional context

        Returns:
            Model state with scaled parameters
        """
        if not self.is_compromised(node_id):
            return model_state

        attacked_state = {}
        for key, param in model_state.items():
            # Scale parameters by lambda (typically negative)
            attacked_state[key] = param * self.lambda_param

        return attacked_state
