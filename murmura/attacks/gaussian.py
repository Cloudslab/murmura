"""Gaussian noise attack."""

from typing import Set
import random
import torch

from murmura.core.types import ModelState


class GaussianAttack:
    """Gaussian noise attack.

    Compromised nodes add Gaussian noise to their model parameters,
    disrupting the learning process.
    """

    def __init__(
        self,
        num_nodes: int,
        attack_percentage: float,
        noise_std: float = 10.0,
        seed: int = 42
    ):
        """Initialize Gaussian attack.

        Args:
            num_nodes: Total number of nodes
            attack_percentage: Fraction of nodes to compromise
            noise_std: Standard deviation of Gaussian noise
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.attack_percentage = attack_percentage
        self.noise_std = noise_std

        # Select compromised nodes
        num_compromised = int(num_nodes * attack_percentage)
        if num_compromised == 0 and attack_percentage > 0:
            num_compromised = 1

        random.seed(seed)
        self.compromised_nodes: Set[int] = set(
            random.sample(range(num_nodes), min(num_compromised, num_nodes))
        )

        print(f"Gaussian Attack: Compromised {len(self.compromised_nodes)}/{num_nodes} nodes")
        print(f"  Compromised nodes: {sorted(self.compromised_nodes)}")
        print(f"  Noise std: {noise_std}")

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
        """Apply Gaussian noise to model parameters.

        Args:
            node_id: Compromised node ID
            model_state: Original model state
            round_num: Current round
            **kwargs: Additional context

        Returns:
            Model state with Gaussian noise added
        """
        if not self.is_compromised(node_id):
            return model_state

        attacked_state = {}
        for key, param in model_state.items():
            # Generate Gaussian noise
            if param.dtype in [torch.long, torch.int, torch.int32, torch.int64, torch.int8, torch.int16]:
                # For integer tensors, create float noise
                noise = torch.randn(param.shape, device=param.device, dtype=torch.float32) * self.noise_std
            else:
                noise = torch.randn_like(param) * self.noise_std

            # Add noise to parameters
            if param.dtype == noise.dtype:
                attacked_state[key] = param + noise
            else:
                attacked_state[key] = param + noise.to(param.dtype)

        return attacked_state
