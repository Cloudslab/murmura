"""Core type definitions and protocols for Murmura."""

from typing import Protocol, Dict, List, runtime_checkable
import torch


# Type aliases
ModelState = Dict[str, torch.Tensor]
"""Dictionary mapping parameter names to tensors"""

DataPartition = List[int]
"""List of indices representing a data partition"""


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for PyTorch models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        ...

    def parameters(self):
        """Return model parameters."""
        ...

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dictionary."""
        ...

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load state dictionary."""
        ...

    def train(self, mode: bool = True) -> None:
        """Set training mode."""
        ...

    def eval(self) -> None:
        """Set evaluation mode."""
        ...

    def to(self, device: torch.device):
        """Move model to device."""
        ...
