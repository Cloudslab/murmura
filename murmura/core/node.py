"""Individual node in a decentralized learning network."""

from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

from murmura.core.types import ModelState
from murmura.aggregation.base import Aggregator, get_model_state, set_model_state
from murmura.utils.metrics import evaluate_model


class Node:
    """Represents a single node in a decentralized learning network.

    Each node maintains its own model, trains on local data, and participates
    in decentralized aggregation with neighbors.
    """

    def __init__(
        self,
        node_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        aggregator: Optional[Aggregator] = None,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        evidential: bool = False,
    ):
        """Initialize a node.

        Args:
            node_id: Unique identifier for this node
            model: PyTorch model for this node
            train_loader: DataLoader for training data
            test_loader: Optional DataLoader for test data
            aggregator: Aggregation algorithm to use
            device: Device to run computations on
            criterion: Loss function (defaults to CrossEntropyLoss)
            evidential: Whether model outputs Dirichlet parameters for EDL
        """
        self.node_id = node_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.aggregator = aggregator
        self.device = device or torch.device("cpu")
        self.evidential = evidential
        self.current_round = 0  # Track round for EDL annealing (not cumulative epochs)

        # Move model to device
        self.model.to(self.device)

        # Training configuration
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def local_train(self, epochs: int, lr: float = 0.01, round_num: int = 0) -> Dict[str, Any]:
        """Perform local training for specified epochs.

        Args:
            epochs: Number of training epochs
            lr: Learning rate
            round_num: Current FL round (for EDL annealing)

        Returns:
            Dictionary with training statistics
        """
        self.model.train()
        self.model.to(self.device)
        self.current_round = round_num

        optimizer = SGD(self.model.parameters(), lr=lr)
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for inputs, targets in self.train_loader:
                # Skip batches with fewer than 2 samples (BatchNorm requirement)
                if inputs.size(0) < 2:
                    continue

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)

                # Handle evidential loss (needs round for annealing in FL setting)
                if self.evidential and hasattr(self.criterion, 'forward'):
                    # EvidentialLoss expects (alpha, targets, epoch)
                    # In FL, use round_num for annealing, not cumulative local epochs
                    loss = self.criterion(outputs, targets, epoch=self.current_round)
                else:
                    loss = self.criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            "avg_loss": avg_loss,
            "num_batches": num_batches,
            "epochs": epochs
        }

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model on test data.

        Returns:
            Dictionary with evaluation metrics. For evidential models,
            also includes uncertainty metrics (vacuity, entropy).
        """
        if self.test_loader is None:
            return {"accuracy": 0.0, "loss": 0.0, "note": "No test data available"}

        if self.evidential:
            return self._evaluate_evidential()
        else:
            accuracy, loss, correct, total = evaluate_model(
                self.model, self.test_loader, self.device
            )
            return {
                "accuracy": accuracy,
                "loss": loss,
                "correct": correct,
                "total": total
            }

    def _evaluate_evidential(self) -> Dict[str, Any]:
        """Evaluate evidential model with uncertainty metrics.

        Returns:
            Dictionary with accuracy, loss, and uncertainty metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        # Uncertainty accumulators
        total_vacuity = 0.0
        total_entropy = 0.0
        total_strength = 0.0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass - get Dirichlet alpha parameters
                alpha = self.model(inputs)

                # Compute predictions from alpha
                S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength
                probs = alpha / S  # Expected probabilities
                predictions = alpha.argmax(dim=-1)

                # Accuracy
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

                # Compute uncertainty metrics
                K = alpha.shape[-1]
                vacuity = (K / S.squeeze(-1))  # Epistemic uncertainty
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # Aleatoric

                total_vacuity += vacuity.sum().item()
                total_entropy += entropy.sum().item()
                total_strength += S.squeeze(-1).sum().item()

                # Loss (simple MSE-based for evaluation)
                y_onehot = torch.zeros_like(alpha)
                y_onehot.scatter_(1, targets.unsqueeze(1), 1)
                mse = ((y_onehot - probs) ** 2).sum(dim=-1).mean()
                total_loss += mse.item() * targets.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        avg_vacuity = total_vacuity / total if total > 0 else 0.0
        avg_entropy = total_entropy / total if total > 0 else 0.0
        avg_strength = total_strength / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "correct": correct,
            "total": total,
            # Evidential uncertainty metrics
            "vacuity": avg_vacuity,      # Epistemic uncertainty (lack of evidence)
            "entropy": avg_entropy,       # Aleatoric uncertainty (prediction entropy)
            "strength": avg_strength,     # Dirichlet strength (total evidence)
        }

    def get_state(self) -> ModelState:
        """Get current model state.

        Returns:
            Model state dictionary
        """
        return get_model_state(self.model)

    def set_state(self, state: ModelState) -> None:
        """Set model state.

        Args:
            state: Model state dictionary to load
        """
        set_model_state(self.model, state)

    def aggregate_with_neighbors(
        self,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Aggregate own model with neighbor models.

        Args:
            neighbor_states: Dictionary mapping neighbor IDs to their model states
            round_num: Current training round
            **kwargs: Additional context for aggregation (e.g., train_loader for UBAR)

        Returns:
            Aggregated model state
        """
        if self.aggregator is None:
            # No aggregation, return own state
            return self.get_state()

        own_state = self.get_state()

        # Pass additional context to aggregator
        aggregation_kwargs = kwargs.copy()
        aggregation_kwargs.update({
            "train_loader": self.train_loader,
            "model_template": self.model,
            "device": self.device
        })

        aggregated_state = self.aggregator.aggregate(
            node_id=self.node_id,
            own_state=own_state,
            neighbor_states=neighbor_states,
            round_num=round_num,
            **aggregation_kwargs
        )

        return aggregated_state

    def apply_aggregated_state(self, aggregated_state: ModelState) -> None:
        """Apply aggregated state to own model.

        Args:
            aggregated_state: Aggregated model state to apply
        """
        self.set_state(aggregated_state)

    def get_aggregator_statistics(self) -> Dict[str, Any]:
        """Get statistics from aggregator if available.

        Returns:
            Dictionary of aggregator statistics
        """
        if self.aggregator is None:
            return {}
        return self.aggregator.get_statistics()
