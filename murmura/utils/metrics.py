"""Evaluation and metrics utilities."""

from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, int, int]:
    """Evaluate a model on a dataset.

    Args:
        model: PyTorch model to evaluate
        loader: DataLoader containing evaluation data
        device: Device to run evaluation on

    Returns:
        Tuple of (accuracy, loss, correct_count, total_count)
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return accuracy, avg_loss, correct, total


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from predictions and targets.

    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels

    Returns:
        Accuracy as a float between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    correct = predictions.eq(targets).sum().item()
    total = targets.size(0)

    return correct / total if total > 0 else 0.0
