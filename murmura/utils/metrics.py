"""Evaluation and metrics utilities."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
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


def evaluate_model_comprehensive(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a model with comprehensive metrics.

    Args:
        model: PyTorch model to evaluate
        loader: DataLoader containing evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes (auto-detected if None)

    Returns:
        Dictionary with accuracy, precision, recall, F1, AUC, loss, etc.
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_predictions.append(predicted.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)

    total = len(all_targets)
    if total == 0:
        return {"accuracy": 0.0, "loss": 0.0, "total": 0}

    # Auto-detect num_classes
    if num_classes is None:
        num_classes = all_probs.shape[1]

    # Basic accuracy
    correct = all_predictions.eq(all_targets).sum().item()
    accuracy = correct / total
    avg_loss = total_loss / total

    # Per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for c in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((all_predictions == c) & (all_targets == c)).sum().item()
        fp = ((all_predictions == c) & (all_targets != c)).sum().item()
        fn = ((all_predictions != c) & (all_targets == c)).sum().item()

        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    # Macro-averaged metrics
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    # Weighted metrics (by class frequency)
    class_counts = [(all_targets == c).sum().item() for c in range(num_classes)]
    total_samples = sum(class_counts)
    if total_samples > 0:
        weights = [c / total_samples for c in class_counts]
        weighted_precision = sum(p * w for p, w in zip(precision_per_class, weights))
        weighted_recall = sum(r * w for r, w in zip(recall_per_class, weights))
        weighted_f1 = sum(f * w for f, w in zip(f1_per_class, weights))
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    # AUC (one-vs-rest, macro-averaged)
    auc = compute_multiclass_auc(all_targets.numpy(), all_probs.numpy(), num_classes)

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "correct": correct,
        "total": total,
        # Macro-averaged metrics
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "auc": auc,
        # Weighted metrics
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        # Per-class metrics
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "class_counts": class_counts,
    }


def compute_multiclass_auc(
    targets: np.ndarray,
    probs: np.ndarray,
    num_classes: int
) -> float:
    """Compute macro-averaged AUC for multiclass classification.

    Uses one-vs-rest approach and averages across classes.

    Args:
        targets: Ground truth labels (N,)
        probs: Predicted probabilities (N, num_classes)
        num_classes: Number of classes

    Returns:
        Macro-averaged AUC score
    """
    auc_scores = []

    for c in range(num_classes):
        # Binary labels for this class
        binary_targets = (targets == c).astype(np.float32)
        class_probs = probs[:, c]

        # Only compute if we have both positive and negative samples
        if binary_targets.sum() > 0 and binary_targets.sum() < len(binary_targets):
            auc = compute_binary_auc(binary_targets, class_probs)
            auc_scores.append(auc)

    return np.mean(auc_scores) if auc_scores else 0.5


def compute_binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC for binary classification using trapezoidal rule.

    Args:
        targets: Binary ground truth (0 or 1)
        scores: Predicted scores/probabilities

    Returns:
        AUC score
    """
    # Sort by scores descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_targets = targets[sorted_indices]

    # Compute TPR and FPR at each threshold
    num_pos = targets.sum()
    num_neg = len(targets) - num_pos

    if num_pos == 0 or num_neg == 0:
        return 0.5

    tpr = np.cumsum(sorted_targets) / num_pos
    fpr = np.cumsum(1 - sorted_targets) / num_neg

    # Add (0, 0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)

    return float(auc)


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
