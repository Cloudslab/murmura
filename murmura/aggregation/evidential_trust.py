"""Evidential Trust-Aware Aggregation for Byzantine-Resilient Decentralized FL.

This module implements the trust-aware aggregation algorithm from:
    "Evidential Trust-Aware Model Personalization in Decentralized
    Federated Learning for Wearable IoT"
    Rangwala, Sinnott, Buyya - University of Melbourne

The key insight is that epistemic-aleatoric uncertainty decomposition from
Dirichlet-based evidential models directly indicates peer reliability:
- High epistemic uncertainty (vacuity) → insufficient learning, possibly Byzantine
- High aleatoric uncertainty (entropy) → inherent data ambiguity, still trustworthy
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from murmura.aggregation.base import Aggregator, average_states, set_model_state
from murmura.core.types import ModelState


class EvidentialTrustAggregator(Aggregator):
    """Trust-aware aggregation using evidential uncertainty for peer evaluation.

    This aggregator evaluates peer models by running them on local validation
    samples and examining their uncertainty profiles. Peers with low epistemic
    uncertainty (well-trained) receive high trust; peers with high epistemic
    uncertainty (undertrained or Byzantine) are filtered.

    Uses a BALANCE-style tightening threshold: as training progresses and models
    accumulate more evidence, we require higher trust scores (lower vacuity)
    from peers to be accepted.

    The algorithm proceeds in three phases:
    1. Compute trust scores for each neighbor via cross-evaluation
    2. Filter neighbors below trust threshold and normalize weights
    3. Perform weighted aggregation with self-weight for personalization
    """

    def __init__(
        self,
        # Trust computation parameters
        vacuity_threshold: float = 0.5,
        accuracy_weight: float = 0.5,
        trust_threshold: float = 0.3,
        # Personalization parameter
        self_weight: float = 0.5,
        # Adaptive trust dynamics
        use_adaptive_trust: bool = True,
        trust_momentum: float = 0.7,  # γ in EMA: trust_t = γ*new + (1-γ)*old
        # BALANCE-style tightening threshold
        use_tightening_threshold: bool = True,
        gamma: float = 0.5,  # Initial threshold multiplier (starts lenient)
        kappa: float = 1.0,  # Tightening rate (higher = faster tightening)
        total_rounds: int = 50,
        # Minimum neighbors fallback
        min_neighbors: int = 1,
        # Validation samples
        max_eval_samples: int = 100,
        # Statistics tracking
        track_statistics: bool = True,
        **kwargs
    ):
        """Initialize Evidential Trust Aggregator.

        Args:
            vacuity_threshold: τ_u - threshold above which vacuity incurs penalty
            accuracy_weight: w_a - weight of accuracy in trust score (0-1)
            trust_threshold: τ_min - base minimum trust to include peer
            self_weight: α - weight for own model vs neighbor aggregate (0-1)
            use_adaptive_trust: Whether to use EMA smoothing on trust scores
            trust_momentum: γ_ema - momentum for trust EMA (higher = faster adaptation)
            use_tightening_threshold: Whether to tighten threshold over training
                (BALANCE-style: start lenient, become stricter as models converge)
            gamma: γ - initial threshold multiplier (threshold starts at γ * base)
            kappa: κ - exponential tightening rate
            total_rounds: T - total training rounds (for threshold scheduling)
            min_neighbors: Minimum neighbors to accept (fallback to highest trust)
            max_eval_samples: Maximum validation samples for trust computation
            track_statistics: Whether to track detailed statistics
        """
        super().__init__(**kwargs)

        # Trust computation
        self.vacuity_threshold = vacuity_threshold
        self.accuracy_weight = accuracy_weight
        self.trust_threshold = trust_threshold
        self.base_trust_threshold = trust_threshold

        # Personalization
        self.self_weight = self_weight

        # Adaptive dynamics
        self.use_adaptive_trust = use_adaptive_trust
        self.trust_momentum = trust_momentum

        # BALANCE-style tightening threshold
        self.use_tightening_threshold = use_tightening_threshold
        self.gamma = gamma
        self.kappa = kappa
        self.total_rounds = total_rounds
        self.min_neighbors = min_neighbors

        # Evaluation
        self.max_eval_samples = max_eval_samples

        # Statistics
        self.track_statistics = track_statistics
        self._trust_history: Dict[int, List[float]] = defaultdict(list)
        self._smoothed_trust: Dict[int, float] = {}
        self._statistics: Dict[str, Any] = {
            "rounds_processed": 0,
            "neighbors_evaluated": 0,
            "neighbors_accepted": 0,
            "neighbors_rejected": 0,
            "avg_trust_scores": [],
            "avg_vacuity": [],
            "avg_entropy": [],
            "threshold_history": [],
        }

    def aggregate(
        self,
        node_id: int,
        own_state: ModelState,
        neighbor_states: Dict[int, ModelState],
        round_num: int,
        **kwargs
    ) -> ModelState:
        """Aggregate using evidential trust-aware weighting.

        Args:
            node_id: ID of node performing aggregation
            own_state: Own model state
            neighbor_states: Dictionary of neighbor ID -> model state
            round_num: Current training round
            **kwargs: Must include 'train_loader', 'model_template', 'device'

        Returns:
            Aggregated model state
        """
        # Extract required context
        train_loader = kwargs.get("train_loader")
        model_template = kwargs.get("model_template")
        device = kwargs.get("device", torch.device("cpu"))

        if train_loader is None or model_template is None:
            # Fallback to simple averaging if no evaluation context
            all_states = [own_state] + list(neighbor_states.values())
            return average_states(all_states)

        # Update adaptive threshold if enabled
        current_threshold = self._get_current_threshold(round_num)

        # Phase 1: Compute trust scores for each neighbor
        trust_scores = {}
        neighbor_metrics = {}

        for neighbor_id, neighbor_state in neighbor_states.items():
            trust, metrics = self._compute_trust_score(
                neighbor_state=neighbor_state,
                model_template=model_template,
                train_loader=train_loader,
                device=device,
            )

            # Apply adaptive trust smoothing
            if self.use_adaptive_trust:
                trust = self._apply_trust_smoothing(neighbor_id, trust)

            trust_scores[neighbor_id] = trust
            neighbor_metrics[neighbor_id] = metrics

        # Phase 2: Filter by threshold and normalize weights
        accepted_neighbors = {
            nid: score for nid, score in trust_scores.items()
            if score >= current_threshold
        }

        # Track statistics
        if self.track_statistics:
            self._update_statistics(
                trust_scores, neighbor_metrics, accepted_neighbors,
                current_threshold, round_num
            )

        # If no neighbors pass threshold, return own state
        if not accepted_neighbors:
            return own_state

        # Normalize trust scores to weights
        total_trust = sum(accepted_neighbors.values())
        neighbor_weights = {
            nid: score / total_trust for nid, score in accepted_neighbors.items()
        }

        # Phase 3: Weighted aggregation
        # First, compute weighted average of accepted neighbors
        accepted_states = [neighbor_states[nid] for nid in neighbor_weights.keys()]
        weights = list(neighbor_weights.values())
        neighbor_aggregate = average_states(accepted_states, weights)

        # Combine with own state using self_weight
        final_state = average_states(
            [own_state, neighbor_aggregate],
            [self.self_weight, 1.0 - self.self_weight]
        )

        return final_state

    def _compute_trust_score(
        self,
        neighbor_state: ModelState,
        model_template: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute trust score for a neighbor by cross-evaluation.

        Implements Algorithm 2 from the paper:
        trust_j = (1 - vacuity) * (w_a * accuracy + (1 - w_a))
        with exponential penalty if vacuity > threshold

        Args:
            neighbor_state: Neighbor's model state
            model_template: Model architecture template
            train_loader: Local validation/training data
            device: Computation device

        Returns:
            Tuple of (trust_score, metrics_dict)
        """
        # Create evaluation model from neighbor state
        eval_model = copy.deepcopy(model_template)
        set_model_state(eval_model, neighbor_state)
        eval_model.to(device)
        eval_model.eval()

        # Accumulators
        total_vacuity = 0.0
        total_entropy = 0.0
        total_strength = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Limit evaluation samples for efficiency
                if total >= self.max_eval_samples:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass - get Dirichlet alpha parameters
                alpha = eval_model(inputs)

                # Compute uncertainty metrics
                S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength
                K = alpha.shape[-1]  # Number of classes
                probs = alpha / S  # Expected probabilities

                # Epistemic uncertainty: vacuity = K / S
                vacuity = (K / S.squeeze(-1))

                # Aleatoric uncertainty: entropy of expected probabilities
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

                # Predictions and accuracy
                predictions = alpha.argmax(dim=-1)
                correct += (predictions == targets).sum().item()

                # Accumulate
                batch_size = inputs.size(0)
                total_vacuity += vacuity.sum().item()
                total_entropy += entropy.sum().item()
                total_strength += S.squeeze(-1).sum().item()
                total += batch_size

        # Compute averages
        avg_vacuity = total_vacuity / total if total > 0 else 1.0
        avg_entropy = total_entropy / total if total > 0 else 0.0
        avg_strength = total_strength / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # Compute trust score (Algorithm 2 from paper)
        # trust = (1 - vacuity) * (w_a * accuracy + (1 - w_a))
        base_trust = (1.0 - avg_vacuity) * (
            self.accuracy_weight * accuracy + (1.0 - self.accuracy_weight)
        )

        # Apply exponential penalty if vacuity exceeds threshold
        if avg_vacuity > self.vacuity_threshold:
            penalty = torch.exp(
                torch.tensor(-(avg_vacuity - self.vacuity_threshold))
            ).item()
            trust = base_trust * penalty
        else:
            trust = base_trust

        # Ensure trust is in [0, 1]
        trust = max(0.0, min(1.0, trust))

        metrics = {
            "vacuity": avg_vacuity,
            "entropy": avg_entropy,
            "strength": avg_strength,
            "accuracy": accuracy,
            "base_trust": base_trust,
            "final_trust": trust,
        }

        return trust, metrics

    def _apply_trust_smoothing(self, neighbor_id: int, new_trust: float) -> float:
        """Apply exponential moving average smoothing to trust scores.

        trust_t = γ * new_trust + (1 - γ) * trust_{t-1}

        Args:
            neighbor_id: ID of the neighbor
            new_trust: Newly computed trust score

        Returns:
            Smoothed trust score
        """
        if neighbor_id in self._smoothed_trust:
            old_trust = self._smoothed_trust[neighbor_id]
            smoothed = (
                self.trust_momentum * new_trust +
                (1.0 - self.trust_momentum) * old_trust
            )
        else:
            smoothed = new_trust

        self._smoothed_trust[neighbor_id] = smoothed
        self._trust_history[neighbor_id].append(smoothed)

        return smoothed

    def _get_current_threshold(self, round_num: int) -> float:
        """Get current trust threshold with BALANCE-style tightening.

        BALANCE-style: threshold TIGHTENS as training progresses.
        As models accumulate more evidence and vacuity decreases,
        we require higher trust scores from peers.

        Formula: τ(t) = τ_base * (1 - γ * exp(-κ * t / T))

        - At t=0: threshold ≈ τ_base * (1 - γ) (lenient)
        - As t→T: threshold → τ_base (strict)

        Args:
            round_num: Current training round

        Returns:
            Current threshold value
        """
        if not self.use_tightening_threshold:
            return self.trust_threshold

        # BALANCE-style tightening: start lenient, become strict
        # λ_t = t / T (progress ratio)
        lambda_t = round_num / max(1, self.total_rounds)

        # exp(-κ * λ_t) decreases from 1 to ~0 as training progresses
        decay_factor = torch.exp(torch.tensor(-self.kappa * lambda_t)).item()

        # Threshold starts at γ * base, tightens toward base
        # threshold = base * (1 - γ * decay_factor)
        # At t=0: decay_factor ≈ 1, threshold ≈ base * (1 - γ) [lenient]
        # At t=T: decay_factor ≈ 0, threshold ≈ base [strict]
        current = self.base_trust_threshold * (1.0 - self.gamma * decay_factor)

        # Ensure threshold is positive and reasonable
        current = max(0.05, min(current, self.base_trust_threshold))

        return current

    def _update_statistics(
        self,
        trust_scores: Dict[int, float],
        neighbor_metrics: Dict[int, Dict[str, float]],
        accepted_neighbors: Dict[int, float],
        current_threshold: float,
        round_num: int,
    ) -> None:
        """Update internal statistics for monitoring."""
        self._statistics["rounds_processed"] += 1
        self._statistics["neighbors_evaluated"] += len(trust_scores)
        self._statistics["neighbors_accepted"] += len(accepted_neighbors)
        self._statistics["neighbors_rejected"] += len(trust_scores) - len(accepted_neighbors)

        if trust_scores:
            self._statistics["avg_trust_scores"].append(
                sum(trust_scores.values()) / len(trust_scores)
            )

        if neighbor_metrics:
            vacuities = [m["vacuity"] for m in neighbor_metrics.values()]
            entropies = [m["entropy"] for m in neighbor_metrics.values()]
            self._statistics["avg_vacuity"].append(sum(vacuities) / len(vacuities))
            self._statistics["avg_entropy"].append(sum(entropies) / len(entropies))

        self._statistics["threshold_history"].append(current_threshold)

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics for monitoring.

        Returns:
            Dictionary containing:
            - rounds_processed: Total rounds of aggregation
            - neighbors_evaluated: Total neighbor evaluations
            - neighbors_accepted: Neighbors passing trust threshold
            - neighbors_rejected: Neighbors filtered out
            - acceptance_rate: Proportion of neighbors accepted
            - avg_trust: Mean trust score across all evaluations
            - avg_vacuity: Mean epistemic uncertainty
            - avg_entropy: Mean aleatoric uncertainty
            - trust_history: Per-neighbor trust score history
        """
        stats = self._statistics.copy()

        # Compute summary statistics
        if stats["neighbors_evaluated"] > 0:
            stats["acceptance_rate"] = (
                stats["neighbors_accepted"] / stats["neighbors_evaluated"]
            )
        else:
            stats["acceptance_rate"] = 0.0

        if stats["avg_trust_scores"]:
            stats["mean_trust"] = sum(stats["avg_trust_scores"]) / len(stats["avg_trust_scores"])
        else:
            stats["mean_trust"] = 0.0

        if stats["avg_vacuity"]:
            stats["mean_vacuity"] = sum(stats["avg_vacuity"]) / len(stats["avg_vacuity"])
        else:
            stats["mean_vacuity"] = 0.0

        if stats["avg_entropy"]:
            stats["mean_entropy"] = sum(stats["avg_entropy"]) / len(stats["avg_entropy"])
        else:
            stats["mean_entropy"] = 0.0

        # Include per-neighbor trust history
        stats["trust_history_per_neighbor"] = dict(self._trust_history)
        stats["current_smoothed_trust"] = dict(self._smoothed_trust)

        return stats

    def reset_statistics(self) -> None:
        """Reset all tracked statistics."""
        self._trust_history.clear()
        self._smoothed_trust.clear()
        self._statistics = {
            "rounds_processed": 0,
            "neighbors_evaluated": 0,
            "neighbors_accepted": 0,
            "neighbors_rejected": 0,
            "avg_trust_scores": [],
            "avg_vacuity": [],
            "avg_entropy": [],
            "threshold_history": [],
        }
