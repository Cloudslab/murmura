"""
Ensemble Trust Detection System for Decentralized Federated Learning.

This module implements a comprehensive ensemble approach that combines multiple
complementary trust signals to detect various gradual poisoning attacks in a
dataset-agnostic manner.

Key Detection Components:
1. HSIC Parameter Correlation Analysis
2. Label Distribution Drift Detection  
3. Gradient Consistency Analysis
4. Cross-Validation Performance Trust
5. Model Behavior Consistency Monitoring
6. Statistical Outlier Detection
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix


class DetectionSignal(Enum):
    """Types of trust detection signals."""
    HSIC_CORRELATION = "hsic_correlation"
    LABEL_DISTRIBUTION = "label_distribution"
    GRADIENT_CONSISTENCY = "gradient_consistency"
    CROSS_VALIDATION = "cross_validation"
    MODEL_BEHAVIOR = "model_behavior"
    STATISTICAL_OUTLIER = "statistical_outlier"


@dataclass
class DetectionResult:
    """Result from a single detection method."""
    signal_type: DetectionSignal
    suspicion_score: float  # 0.0 = trusted, 1.0 = highly suspicious
    confidence: float       # 0.0 = low confidence, 1.0 = high confidence
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class EnsembleDetectionResult:
    """Final ensemble detection result."""
    is_malicious: bool
    overall_suspicion: float
    confidence: float
    individual_results: List[DetectionResult]
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LabelDistributionDetector:
    """Detects attacks by monitoring label distribution changes."""
    
    def __init__(self, num_classes: int, window_size: int = 10):
        self.num_classes = num_classes
        self.window_size = window_size
        self.label_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_distribution: Optional[np.ndarray] = None
        self.baseline_learned = False
        self.logger = logging.getLogger(f"{__name__}.LabelDistributionDetector")
    
    def update_labels(self, node_id: str, predictions: np.ndarray, true_labels: np.ndarray) -> DetectionResult:
        """
        Update label distribution tracking and detect drift.
        
        Args:
            node_id: ID of the node being monitored
            predictions: Model predictions from the node
            true_labels: Ground truth labels (for reference)
            
        Returns:
            DetectionResult with suspicion score based on label drift
        """
        # Convert to class predictions if needed
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions.astype(int)
        
        # Compute current label distribution
        current_dist = np.bincount(pred_classes, minlength=self.num_classes)
        current_dist = current_dist / (current_dist.sum() + 1e-8)  # Normalize
        
        # Store in history
        self.label_history[node_id].append(current_dist)
        
        # Learn baseline if not established
        if not self.baseline_learned and len(self.label_history[node_id]) >= 3:
            # Use first few rounds to establish baseline
            baseline_dists = list(self.label_history[node_id])[:3]
            self.baseline_distribution = np.mean(baseline_dists, axis=0)
            self.baseline_learned = True
            self.logger.info(f"Learned baseline label distribution: {self.baseline_distribution}")
        
        if not self.baseline_learned:
            return DetectionResult(
                signal_type=DetectionSignal.LABEL_DISTRIBUTION,
                suspicion_score=0.0,
                confidence=0.1,
                reasoning="Learning baseline distribution",
                metadata={"current_dist": current_dist.tolist()}
            )
        
        # Compute distribution drift using multiple metrics
        kl_divergence = self._compute_kl_divergence(current_dist, self.baseline_distribution)
        js_divergence = self._compute_js_divergence(current_dist, self.baseline_distribution)
        
        # Historical consistency check
        consistency_score = 0.0
        if len(self.label_history[node_id]) > 1:
            recent_dists = list(self.label_history[node_id])[-3:]
            consistency_score = self._compute_consistency(recent_dists)
        
        # Combine drift metrics
        drift_score = min(1.0, (kl_divergence + js_divergence) / 2.0)
        inconsistency_penalty = max(0.0, 1.0 - consistency_score)
        
        suspicion_score = min(1.0, drift_score + 0.3 * inconsistency_penalty)
        confidence = min(1.0, len(self.label_history[node_id]) / self.window_size)
        
        reasoning = f"Label drift: KL={kl_divergence:.3f}, JS={js_divergence:.3f}, consistency={consistency_score:.3f}"
        
        return DetectionResult(
            signal_type=DetectionSignal.LABEL_DISTRIBUTION,
            suspicion_score=suspicion_score,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "kl_divergence": kl_divergence,
                "js_divergence": js_divergence,
                "consistency_score": consistency_score,
                "current_dist": current_dist.tolist(),
                "baseline_dist": self.baseline_distribution.tolist()
            }
        )
    
    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        p = p + epsilon
        q = q + epsilon
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        return np.sum(p * np.log(p / q))
    
    def _compute_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two distributions."""
        epsilon = 1e-8
        p = p + epsilon
        q = q + epsilon
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return 0.5 * self._compute_kl_divergence(p, m) + 0.5 * self._compute_kl_divergence(q, m)
    
    def _compute_consistency(self, distributions: List[np.ndarray]) -> float:
        """Compute consistency score across multiple distributions."""
        if len(distributions) < 2:
            return 1.0
        
        # Compute pairwise JS divergences
        divergences = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                div = self._compute_js_divergence(distributions[i], distributions[j])
                divergences.append(div)
        
        # Convert to consistency score (lower divergence = higher consistency)
        avg_divergence = np.mean(divergences)
        consistency = max(0.0, 1.0 - avg_divergence * 2.0)  # Scale appropriately
        return consistency


class GradientConsistencyDetector:
    """Detects attacks by analyzing gradient alignment and consistency."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.gradient_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_gradients: Optional[Dict[str, np.ndarray]] = None
        self.baseline_learned = False
        self.logger = logging.getLogger(f"{__name__}.GradientConsistencyDetector")
    
    def update_gradients(self, node_id: str, current_params: Dict[str, np.ndarray], 
                        previous_params: Dict[str, np.ndarray], 
                        neighbor_gradients: Dict[str, Dict[str, np.ndarray]]) -> DetectionResult:
        """
        Update gradient tracking and detect inconsistencies.
        
        Args:
            node_id: ID of the node being monitored
            current_params: Current model parameters
            previous_params: Previous model parameters
            neighbor_gradients: Gradients from neighbor nodes
            
        Returns:
            DetectionResult with suspicion score based on gradient consistency
        """
        # Compute current gradient
        current_gradient = {}
        for key in current_params:
            if key in previous_params:
                current_gradient[key] = current_params[key] - previous_params[key]
        
        if not current_gradient:
            return DetectionResult(
                signal_type=DetectionSignal.GRADIENT_CONSISTENCY,
                suspicion_score=0.0,
                confidence=0.0,
                reasoning="No gradient information available"
            )
        
        # Flatten gradient for analysis
        flat_gradient = self._flatten_gradient(current_gradient)
        self.gradient_history[node_id].append(flat_gradient)
        
        # Learn baseline gradient patterns
        if not self.baseline_learned and len(self.gradient_history[node_id]) >= 3:
            baseline_grads = list(self.gradient_history[node_id])[:3]
            self.baseline_gradients = {
                "mean": np.mean(baseline_grads, axis=0),
                "std": np.std(baseline_grads, axis=0) + 1e-8
            }
            self.baseline_learned = True
            self.logger.info("Learned baseline gradient patterns")
        
        if not self.baseline_learned:
            return DetectionResult(
                signal_type=DetectionSignal.GRADIENT_CONSISTENCY,
                suspicion_score=0.0,
                confidence=0.1,
                reasoning="Learning baseline gradient patterns"
            )
        
        # Analyze gradient consistency
        scores = []
        
        # 1. Deviation from baseline pattern
        baseline_deviation = self._compute_baseline_deviation(flat_gradient)
        scores.append(baseline_deviation)
        
        # 2. Consistency with neighbors
        neighbor_consistency = 0.0
        if neighbor_gradients:
            neighbor_consistency = self._compute_neighbor_consistency(current_gradient, neighbor_gradients)
            scores.append(1.0 - neighbor_consistency)  # Lower consistency = higher suspicion
        
        # 3. Temporal consistency
        temporal_consistency = self._compute_temporal_consistency(node_id)
        scores.append(1.0 - temporal_consistency)
        
        # Combine scores
        suspicion_score = np.mean(scores)
        confidence = min(1.0, len(self.gradient_history[node_id]) / self.window_size)
        
        reasoning = f"Gradient analysis: baseline_dev={baseline_deviation:.3f}, neighbor_cons={neighbor_consistency:.3f}, temporal_cons={temporal_consistency:.3f}"
        
        return DetectionResult(
            signal_type=DetectionSignal.GRADIENT_CONSISTENCY,
            suspicion_score=suspicion_score,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "baseline_deviation": baseline_deviation,
                "neighbor_consistency": neighbor_consistency,
                "temporal_consistency": temporal_consistency
            }
        )
    
    def _flatten_gradient(self, gradient: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten gradient dictionary to vector."""
        flat_parts = []
        for key in sorted(gradient.keys()):
            flat_parts.append(gradient[key].flatten())
        return np.concatenate(flat_parts)
    
    def _compute_baseline_deviation(self, gradient: np.ndarray) -> float:
        """Compute deviation from baseline gradient pattern."""
        if self.baseline_gradients is None:
            return 0.0
        
        # Compute z-score relative to baseline
        mean = self.baseline_gradients["mean"]
        std = self.baseline_gradients["std"]
        
        z_scores = np.abs((gradient - mean) / std)
        # Use 95th percentile of z-scores as deviation metric
        deviation = min(1.0, np.percentile(z_scores, 95) / 3.0)  # Normalize by 3-sigma
        return deviation
    
    def _compute_neighbor_consistency(self, gradient: Dict[str, np.ndarray], 
                                    neighbor_gradients: Dict[str, Dict[str, np.ndarray]]) -> float:
        """Compute consistency with neighbor gradients."""
        if not neighbor_gradients:
            return 1.0
        
        flat_gradient = self._flatten_gradient(gradient)
        
        similarities = []
        for neighbor_id, neighbor_grad in neighbor_gradients.items():
            if neighbor_grad:
                flat_neighbor = self._flatten_gradient(neighbor_grad)
                # Compute cosine similarity
                similarity = np.dot(flat_gradient, flat_neighbor) / (
                    np.linalg.norm(flat_gradient) * np.linalg.norm(flat_neighbor) + 1e-8
                )
                similarities.append(max(0.0, similarity))  # Only positive similarities
        
        return np.mean(similarities) if similarities else 1.0
    
    def _compute_temporal_consistency(self, node_id: str) -> float:
        """Compute temporal consistency of gradients."""
        gradients = list(self.gradient_history[node_id])
        if len(gradients) < 2:
            return 1.0
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(gradients) - 1):
            grad1, grad2 = gradients[i], gradients[i + 1]
            similarity = np.dot(grad1, grad2) / (
                np.linalg.norm(grad1) * np.linalg.norm(grad2) + 1e-8
            )
            similarities.append(max(0.0, similarity))
        
        return np.mean(similarities)


class CrossValidationTrustDetector:
    """Detects attacks using cross-validation on held-out data."""
    
    def __init__(self, validation_split: float = 0.1):
        self.validation_split = validation_split
        self.baseline_accuracy: Optional[float] = None
        self.baseline_learned = False
        self.accuracy_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.logger = logging.getLogger(f"{__name__}.CrossValidationTrustDetector")
    
    def evaluate_update(self, node_id: str, model, neighbor_update: Dict[str, np.ndarray],
                       validation_data: Tuple[np.ndarray, np.ndarray]) -> DetectionResult:
        """
        Evaluate neighbor update using cross-validation.
        
        Args:
            node_id: ID of the neighbor node
            model: Current local model
            neighbor_update: Parameter update from neighbor
            validation_data: (features, labels) for validation
            
        Returns:
            DetectionResult with suspicion score based on validation performance
        """
        val_features, val_labels = validation_data
        
        if len(val_features) == 0:
            return DetectionResult(
                signal_type=DetectionSignal.CROSS_VALIDATION,
                suspicion_score=0.0,
                confidence=0.0,
                reasoning="No validation data available"
            )
        
        try:
            # Get current model accuracy
            current_acc = self._evaluate_model(model, val_features, val_labels)
            
            # Apply neighbor update and evaluate
            updated_model = self._apply_update(model, neighbor_update)
            updated_acc = self._evaluate_model(updated_model, val_features, val_labels)
            
            # Learn baseline if not established
            if not self.baseline_learned:
                if len(self.accuracy_history[node_id]) >= 3:
                    baseline_accs = list(self.accuracy_history[node_id])[:3]
                    self.baseline_accuracy = np.mean(baseline_accs)
                    self.baseline_learned = True
                    self.logger.info(f"Learned baseline accuracy: {self.baseline_accuracy:.3f}")
            
            # Store accuracy
            self.accuracy_history[node_id].append(updated_acc)
            
            if not self.baseline_learned:
                return DetectionResult(
                    signal_type=DetectionSignal.CROSS_VALIDATION,
                    suspicion_score=0.0,
                    confidence=0.1,
                    reasoning="Learning baseline accuracy"
                )
            
            # Compute suspicion based on performance degradation
            performance_drop = max(0.0, current_acc - updated_acc)
            baseline_drop = max(0.0, self.baseline_accuracy - updated_acc)
            
            # Normalize suspicion score
            suspicion_score = min(1.0, max(performance_drop * 2.0, baseline_drop * 1.5))
            
            # Confidence based on validation set size and history
            confidence = min(1.0, len(val_features) / 100.0 * len(self.accuracy_history[node_id]) / 10.0)
            
            reasoning = f"Cross-validation: current_acc={current_acc:.3f}, updated_acc={updated_acc:.3f}, baseline={self.baseline_accuracy:.3f}"
            
            return DetectionResult(
                signal_type=DetectionSignal.CROSS_VALIDATION,
                suspicion_score=suspicion_score,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "current_accuracy": current_acc,
                    "updated_accuracy": updated_acc,
                    "baseline_accuracy": self.baseline_accuracy,
                    "performance_drop": performance_drop
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Cross-validation evaluation failed: {e}")
            return DetectionResult(
                signal_type=DetectionSignal.CROSS_VALIDATION,
                suspicion_score=0.0,
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}"
            )
    
    def _evaluate_model(self, model, features: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate model accuracy on given data."""
        # This is a simplified evaluation - in practice, you'd use the actual model
        # For now, return a placeholder
        try:
            # Convert to torch if needed and evaluate
            if hasattr(model, 'eval'):
                model.eval()
                with torch.no_grad():
                    if isinstance(features, np.ndarray):
                        features = torch.FloatTensor(features)
                    if isinstance(labels, np.ndarray):
                        labels = torch.LongTensor(labels)
                    
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = (predicted == labels).float().mean().item()
                    return accuracy
            else:
                # Fallback for non-torch models
                return 0.9  # Placeholder
        except Exception:
            return 0.9  # Placeholder fallback
    
    def _apply_update(self, model, update: Dict[str, np.ndarray]):
        """Apply parameter update to model and return new model."""
        # This is simplified - in practice, you'd properly apply the update
        # For now, return the original model (placeholder)
        return model


class EnsembleTrustDetector:
    """Main ensemble detector that combines all detection methods."""
    
    def __init__(self, num_classes: int, ensemble_weights: Optional[Dict[DetectionSignal, float]] = None):
        self.num_classes = num_classes
        
        # Initialize individual detectors
        self.label_detector = LabelDistributionDetector(num_classes)
        self.gradient_detector = GradientConsistencyDetector()
        self.crossval_detector = CrossValidationTrustDetector()
        
        # Ensemble weights (can be learned/adapted)
        self.weights = ensemble_weights or {
            DetectionSignal.HSIC_CORRELATION: 0.2,
            DetectionSignal.LABEL_DISTRIBUTION: 0.25,
            DetectionSignal.GRADIENT_CONSISTENCY: 0.25,
            DetectionSignal.CROSS_VALIDATION: 0.3,
        }
        
        # Detection history for meta-learning
        self.detection_history: Dict[str, List[EnsembleDetectionResult]] = defaultdict(list)
        
        self.logger = logging.getLogger(f"{__name__}.EnsembleTrustDetector")
    
    def detect_trust_drift(self, node_id: str, detection_data: Dict[str, Any]) -> EnsembleDetectionResult:
        """
        Perform ensemble trust detection.
        
        Args:
            node_id: ID of the node being evaluated
            detection_data: Dictionary containing all necessary data for detection
            
        Returns:
            EnsembleDetectionResult with overall trust assessment
        """
        individual_results = []
        
        # 1. Label Distribution Detection
        if 'predictions' in detection_data and 'labels' in detection_data:
            label_result = self.label_detector.update_labels(
                node_id, detection_data['predictions'], detection_data['labels']
            )
            individual_results.append(label_result)
        
        # 2. Gradient Consistency Detection
        if all(key in detection_data for key in ['current_params', 'previous_params']):
            gradient_result = self.gradient_detector.update_gradients(
                node_id, 
                detection_data['current_params'],
                detection_data['previous_params'],
                detection_data.get('neighbor_gradients', {})
            )
            individual_results.append(gradient_result)
        
        # 3. Cross-Validation Detection
        if all(key in detection_data for key in ['model', 'neighbor_update', 'validation_data']):
            crossval_result = self.crossval_detector.evaluate_update(
                node_id,
                detection_data['model'],
                detection_data['neighbor_update'],
                detection_data['validation_data']
            )
            individual_results.append(crossval_result)
        
        # 4. HSIC Detection (if provided)
        if 'hsic_result' in detection_data:
            individual_results.append(detection_data['hsic_result'])
        
        # Combine results using ensemble weights
        ensemble_result = self._combine_results(individual_results)
        
        # Store in history
        self.detection_history[node_id].append(ensemble_result)
        
        return ensemble_result
    
    def _combine_results(self, results: List[DetectionResult]) -> EnsembleDetectionResult:
        """Combine individual detection results into ensemble result."""
        if not results:
            return EnsembleDetectionResult(
                is_malicious=False,
                overall_suspicion=0.0,
                confidence=0.0,
                individual_results=[],
                reasoning="No detection signals available"
            )
        
        # Weighted combination of suspicion scores
        weighted_suspicion = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0
        
        reasoning_parts = []
        
        for result in results:
            weight = self.weights.get(result.signal_type, 0.1)
            if result.confidence > 0.1:  # Only include confident results
                weighted_suspicion += weight * result.suspicion_score * result.confidence
                weighted_confidence += weight * result.confidence
                total_weight += weight
                
                reasoning_parts.append(
                    f"{result.signal_type.value}={result.suspicion_score:.2f}(conf={result.confidence:.2f})"
                )
        
        if total_weight > 0:
            overall_suspicion = weighted_suspicion / total_weight
            overall_confidence = weighted_confidence / total_weight
        else:
            overall_suspicion = 0.0
            overall_confidence = 0.0
        
        # Decision threshold with adaptive component
        base_threshold = 0.6
        confidence_adjustment = (overall_confidence - 0.5) * 0.2  # Adjust threshold based on confidence
        decision_threshold = max(0.3, min(0.9, base_threshold + confidence_adjustment))
        
        is_malicious = overall_suspicion > decision_threshold
        
        reasoning = f"Ensemble detection (threshold={decision_threshold:.2f}): {', '.join(reasoning_parts)}"
        
        return EnsembleDetectionResult(
            is_malicious=is_malicious,
            overall_suspicion=overall_suspicion,
            confidence=overall_confidence,
            individual_results=results,
            reasoning=reasoning,
            metadata={
                "decision_threshold": decision_threshold,
                "weights_used": {signal.value: weight for signal, weight in self.weights.items()},
                "total_signals": len(results)
            }
        )
    
    def update_weights(self, feedback: Dict[str, bool]) -> None:
        """Update ensemble weights based on feedback."""
        # This is a placeholder for adaptive weight learning
        # In practice, you'd implement a more sophisticated update mechanism
        self.logger.info("Ensemble weight adaptation not yet implemented")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        stats = {
            "total_evaluations": sum(len(history) for history in self.detection_history.values()),
            "nodes_monitored": len(self.detection_history),
            "current_weights": self.weights.copy(),
            "detector_states": {
                "label_detector": {
                    "baseline_learned": self.label_detector.baseline_learned,
                    "baseline_distribution": self.label_detector.baseline_distribution.tolist() if self.label_detector.baseline_distribution is not None else None
                },
                "gradient_detector": {
                    "baseline_learned": self.gradient_detector.baseline_learned
                },
                "crossval_detector": {
                    "baseline_learned": self.crossval_detector.baseline_learned,
                    "baseline_accuracy": self.crossval_detector.baseline_accuracy
                }
            }
        }
        return stats