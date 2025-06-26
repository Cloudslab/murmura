"""
Performance-based Trust Monitoring for Federated Learning.

This module implements trust monitoring based on actual model performance
rather than just parameter correlations. It evaluates neighbor models on
local test data to detect functional degradation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import logging
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance-based trust metrics."""
    accuracy: float
    loss: float
    prediction_entropy: float
    confidence_score: float
    agreement_rate: float


class PerformanceTrustMonitor:
    """
    Trust monitor that evaluates neighbor models based on actual performance
    on local test data, rather than just parameter similarity.
    
    This approach is more intuitive and directly measures functional quality.
    """
    
    def __init__(
        self,
        node_id: str,
        performance_threshold: float = 0.05,  # 5% performance drop threshold
        window_size: int = 10,  # Number of recent evaluations to track
        min_samples: int = 50,  # Minimum test samples needed
        confidence_threshold: float = 0.8,  # Minimum confidence for decisions
        enable_prediction_analysis: bool = True,
    ):
        """
        Initialize performance-based trust monitor.
        
        Args:
            node_id: ID of the node this monitor belongs to
            performance_threshold: Max acceptable performance drop (0.05 = 5%)
            window_size: Number of recent evaluations to maintain
            min_samples: Minimum test samples required for reliable evaluation
            confidence_threshold: Minimum confidence score for trust decisions
            enable_prediction_analysis: Whether to analyze prediction patterns
        """
        self.node_id = node_id
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.enable_prediction_analysis = enable_prediction_analysis
        
        # Test data for evaluation (set by external caller)
        self.test_features: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        
        # My own model performance baseline
        self.baseline_accuracy: float = 0.0
        self.baseline_loss: float = float('inf')
        
        # History tracking
        self.neighbor_performance: Dict[str, deque] = {}
        self.trust_scores: Dict[str, float] = {}
        self.performance_history: Dict[str, deque] = {}
        
        # Logger
        self.logger = logging.getLogger(f"murmura.trust.PerformanceTrust.{node_id}")
        
    def set_test_data(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Set the local test data for evaluating neighbor models.
        
        Args:
            features: Test features
            labels: Test labels
        """
        if len(features) < self.min_samples:
            self.logger.warning(
                f"Test set has only {len(features)} samples, minimum {self.min_samples} recommended"
            )
        
        self.test_features = features
        self.test_labels = labels
        self.logger.info(f"Set test data: {len(features)} samples")
        
    def set_baseline_performance(self, my_model_params: Dict[str, np.ndarray]) -> None:
        """
        Establish baseline performance using my own model.
        
        Args:
            my_model_params: Current model parameters
        """
        if self.test_features is None:
            self.logger.warning("No test data available for baseline evaluation")
            return
            
        try:
            metrics = self._evaluate_model_performance(my_model_params)
            self.baseline_accuracy = metrics.accuracy
            self.baseline_loss = metrics.loss
            
            self.logger.info(
                f"Baseline performance established: "
                f"Accuracy={self.baseline_accuracy:.4f}, Loss={self.baseline_loss:.4f}"
            )
        except Exception as e:
            self.logger.error(f"Failed to establish baseline: {e}")
    
    def _evaluate_model_performance(
        self, 
        model_params: Dict[str, np.ndarray]
    ) -> PerformanceMetrics:
        """
        Evaluate model performance on local test data using actual model inference.
        
        Args:
            model_params: Model parameters to evaluate
            
        Returns:
            Performance metrics from actual model evaluation
        """
        if self.test_features is None or self.test_labels is None:
            raise ValueError("Test data not set")
        
        try:
            # Use actual model evaluation with ModelEvaluator
            from murmura.trust.model_evaluator import ModelEvaluator
            
            # Create evaluator if not exists
            if not hasattr(self, '_model_evaluator'):
                self._model_evaluator = ModelEvaluator(
                    model_template=None,  # Will create MNIST model
                    device="cpu",
                    batch_size=32
                )
            
            # Evaluate parameters on test data
            eval_results = self._model_evaluator.evaluate_parameters_on_data(
                model_params, self.test_features, self.test_labels
            )
            
            if eval_results["evaluation_successful"]:
                return PerformanceMetrics(
                    accuracy=eval_results["accuracy"],
                    loss=eval_results["loss"],
                    prediction_entropy=eval_results["prediction_entropy"],
                    confidence_score=eval_results["confidence"],
                    agreement_rate=eval_results["confidence"],  # Use confidence as proxy for agreement
                )
            else:
                # Fall back to parameter-based metrics if evaluation fails
                return self._fallback_parameter_metrics(model_params)
                
        except Exception as e:
            self.logger.warning(f"Model evaluation failed, using fallback metrics: {e}")
            return self._fallback_parameter_metrics(model_params)
    
    def _fallback_parameter_metrics(self, model_params: Dict[str, np.ndarray]) -> PerformanceMetrics:
        """Fallback to parameter-based metrics if model evaluation fails."""
        try:
            # Parameter-based quality metrics (simplified version of previous implementation)
            param_norms = [np.linalg.norm(param) for param in model_params.values()]
            avg_norm = np.mean(param_norms)
            norm_stability = 1.0 / (1.0 + np.std(param_norms))
            
            all_params = np.concatenate([p.flatten() for p in model_params.values()])
            param_mean = np.mean(all_params)
            param_std = np.std(all_params)
            distribution_quality = np.exp(-abs(param_mean)) * min(param_std, 1.0)
            
            proxy_accuracy = np.clip(0.5 * norm_stability + 0.5 * distribution_quality, 0.1, 0.9)
            
            return PerformanceMetrics(
                accuracy=proxy_accuracy,
                loss=-np.log(max(proxy_accuracy, 0.001)),
                prediction_entropy=2.0 * (1.0 - proxy_accuracy),
                confidence_score=proxy_accuracy,
                agreement_rate=proxy_accuracy
            )
        except:
            # Ultimate fallback
            return PerformanceMetrics(
                accuracy=0.5, loss=1.0, prediction_entropy=1.0, 
                confidence_score=0.5, agreement_rate=0.5
            )
    
    def assess_neighbor_trust(
        self, 
        neighbor_id: str, 
        neighbor_params: Dict[str, np.ndarray]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Assess trust in a neighbor based on their model's performance.
        
        Args:
            neighbor_id: ID of the neighbor
            neighbor_params: Neighbor's model parameters
            
        Returns:
            Tuple of (trust_score, is_suspicious, detailed_stats)
        """
        if self.test_features is None:
            self.logger.warning("No test data available for performance evaluation")
            return 1.0, False, {"error": "no_test_data"}
        
        try:
            # Evaluate neighbor's model performance
            metrics = self._evaluate_model_performance(neighbor_params)
            
            # Initialize tracking for new neighbor
            if neighbor_id not in self.neighbor_performance:
                self.neighbor_performance[neighbor_id] = deque(maxlen=self.window_size)
                self.performance_history[neighbor_id] = deque(maxlen=self.window_size)
            
            # Store performance metrics
            self.neighbor_performance[neighbor_id].append(metrics)
            
            # Calculate trust score based on multiple factors
            trust_score = self._calculate_trust_score(neighbor_id, metrics)
            self.trust_scores[neighbor_id] = trust_score
            
            # Determine if neighbor is suspicious
            is_suspicious = self._is_neighbor_suspicious(neighbor_id, metrics)
            
            # Compile detailed statistics
            stats = self._compile_performance_stats(neighbor_id, metrics)
            
            # Log significant events
            if is_suspicious:
                self.logger.warning(
                    f"Suspicious behavior detected from {neighbor_id}: "
                    f"Trust={trust_score:.3f}, Accuracy={metrics.accuracy:.3f} "
                    f"(baseline: {self.baseline_accuracy:.3f})"
                )
            
            return trust_score, is_suspicious, stats
            
        except Exception as e:
            self.logger.error(f"Error assessing neighbor {neighbor_id}: {e}")
            return 0.5, True, {"error": str(e)}
    
    def _calculate_trust_score(
        self, 
        neighbor_id: str, 
        metrics: PerformanceMetrics
    ) -> float:
        """
        Calculate trust score based on performance metrics.
        
        Args:
            neighbor_id: Neighbor ID
            metrics: Performance metrics
            
        Returns:
            Trust score between 0.0 and 1.0
        """
        # Performance-based trust components
        
        # 1. Absolute performance quality
        if self.baseline_accuracy > 0:
            performance_ratio = metrics.accuracy / self.baseline_accuracy
            performance_component = min(performance_ratio, 1.0)
        else:
            performance_component = metrics.accuracy
        
        # 2. Loss quality (lower is better)
        if self.baseline_loss < float('inf'):
            loss_ratio = self.baseline_loss / max(metrics.loss, 0.001)
            loss_component = min(loss_ratio, 1.0)
        else:
            loss_component = 1.0 / (1.0 + metrics.loss)
        
        # 3. Prediction confidence
        confidence_component = metrics.confidence_score
        
        # 4. Consistency over time
        consistency_component = self._calculate_consistency(neighbor_id)
        
        # 5. Agreement with my predictions
        agreement_component = metrics.agreement_rate
        
        # Weighted combination
        trust_score = (
            0.3 * performance_component +    # Performance quality
            0.2 * loss_component +          # Loss quality  
            0.2 * confidence_component +    # Prediction confidence
            0.2 * consistency_component +   # Temporal consistency
            0.1 * agreement_component       # Prediction agreement
        )
        
        return np.clip(trust_score, 0.0, 1.0)
    
    def _calculate_consistency(self, neighbor_id: str) -> float:
        """
        Calculate temporal consistency of neighbor's performance.
        
        Args:
            neighbor_id: Neighbor ID
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        if neighbor_id not in self.neighbor_performance:
            return 1.0  # No history yet, assume consistent
        
        history = self.neighbor_performance[neighbor_id]
        if len(history) < 2:
            return 1.0
        
        # Calculate variance in accuracy over time
        accuracies = [m.accuracy for m in history]
        accuracy_std = np.std(accuracies)
        
        # Convert std to consistency score (lower std = higher consistency)
        # Use exponential decay: consistency = exp(-k * std)
        consistency = np.exp(-5.0 * accuracy_std)
        
        return np.clip(consistency, 0.0, 1.0)
    
    def _is_neighbor_suspicious(
        self, 
        neighbor_id: str, 
        metrics: PerformanceMetrics
    ) -> bool:
        """
        Determine if neighbor behavior is suspicious.
        
        Args:
            neighbor_id: Neighbor ID
            metrics: Current performance metrics
            
        Returns:
            True if neighbor is suspicious
        """
        # 1. Significant performance drop
        if self.baseline_accuracy > 0:
            performance_drop = self.baseline_accuracy - metrics.accuracy
            if performance_drop > self.performance_threshold:
                return True
        
        # 2. Very low absolute performance
        if metrics.accuracy < 0.5:  # Worse than random for binary classification
            return True
        
        # 3. Extremely high loss
        if metrics.loss > 3.0:  # Unreasonably high loss
            return True
        
        # 4. Low prediction confidence
        if metrics.confidence_score < 0.3:
            return True
        
        # 5. Inconsistent performance over time
        consistency = self._calculate_consistency(neighbor_id)
        if consistency < 0.3:
            return True
        
        return False
    
    def _compile_performance_stats(
        self, 
        neighbor_id: str, 
        metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """
        Compile detailed performance statistics.
        
        Args:
            neighbor_id: Neighbor ID
            metrics: Current performance metrics
            
        Returns:
            Dictionary of statistics
        """
        history = self.neighbor_performance.get(neighbor_id, deque())
        
        # Historical statistics
        if len(history) > 1:
            accuracies = [m.accuracy for m in history]
            losses = [m.loss for m in history]
            
            accuracy_trend = accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0.0
            loss_trend = losses[-1] - losses[0] if len(losses) > 1 else 0.0
        else:
            accuracy_trend = 0.0
            loss_trend = 0.0
        
        stats = {
            # Current metrics
            "current_accuracy": metrics.accuracy,
            "current_loss": metrics.loss,
            "prediction_entropy": metrics.prediction_entropy,
            "confidence_score": metrics.confidence_score,
            "agreement_rate": metrics.agreement_rate,
            
            # Baseline comparison
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_loss": self.baseline_loss,
            "performance_drop": self.baseline_accuracy - metrics.accuracy,
            "loss_increase": metrics.loss - self.baseline_loss,
            
            # Temporal patterns
            "accuracy_trend": accuracy_trend,
            "loss_trend": loss_trend,
            "consistency_score": self._calculate_consistency(neighbor_id),
            "evaluation_count": len(history),
            
            # Trust assessment
            "trust_score": self.trust_scores.get(neighbor_id, 1.0),
            "is_performance_degraded": (
                self.baseline_accuracy - metrics.accuracy > self.performance_threshold
            ),
            "evaluation_method": "performance_based",
        }
        
        return stats
    
    def get_neighbor_summary(self) -> Dict[str, Any]:
        """
        Get summary of all neighbor trust assessments.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "total_neighbors": len(self.trust_scores),
            "average_trust": np.mean(list(self.trust_scores.values())) if self.trust_scores else 1.0,
            "suspicious_count": sum(1 for score in self.trust_scores.values() if score < 0.5),
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_loss": self.baseline_loss,
            "test_samples": len(self.test_features) if self.test_features is not None else 0,
        }
        
        # Per-neighbor details
        neighbor_details = {}
        for neighbor_id, trust_score in self.trust_scores.items():
            history = self.neighbor_performance.get(neighbor_id, deque())
            if history:
                latest_metrics = history[-1]
                neighbor_details[neighbor_id] = {
                    "trust_score": trust_score,
                    "latest_accuracy": latest_metrics.accuracy,
                    "latest_loss": latest_metrics.loss,
                    "evaluation_count": len(history),
                    "consistency": self._calculate_consistency(neighbor_id),
                }
        
        summary["neighbors"] = neighbor_details
        return summary


class IntegratedTrustMonitor:
    """
    Combines HSIC-based parameter analysis with performance-based evaluation
    for comprehensive trust monitoring.
    """
    
    def __init__(
        self,
        node_id: str,
        hsic_weight: float = 0.3,
        performance_weight: float = 0.7,
        **kwargs
    ):
        """
        Initialize integrated trust monitor.
        
        Args:
            node_id: Node identifier
            hsic_weight: Weight for HSIC-based trust component
            performance_weight: Weight for performance-based trust component
            **kwargs: Additional arguments for component monitors
        """
        self.node_id = node_id
        self.hsic_weight = hsic_weight
        self.performance_weight = performance_weight
        
        # Component monitors
        from murmura.trust.hsic import ModelUpdateHSIC
        self.hsic_monitor = ModelUpdateHSIC(**kwargs)
        self.performance_monitor = PerformanceTrustMonitor(node_id, **kwargs)
        
        self.logger = logging.getLogger(f"murmura.trust.IntegratedTrust.{node_id}")
    
    def assess_comprehensive_trust(
        self,
        neighbor_id: str,
        my_params: Dict[str, np.ndarray],
        neighbor_params: Dict[str, np.ndarray]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Comprehensive trust assessment combining HSIC and performance.
        
        Args:
            neighbor_id: Neighbor identifier
            my_params: My current model parameters
            neighbor_params: Neighbor's model parameters
            
        Returns:
            Tuple of (combined_trust_score, is_suspicious, detailed_stats)
        """
        # HSIC-based assessment
        hsic_value, hsic_drift, hsic_stats = self.hsic_monitor.update_with_parameters(
            my_params, neighbor_params, neighbor_id
        )
        hsic_trust = 1.0 - hsic_value  # Convert HSIC to trust score
        
        # Performance-based assessment
        perf_trust, perf_suspicious, perf_stats = self.performance_monitor.assess_neighbor_trust(
            neighbor_id, neighbor_params
        )
        
        # Combine trust scores
        combined_trust = (
            self.hsic_weight * hsic_trust +
            self.performance_weight * perf_trust
        )
        
        # Combined suspicion detection
        is_suspicious = hsic_drift or perf_suspicious
        
        # Merge statistics
        combined_stats = {
            "combined_trust_score": combined_trust,
            "hsic_component": {
                "trust_score": hsic_trust,
                "hsic_value": hsic_value,
                "drift_detected": hsic_drift,
                **hsic_stats
            },
            "performance_component": {
                "trust_score": perf_trust,
                "suspicious": perf_suspicious,
                **perf_stats
            },
            "weights": {
                "hsic_weight": self.hsic_weight,
                "performance_weight": self.performance_weight
            }
        }
        
        return combined_trust, is_suspicious, combined_stats