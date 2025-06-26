"""
Model Evaluator for Performance-based Trust Assessment.

This module provides utilities to evaluate model parameters on local test data
for realistic performance-based trust monitoring in federated learning.
"""

import logging
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelEvaluator:
    """
    Utility class to evaluate model parameters on local test data.
    
    This enables realistic performance-based trust monitoring by actually
    testing how well a neighbor's model performs on local data.
    """
    
    def __init__(
        self,
        model_template: Optional[nn.Module] = None,
        device: str = "cpu",
        batch_size: int = 64,
    ):
        """
        Initialize model evaluator.
        
        Args:
            model_template: Template model architecture (will be copied), can be None
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
        """
        self.model_template = model_template
        self.device = device
        self.batch_size = batch_size
        self.logger = logging.getLogger("murmura.trust.ModelEvaluator")
        
        # Create evaluation model (copy of template) if provided
        if model_template is not None:
            self.eval_model = deepcopy(model_template).to(device)
            self.eval_model.eval()
        else:
            # Create MNIST model directly if no template provided
            from murmura.models.mnist_models import MNISTModel
            self.eval_model = MNISTModel().to(device)
            self.eval_model.eval()
    
    def evaluate_parameters_on_data(
        self,
        model_params: Dict[str, np.ndarray],
        test_features: np.ndarray,
        test_labels: np.ndarray,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model parameters on test data.
        
        Args:
            model_params: Model parameters to evaluate
            test_features: Test input features
            test_labels: Test labels
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Load parameters into evaluation model
            self._load_parameters_into_model(model_params)
            
            # Convert data to tensors
            if isinstance(test_features, np.ndarray):
                test_features = torch.from_numpy(test_features).float()
            if isinstance(test_labels, np.ndarray):
                test_labels = torch.from_numpy(test_labels).long()
            
            test_features = test_features.to(self.device)
            test_labels = test_labels.to(self.device)
            
            # Evaluate in batches
            total_correct = 0
            total_loss = 0.0
            total_samples = 0
            all_predictions = []
            all_confidences = []
            
            with torch.no_grad():
                for i in range(0, len(test_features), self.batch_size):
                    batch_features = test_features[i:i + self.batch_size]
                    batch_labels = test_labels[i:i + self.batch_size]
                    
                    # Forward pass
                    outputs = self.eval_model(batch_features)
                    
                    # Calculate loss
                    loss = F.cross_entropy(outputs, batch_labels, reduction='sum')
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == batch_labels).sum().item()
                    total_correct += correct
                    total_samples += batch_labels.size(0)
                    
                    if return_predictions:
                        # Store predictions and confidence scores
                        probs = F.softmax(outputs, dim=1)
                        max_probs, _ = torch.max(probs, 1)
                        
                        all_predictions.extend(predicted.cpu().numpy())
                        all_confidences.extend(max_probs.cpu().numpy())
            
            # Calculate final metrics
            accuracy = total_correct / total_samples
            avg_loss = total_loss / total_samples
            
            # Calculate prediction entropy (measure of confidence)
            if all_confidences:
                avg_confidence = np.mean(all_confidences)
                # Entropy approximation: higher confidence = lower entropy
                prediction_entropy = -np.mean([
                    conf * np.log(conf + 1e-8) + (1-conf) * np.log(1-conf + 1e-8)
                    for conf in all_confidences
                ])
            else:
                avg_confidence = 0.0
                prediction_entropy = 0.0
            
            results = {
                "accuracy": accuracy,
                "loss": avg_loss,
                "confidence": avg_confidence,
                "prediction_entropy": prediction_entropy,
                "total_samples": total_samples,
                "evaluation_successful": True,
            }
            
            if return_predictions:
                results["predictions"] = all_predictions
                results["confidences"] = all_confidences
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {e}")
            return {
                "accuracy": 0.0,
                "loss": float('inf'),
                "confidence": 0.0,
                "prediction_entropy": 0.0,
                "total_samples": 0,
                "evaluation_successful": False,
                "error": str(e)
            }
    
    def _load_parameters_into_model(self, model_params: Dict[str, np.ndarray]) -> None:
        """
        Load numpy parameters into PyTorch model.
        
        Args:
            model_params: Dictionary of parameter name -> numpy array
        """
        state_dict = {}
        
        for name, param_array in model_params.items():
            # Convert numpy to tensor
            if isinstance(param_array, np.ndarray):
                param_tensor = torch.from_numpy(param_array).float()
            else:
                param_tensor = param_array
            
            state_dict[name] = param_tensor
        
        # Load into model
        self.eval_model.load_state_dict(state_dict, strict=False)
    
    def compare_model_predictions(
        self,
        params1: Dict[str, np.ndarray],
        params2: Dict[str, np.ndarray],
        test_features: np.ndarray,
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compare predictions between two sets of model parameters.
        
        Args:
            params1: First model parameters
            params2: Second model parameters  
            test_features: Test input features
            sample_size: Number of samples to use (None for all)
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Sample data if requested
            if sample_size and len(test_features) > sample_size:
                indices = np.random.choice(len(test_features), sample_size, replace=False)
                test_features = test_features[indices]
            
            # Get predictions from both models
            results1 = self.evaluate_parameters_on_data(
                params1, test_features, np.zeros(len(test_features)), return_predictions=True
            )
            results2 = self.evaluate_parameters_on_data(
                params2, test_features, np.zeros(len(test_features)), return_predictions=True
            )
            
            if not (results1["evaluation_successful"] and results2["evaluation_successful"]):
                return {"agreement_rate": 0.0, "confidence_similarity": 0.0, "comparison_successful": False}
            
            pred1 = np.array(results1["predictions"])
            pred2 = np.array(results2["predictions"])
            conf1 = np.array(results1["confidences"])
            conf2 = np.array(results2["confidences"])
            
            # Calculate agreement rate
            agreement_rate = np.mean(pred1 == pred2)
            
            # Calculate confidence similarity (correlation)
            if len(conf1) > 1 and len(conf2) > 1:
                confidence_correlation = np.corrcoef(conf1, conf2)[0, 1]
                if np.isnan(confidence_correlation):
                    confidence_correlation = 0.0
            else:
                confidence_correlation = 0.0
            
            return {
                "agreement_rate": agreement_rate,
                "confidence_similarity": confidence_correlation,
                "avg_confidence_diff": np.mean(np.abs(conf1 - conf2)),
                "prediction_entropy_diff": abs(results1["prediction_entropy"] - results2["prediction_entropy"]),
                "comparison_successful": True,
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing predictions: {e}")
            return {
                "agreement_rate": 0.0,
                "confidence_similarity": 0.0,
                "comparison_successful": False,
                "error": str(e)
            }
    
    def calculate_gradient_similarity(
        self,
        params1: Dict[str, np.ndarray],
        params2: Dict[str, np.ndarray],
        prev_params1: Optional[Dict[str, np.ndarray]] = None,
        prev_params2: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Calculate similarity in gradient directions between two models.
        
        Args:
            params1: Current parameters of model 1
            params2: Current parameters of model 2
            prev_params1: Previous parameters of model 1 (for gradient calculation)
            prev_params2: Previous parameters of model 2 (for gradient calculation)
            
        Returns:
            Dictionary with gradient similarity metrics
        """
        try:
            # Calculate gradients (parameter changes)
            if prev_params1 is not None and prev_params2 is not None:
                grad1 = {k: params1[k] - prev_params1[k] for k in params1 if k in prev_params1}
                grad2 = {k: params2[k] - prev_params2[k] for k in params2 if k in prev_params2}
            else:
                # If no previous parameters, use current parameters as "gradients"
                grad1 = params1
                grad2 = params2
            
            # Flatten gradients
            flat_grad1 = np.concatenate([g.flatten() for g in grad1.values()])
            flat_grad2 = np.concatenate([g.flatten() for g in grad2.values()])
            
            # Calculate similarity metrics
            cosine_similarity = np.dot(flat_grad1, flat_grad2) / (
                np.linalg.norm(flat_grad1) * np.linalg.norm(flat_grad2) + 1e-8
            )
            
            # L2 distance (normalized)
            l2_distance = np.linalg.norm(flat_grad1 - flat_grad2)
            max_norm = max(np.linalg.norm(flat_grad1), np.linalg.norm(flat_grad2))
            normalized_l2_distance = l2_distance / (max_norm + 1e-8)
            
            # Magnitude similarity
            magnitude_ratio = np.linalg.norm(flat_grad1) / (np.linalg.norm(flat_grad2) + 1e-8)
            magnitude_similarity = min(magnitude_ratio, 1.0 / magnitude_ratio)
            
            return {
                "cosine_similarity": cosine_similarity,
                "l2_distance": l2_distance,
                "normalized_l2_distance": normalized_l2_distance,
                "magnitude_similarity": magnitude_similarity,
                "gradient_analysis_successful": True,
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating gradient similarity: {e}")
            return {
                "cosine_similarity": 0.0,
                "l2_distance": float('inf'),
                "normalized_l2_distance": 1.0,
                "magnitude_similarity": 0.0,
                "gradient_analysis_successful": False,
                "error": str(e)
            }


class EnhancedPerformanceTrustMonitor:
    """
    Enhanced performance-based trust monitor that uses actual model evaluation.
    """
    
    def __init__(
        self,
        node_id: str,
        model_template: nn.Module,
        performance_threshold: float = 0.05,
        window_size: int = 10,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize enhanced performance trust monitor.
        
        Args:
            node_id: Node identifier
            model_template: Model architecture for evaluation
            performance_threshold: Performance drop threshold
            window_size: History window size
            device: Device for evaluation
        """
        self.node_id = node_id
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        
        # Model evaluator
        self.evaluator = ModelEvaluator(model_template, device)
        
        # Test data (set externally)
        self.test_features: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        
        # Baseline and history
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.neighbor_history: Dict[str, list] = {}
        self.previous_params: Dict[str, Dict[str, np.ndarray]] = {}
        
        self.logger = logging.getLogger(f"murmura.trust.EnhancedPerformanceTrust.{node_id}")
    
    def set_test_data(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Set test data for evaluation."""
        self.test_features = features
        self.test_labels = labels
        self.logger.info(f"Set test data: {len(features)} samples")
    
    def set_baseline(self, my_params: Dict[str, np.ndarray]) -> None:
        """Establish baseline performance with my parameters."""
        if self.test_features is None or self.test_labels is None:
            self.logger.warning("No test data available for baseline")
            return
        
        self.baseline_metrics = self.evaluator.evaluate_parameters_on_data(
            my_params, self.test_features, self.test_labels
        )
        
        self.logger.info(
            f"Baseline established: Accuracy={self.baseline_metrics['accuracy']:.4f}, "
            f"Loss={self.baseline_metrics['loss']:.4f}"
        )
    
    def assess_neighbor_performance(
        self,
        neighbor_id: str,
        neighbor_params: Dict[str, np.ndarray],
        my_params: Dict[str, np.ndarray]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Comprehensive performance-based trust assessment.
        
        Args:
            neighbor_id: Neighbor identifier
            neighbor_params: Neighbor's model parameters
            my_params: My current model parameters
            
        Returns:
            Tuple of (trust_score, is_suspicious, detailed_stats)
        """
        if self.test_features is None or self.test_labels is None:
            return 1.0, False, {"error": "no_test_data"}
        
        try:
            # Evaluate neighbor's performance
            neighbor_metrics = self.evaluator.evaluate_parameters_on_data(
                neighbor_params, self.test_features, self.test_labels, return_predictions=True
            )
            
            if not neighbor_metrics["evaluation_successful"]:
                return 0.0, True, {"error": "evaluation_failed"}
            
            # Compare with my predictions
            comparison_metrics = self.evaluator.compare_model_predictions(
                my_params, neighbor_params, self.test_features, sample_size=500
            )
            
            # Calculate gradient similarity if we have previous parameters
            gradient_metrics = {}
            if neighbor_id in self.previous_params:
                gradient_metrics = self.evaluator.calculate_gradient_similarity(
                    my_params, neighbor_params,
                    self.previous_params.get("my_params"), self.previous_params.get(neighbor_id)
                )
            
            # Store current parameters for next comparison
            self.previous_params[neighbor_id] = neighbor_params.copy()
            self.previous_params["my_params"] = my_params.copy()
            
            # Calculate comprehensive trust score
            trust_score = self._calculate_comprehensive_trust(
                neighbor_metrics, comparison_metrics, gradient_metrics
            )
            
            # Determine suspicion
            is_suspicious = self._is_performance_suspicious(neighbor_metrics, comparison_metrics)
            
            # Store history
            if neighbor_id not in self.neighbor_history:
                self.neighbor_history[neighbor_id] = []
            
            self.neighbor_history[neighbor_id].append({
                "metrics": neighbor_metrics,
                "comparison": comparison_metrics,
                "gradient": gradient_metrics,
                "trust_score": trust_score,
                "suspicious": is_suspicious
            })
            
            # Keep only recent history
            if len(self.neighbor_history[neighbor_id]) > self.window_size:
                self.neighbor_history[neighbor_id].pop(0)
            
            # Compile detailed stats
            stats = {
                **neighbor_metrics,
                **comparison_metrics,
                **gradient_metrics,
                "trust_score": trust_score,
                "baseline_comparison": self._compare_with_baseline(neighbor_metrics),
                "temporal_consistency": self._calculate_temporal_consistency(neighbor_id),
                "evaluation_method": "enhanced_performance",
            }
            
            return trust_score, is_suspicious, stats
            
        except Exception as e:
            self.logger.error(f"Error in performance assessment: {e}")
            return 0.0, True, {"error": str(e)}
    
    def _calculate_comprehensive_trust(
        self,
        neighbor_metrics: Dict[str, Any],
        comparison_metrics: Dict[str, Any],
        gradient_metrics: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive trust score from multiple metrics."""
        
        # Performance quality (0.4 weight)
        if self.baseline_metrics:
            performance_ratio = neighbor_metrics["accuracy"] / max(self.baseline_metrics["accuracy"], 0.01)
            performance_component = min(performance_ratio, 1.0)
        else:
            performance_component = neighbor_metrics["accuracy"]
        
        # Prediction agreement (0.3 weight)
        agreement_component = comparison_metrics.get("agreement_rate", 0.0)
        
        # Gradient alignment (0.2 weight)
        gradient_component = max(gradient_metrics.get("cosine_similarity", 0.0), 0.0)
        
        # Confidence similarity (0.1 weight)
        confidence_component = comparison_metrics.get("confidence_similarity", 0.0)
        confidence_component = max(confidence_component, 0.0)  # Handle negative correlations
        
        trust_score = (
            0.4 * performance_component +
            0.3 * agreement_component +
            0.2 * gradient_component +
            0.1 * confidence_component
        )
        
        return np.clip(trust_score, 0.0, 1.0)
    
    def _is_performance_suspicious(
        self,
        neighbor_metrics: Dict[str, Any],
        comparison_metrics: Dict[str, Any]
    ) -> bool:
        """Determine if performance indicates suspicious behavior."""
        
        # Significant performance drop
        if self.baseline_metrics:
            performance_drop = self.baseline_metrics["accuracy"] - neighbor_metrics["accuracy"]
            if performance_drop > self.performance_threshold:
                return True
        
        # Very low absolute performance
        if neighbor_metrics["accuracy"] < 0.3:
            return True
        
        # Very low prediction agreement
        if comparison_metrics.get("agreement_rate", 1.0) < 0.7:
            return True
        
        # Unreasonably high loss
        if neighbor_metrics["loss"] > 5.0:
            return True
        
        return False
    
    def _compare_with_baseline(self, neighbor_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Compare neighbor metrics with baseline."""
        if not self.baseline_metrics:
            return {}
        
        return {
            "accuracy_drop": self.baseline_metrics["accuracy"] - neighbor_metrics["accuracy"],
            "loss_increase": neighbor_metrics["loss"] - self.baseline_metrics["loss"],
            "confidence_diff": neighbor_metrics["confidence"] - self.baseline_metrics["confidence"],
        }
    
    def _calculate_temporal_consistency(self, neighbor_id: str) -> float:
        """Calculate consistency of neighbor's performance over time."""
        if neighbor_id not in self.neighbor_history or len(self.neighbor_history[neighbor_id]) < 2:
            return 1.0
        
        history = self.neighbor_history[neighbor_id]
        accuracies = [h["metrics"]["accuracy"] for h in history]
        
        # Calculate coefficient of variation (std/mean)
        if len(accuracies) > 1:
            cv = np.std(accuracies) / (np.mean(accuracies) + 1e-8)
            consistency = np.exp(-5.0 * cv)  # Convert CV to consistency score
            return np.clip(consistency, 0.0, 1.0)
        
        return 1.0