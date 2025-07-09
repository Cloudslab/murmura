"""
Statistical Trust Detector for Decentralized Federated Learning.

This module implements a robust statistical approach to detect trust drift in 
decentralized federated learning without relying on HSIC. The approach is based
on analyzing patterns in relative parameter differences between honest and 
malicious nodes discovered through pattern analysis.

Key Features:
- Dataset and topology agnostic
- Works independently per node (no consensus)
- Robust against non-IID data distributions
- Detects gradual attacks through multi-dimensional statistical analysis
- Lightweight and computationally efficient
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest


class DetectionSignal(Enum):
    """Types of statistical signals for trust detection."""
    PARAMETER_MAGNITUDE = "parameter_magnitude"
    RELATIVE_DIFFERENCE = "relative_difference"
    COSINE_SIMILARITY = "cosine_similarity"
    GRADIENT_NORM_RATIO = "gradient_norm_ratio"
    STATISTICAL_OUTLIER = "statistical_outlier"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


@dataclass
class StatisticalFeatures:
    """Statistical features extracted from parameter updates."""
    # Basic parameter metrics
    total_param_norm: float
    total_diff_norm: float
    relative_diff_norm: float
    
    # Layer-wise patterns
    layer_norm_distribution: List[float]
    layer_similarity_distribution: List[float]
    
    # Advanced statistical measures
    parameter_entropy: float
    norm_ratio_variance: float
    cosine_similarity_stats: Dict[str, float]
    
    # Temporal features
    update_frequency: float
    consistency_score: float
    trend_deviation: float


class RobustStatisticalTrustDetector:
    """
    Robust statistical trust detector based on parameter difference analysis.
    
    This detector uses multi-dimensional statistical analysis to identify
    malicious behavior patterns without relying on HSIC or consensus mechanisms.
    
    IMPORTANT: This detector operates in a completely unsupervised manner.
    It has NO access to ground truth labels during detection decisions.
    Ground truth is only used for validation, testing, and performance evaluation.
    
    The detector relies purely on statistical anomaly detection based on:
    - Parameter magnitude differences
    - Relative parameter changes
    - Cosine similarity patterns
    - Statistical outlier detection (Isolation Forest)
    - Temporal consistency analysis
    """
    
    def __init__(
        self,
        node_id: str,
        window_size: int = 20,
        min_samples_for_detection: int = 5,
        outlier_contamination: float = 0.1,
        enable_adaptive_thresholds: bool = True,
        topology: str = "ring",
        relative_difference_threshold: float = 1.0,
        cosine_similarity_threshold: float = 0.9,
        parameter_magnitude_threshold: float = 1.5,
        statistical_outlier_threshold: float = -0.02,
        temporal_consistency_threshold: float = 0.75
    ):
        """
        Initialize the statistical trust detector.
        
        Args:
            node_id: ID of the node this detector belongs to
            window_size: Size of sliding window for statistical analysis
            min_samples_for_detection: Minimum samples before making detection decisions
            outlier_contamination: Expected fraction of outliers for isolation forest
            enable_adaptive_thresholds: Whether to use adaptive statistical thresholds
            topology: Network topology for optimization (ring, complete, line)
            relative_difference_threshold: Z-score threshold for relative parameter differences
            cosine_similarity_threshold: Minimum cosine similarity threshold
            parameter_magnitude_threshold: Z-score threshold for parameter magnitude changes
            statistical_outlier_threshold: Isolation forest threshold for outlier detection
            temporal_consistency_threshold: Minimum temporal consistency threshold
        """
        self.node_id = node_id
        self.window_size = window_size
        self.min_samples_for_detection = min_samples_for_detection
        self.outlier_contamination = outlier_contamination
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.topology = topology
        
        # Statistical models and data storage
        self.neighbor_features: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.baseline_statistics: Dict[str, Dict[str, float]] = {}
        self.isolation_forests: Dict[str, IsolationForest] = {}
        
        # Detection thresholds (configurable, optimized for gradual attacks)
        self.detection_thresholds = {
            DetectionSignal.RELATIVE_DIFFERENCE.value: relative_difference_threshold,
            DetectionSignal.COSINE_SIMILARITY.value: cosine_similarity_threshold,
            DetectionSignal.PARAMETER_MAGNITUDE.value: parameter_magnitude_threshold,
            DetectionSignal.STATISTICAL_OUTLIER.value: statistical_outlier_threshold,
            DetectionSignal.TEMPORAL_CONSISTENCY.value: temporal_consistency_threshold,
        }
        
        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.stability_factor = 0.95
        
        # Topology-specific adjustments
        self._adjust_for_topology()
        
        # Statistical state
        self.update_count: Dict[str, int] = defaultdict(int)
        self.detection_count: Dict[str, int] = defaultdict(int)
        self.false_positive_estimates: Dict[str, float] = defaultdict(float)
        
        # Robustness features
        self.robust_scalers: Dict[str, RobustScaler] = {}
        self.baseline_update_count = 0
        self.baseline_established = False
        
        self.logger = logging.getLogger(f"murmura.trust.StatisticalDetector.{node_id}")
        self.logger.info(f"Initialized robust statistical trust detector for topology: {topology}")
    
    def _adjust_for_topology(self) -> None:
        """Adjust detection parameters based on network topology."""
        if self.topology == "complete":
            # Complete graph: More neighbors, can be stricter
            self.detection_thresholds[DetectionSignal.RELATIVE_DIFFERENCE] = 2.0
            self.min_samples_for_detection = max(3, self.min_samples_for_detection - 2)
            self.outlier_contamination = 0.15  # Expect more diversity
        elif self.topology == "line":
            # Line topology: Fewer neighbors, be more permissive
            self.detection_thresholds[DetectionSignal.RELATIVE_DIFFERENCE] = 3.0
            self.min_samples_for_detection = min(8, self.min_samples_for_detection + 2)
            self.outlier_contamination = 0.05  # Less diversity expected
        # Ring topology uses default values
    
    def analyze_parameter_update(
        self,
        neighbor_id: str,
        current_params: Dict[str, np.ndarray],
        neighbor_params: Dict[str, np.ndarray],
        round_number: int
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyze a parameter update for trust drift using statistical methods.
        
        Args:
            neighbor_id: ID of the neighbor sending the update
            current_params: Current node's parameters
            neighbor_params: Neighbor's parameters
            round_number: Current federated learning round
            
        Returns:
            Tuple of (is_malicious, suspicion_score, detailed_analysis)
        """
        # Extract statistical features
        features = self._extract_statistical_features(
            current_params, neighbor_params, round_number
        )
        
        # Store features for baseline and analysis
        self.neighbor_features[neighbor_id].append({
            'features': features,
            'round': round_number,
            'timestamp': time.time()
        })
        self.update_count[neighbor_id] += 1
        
        # Build baseline if needed
        if not self.baseline_established:
            self._update_baseline_statistics()
        
        # Perform multi-dimensional analysis
        detection_results = self._perform_statistical_analysis(neighbor_id, features)
        
        # Combine signals for final decision
        is_malicious, suspicion_score, confidence = self._combine_detection_signals(
            neighbor_id, detection_results
        )
        
        # Update statistical models
        self._update_statistical_models(neighbor_id, features, is_malicious)
        
        # Compile detailed analysis with JSON-serializable detection signals
        serializable_detection_results = {
            signal.value: result for signal, result in detection_results.items()
        }
        
        detailed_analysis = {
            'neighbor_id': neighbor_id,
            'round': round_number,
            'suspicion_score': suspicion_score,
            'confidence': confidence,
            'is_malicious': is_malicious,
            'features': self._serialize_features(features),
            'detection_signals': serializable_detection_results,
            'baseline_stats': self.baseline_statistics.get(neighbor_id, {}),
            'update_count': self.update_count[neighbor_id],
            'detection_count': self.detection_count[neighbor_id],
            'false_positive_estimate': self.false_positive_estimates[neighbor_id],
            'method': 'robust_statistical'
        }
        
        if is_malicious:
            self.detection_count[neighbor_id] += 1
            self.logger.warning(
                f"🚨 STATISTICAL DETECTION: {neighbor_id} flagged as malicious "
                f"(suspicion: {suspicion_score:.3f}, confidence: {confidence:.3f}) "
                f"in round {round_number}"
            )
        
        return is_malicious, suspicion_score, detailed_analysis
    
    def _extract_statistical_features(
        self,
        current_params: Dict[str, np.ndarray],
        neighbor_params: Dict[str, np.ndarray],
        round_number: int
    ) -> StatisticalFeatures:
        """Extract comprehensive statistical features from parameter updates."""
        # Basic parameter metrics
        total_param_norm = 0.0
        total_diff_norm = 0.0
        layer_norms = []
        layer_similarities = []
        cosine_similarities = []
        
        for layer_name, neighbor_param in neighbor_params.items():
            if layer_name in current_params:
                current_param = current_params[layer_name]
                
                # Layer-specific calculations
                neighbor_norm = np.linalg.norm(neighbor_param.flatten())
                current_norm = np.linalg.norm(current_param.flatten())
                diff = neighbor_param - current_param
                diff_norm = np.linalg.norm(diff.flatten())
                
                # Cosine similarity
                if neighbor_norm > 1e-8 and current_norm > 1e-8:
                    cosine_sim = np.dot(neighbor_param.flatten(), current_param.flatten()) / (
                        neighbor_norm * current_norm
                    )
                    cosine_similarities.append(cosine_sim)
                
                # Accumulate totals
                total_param_norm += neighbor_norm
                total_diff_norm += diff_norm
                layer_norms.append(neighbor_norm)
                layer_similarities.append(cosine_sim if cosine_similarities else 0.0)
        
        # Relative metrics
        relative_diff_norm = total_diff_norm / max(total_param_norm, 1e-8)
        
        # Advanced statistical measures
        parameter_entropy = self._calculate_parameter_entropy(neighbor_params)
        norm_ratio_variance = np.var(layer_norms) if layer_norms else 0.0
        
        # Cosine similarity statistics
        cosine_stats = {
            'mean': np.mean(cosine_similarities) if cosine_similarities else 0.0,
            'std': np.std(cosine_similarities) if cosine_similarities else 0.0,
            'min': np.min(cosine_similarities) if cosine_similarities else 0.0,
            'median': np.median(cosine_similarities) if cosine_similarities else 0.0
        }
        
        # Temporal features (placeholder - would need history)
        update_frequency = 1.0  # Can be computed from timing history
        consistency_score = 1.0  # Placeholder for temporal consistency
        trend_deviation = 0.0   # Placeholder for trend analysis
        
        return StatisticalFeatures(
            total_param_norm=total_param_norm,
            total_diff_norm=total_diff_norm,
            relative_diff_norm=relative_diff_norm,
            layer_norm_distribution=layer_norms,
            layer_similarity_distribution=layer_similarities,
            parameter_entropy=parameter_entropy,
            norm_ratio_variance=norm_ratio_variance,
            cosine_similarity_stats=cosine_stats,
            update_frequency=update_frequency,
            consistency_score=consistency_score,
            trend_deviation=trend_deviation
        )
    
    def _calculate_parameter_entropy(self, params: Dict[str, np.ndarray]) -> float:
        """Calculate entropy of parameter distribution as a complexity measure."""
        all_params = []
        for param_array in params.values():
            all_params.extend(param_array.flatten())
        
        if len(all_params) == 0:
            return 0.0
        
        # Discretize parameters for entropy calculation
        param_array = np.array(all_params)
        hist, _ = np.histogram(param_array, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        return float(entropy)
    
    def _perform_statistical_analysis(
        self,
        neighbor_id: str,
        features: StatisticalFeatures
    ) -> Dict[DetectionSignal, Dict[str, Any]]:
        """Perform multi-dimensional statistical analysis."""
        results = {}
        
        # Perform some analysis even with few samples, but be less strict
        limited_analysis = self.update_count[neighbor_id] < self.min_samples_for_detection
        
        # 1. Relative difference analysis
        results[DetectionSignal.RELATIVE_DIFFERENCE] = self._analyze_relative_difference(
            neighbor_id, features
        )
        
        # 2. Cosine similarity analysis
        results[DetectionSignal.COSINE_SIMILARITY] = self._analyze_cosine_similarity(
            neighbor_id, features
        )
        
        # 3. Parameter magnitude analysis
        results[DetectionSignal.PARAMETER_MAGNITUDE] = self._analyze_parameter_magnitude(
            neighbor_id, features
        )
        
        # 4. Statistical outlier detection
        results[DetectionSignal.STATISTICAL_OUTLIER] = self._analyze_statistical_outliers(
            neighbor_id, features
        )
        
        # 5. Temporal consistency analysis
        results[DetectionSignal.TEMPORAL_CONSISTENCY] = self._analyze_temporal_consistency(
            neighbor_id, features
        )
        
        return results
    
    def _analyze_relative_difference(
        self,
        neighbor_id: str,
        features: StatisticalFeatures
    ) -> Dict[str, Any]:
        """Analyze relative parameter differences for anomalies."""
        current_diff = features.relative_diff_norm
        
        # Use simple threshold for early detection if no baseline
        if neighbor_id not in self.baseline_statistics or len(self.neighbor_features[neighbor_id]) < 3:
            # Simple threshold-based detection for early rounds (balanced)
            simple_threshold = 0.05  # If relative difference > 5%, flag as suspicious
            detected = current_diff > simple_threshold
            score = min(current_diff / simple_threshold, 2.0) if detected else 0.0
            
            return {
                'detected': detected,
                'score': score,
                'current_value': current_diff,
                'threshold': simple_threshold,
                'reason': f'simple_threshold={current_diff:.3f} vs {simple_threshold}',
                'method': 'simple_threshold'
            }
        
        # Statistical analysis with baseline
        baseline = self.baseline_statistics[neighbor_id]
        
        # Z-score analysis
        mean_diff = baseline.get('mean_relative_diff', current_diff)
        std_diff = baseline.get('std_relative_diff', 0.1)
        
        if std_diff < 1e-6:  # Avoid division by zero
            z_score = 0.0
        else:
            z_score = abs(current_diff - mean_diff) / std_diff
        
        # Detection logic
        threshold = self.detection_thresholds[DetectionSignal.RELATIVE_DIFFERENCE.value]
        detected = z_score > threshold
        
        return {
            'detected': detected,
            'score': min(z_score / threshold, 3.0),  # Normalize to 0-3 range
            'z_score': z_score,
            'current_value': current_diff,
            'baseline_mean': mean_diff,
            'baseline_std': std_diff,
            'threshold': threshold,
            'reason': f'z_score={z_score:.2f} vs threshold={threshold}',
            'method': 'statistical'
        }
    
    def _analyze_cosine_similarity(
        self,
        neighbor_id: str,
        features: StatisticalFeatures
    ) -> Dict[str, Any]:
        """Analyze cosine similarity for trust assessment."""
        mean_similarity = features.cosine_similarity_stats['mean']
        min_similarity = features.cosine_similarity_stats['min']
        
        # Detection based on minimum similarity threshold
        threshold = self.detection_thresholds[DetectionSignal.COSINE_SIMILARITY.value]
        detected = mean_similarity < threshold
        
        # Score based on how far below threshold
        if detected:
            score = (threshold - mean_similarity) / threshold
        else:
            score = 0.0
        
        return {
            'detected': detected,
            'score': score,
            'mean_similarity': mean_similarity,
            'min_similarity': min_similarity,
            'threshold': threshold,
            'reason': f'mean_sim={mean_similarity:.3f} vs threshold={threshold}'
        }
    
    def _analyze_parameter_magnitude(
        self,
        neighbor_id: str,
        features: StatisticalFeatures
    ) -> Dict[str, Any]:
        """Analyze parameter magnitude for anomalies."""
        if neighbor_id not in self.baseline_statistics:
            return {'detected': False, 'score': 0.0, 'reason': 'no_baseline'}
        
        baseline = self.baseline_statistics[neighbor_id]
        current_norm = features.total_param_norm
        
        # Z-score analysis for parameter magnitude
        mean_norm = baseline.get('mean_param_norm', current_norm)
        std_norm = baseline.get('std_param_norm', current_norm * 0.1)
        
        if std_norm < 1e-6:
            z_score = 0.0
        else:
            z_score = abs(current_norm - mean_norm) / std_norm
        
        threshold = self.detection_thresholds[DetectionSignal.PARAMETER_MAGNITUDE.value]
        detected = z_score > threshold
        
        return {
            'detected': detected,
            'score': min(z_score / threshold, 2.0),
            'z_score': z_score,
            'current_norm': current_norm,
            'baseline_mean': mean_norm,
            'baseline_std': std_norm,
            'threshold': threshold,
            'reason': f'norm_z_score={z_score:.2f} vs threshold={threshold}'
        }
    
    def _analyze_statistical_outliers(
        self,
        neighbor_id: str,
        features: StatisticalFeatures
    ) -> Dict[str, Any]:
        """Use isolation forest for outlier detection."""
        if neighbor_id not in self.isolation_forests:
            return {'detected': False, 'score': 0.0, 'reason': 'model_not_ready'}
        
        # Prepare feature vector
        feature_vector = np.array([[
            features.total_param_norm,
            features.total_diff_norm,
            features.relative_diff_norm,
            features.parameter_entropy,
            features.norm_ratio_variance,
            features.cosine_similarity_stats['mean'],
            features.cosine_similarity_stats['std']
        ]])
        
        # Get outlier score
        model = self.isolation_forests[neighbor_id]
        outlier_score = model.decision_function(feature_vector)[0]
        is_outlier = model.predict(feature_vector)[0] == -1
        
        threshold = self.detection_thresholds[DetectionSignal.STATISTICAL_OUTLIER.value]
        detected = outlier_score < threshold
        
        return {
            'detected': detected,
            'score': max(0.0, -outlier_score),  # Convert to positive score
            'outlier_score': outlier_score,
            'is_outlier': is_outlier,
            'threshold': threshold,
            'reason': f'outlier_score={outlier_score:.3f} vs threshold={threshold}'
        }
    
    def _analyze_temporal_consistency(
        self,
        neighbor_id: str,
        features: StatisticalFeatures
    ) -> Dict[str, Any]:
        """Analyze temporal consistency of updates."""
        # Placeholder for temporal analysis
        # In a full implementation, this would analyze consistency over time
        
        consistency_score = features.consistency_score
        threshold = self.detection_thresholds[DetectionSignal.TEMPORAL_CONSISTENCY.value]
        detected = consistency_score < threshold
        
        return {
            'detected': detected,
            'score': max(0.0, threshold - consistency_score) if detected else 0.0,
            'consistency_score': consistency_score,
            'threshold': threshold,
            'reason': f'consistency={consistency_score:.3f} vs threshold={threshold}'
        }
    
    def _combine_detection_signals(
        self,
        neighbor_id: str,
        detection_results: Dict[DetectionSignal, Dict[str, Any]]
    ) -> Tuple[bool, float, float]:
        """Combine multiple detection signals for final decision."""
        # Weighted combination of detection signals
        signal_weights = {
            DetectionSignal.RELATIVE_DIFFERENCE.value: 0.3,
            DetectionSignal.COSINE_SIMILARITY.value: 0.25,
            DetectionSignal.PARAMETER_MAGNITUDE.value: 0.2,
            DetectionSignal.STATISTICAL_OUTLIER.value: 0.15,
            DetectionSignal.TEMPORAL_CONSISTENCY.value: 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        detection_count = 0
        
        for signal_value, weight in signal_weights.items():
            # Convert signal value back to enum for lookup
            signal_enum = None
            for enum_val in DetectionSignal:
                if enum_val.value == signal_value:
                    signal_enum = enum_val
                    break
                    
            if signal_enum and signal_enum in detection_results:
                result = detection_results[signal_enum]
                score = result.get('score', 0.0)
                total_score += weight * score
                total_weight += weight
                
                if result.get('detected', False):
                    detection_count += 1
        
        # Normalize suspicion score
        suspicion_score = total_score / max(total_weight, 1e-8)
        
        # Confidence based on agreement between signals
        confidence = detection_count / len(signal_weights)
        
        # Final decision: sensitive for gradual attack detection
        is_malicious = (
            suspicion_score > 0.2 or  # Low threshold for gradual attacks
            (detection_count >= 2 and suspicion_score > 0.1)  # Multiple signals with low suspicion
        )
        
        return is_malicious, float(suspicion_score), float(confidence)
    
    def _update_baseline_statistics(self) -> None:
        """Update baseline statistics from collected data."""
        self.baseline_update_count += 1
        
        for neighbor_id, feature_history in self.neighbor_features.items():
            if len(feature_history) >= 3:  # Need minimum samples
                # Extract features for analysis
                relative_diffs = [f['features'].relative_diff_norm for f in feature_history]
                param_norms = [f['features'].total_param_norm for f in feature_history]
                
                # Update baseline statistics
                self.baseline_statistics[neighbor_id] = {
                    'mean_relative_diff': np.mean(relative_diffs),
                    'std_relative_diff': max(np.std(relative_diffs), 0.01),  # Minimum std
                    'mean_param_norm': np.mean(param_norms),
                    'std_param_norm': max(np.std(param_norms), np.mean(param_norms) * 0.05),
                    'sample_count': len(feature_history)
                }
        
        # Mark baseline as established after sufficient updates
        if self.baseline_update_count >= 3:
            self.baseline_established = True
    
    def _update_statistical_models(
        self,
        neighbor_id: str,
        features: StatisticalFeatures,
        is_malicious: bool
    ) -> None:
        """Update statistical models with new data (unsupervised only)."""
        # Update isolation forest if we have enough data
        if len(self.neighbor_features[neighbor_id]) >= 5:
            # Prepare training data
            training_data = []
            for f_data in self.neighbor_features[neighbor_id]:
                f = f_data['features']
                training_data.append([
                    f.total_param_norm,
                    f.total_diff_norm,
                    f.relative_diff_norm,
                    f.parameter_entropy,
                    f.norm_ratio_variance,
                    f.cosine_similarity_stats['mean'],
                    f.cosine_similarity_stats['std']
                ])
            
            # Train/update isolation forest (unsupervised)
            if neighbor_id not in self.isolation_forests:
                self.isolation_forests[neighbor_id] = IsolationForest(
                    contamination=self.outlier_contamination,
                    random_state=42
                )
            
            try:
                self.isolation_forests[neighbor_id].fit(training_data)
            except Exception as e:
                self.logger.warning(f"Failed to update isolation forest for {neighbor_id}: {e}")
        
        # NOTE: No supervised learning here - all detection is purely unsupervised
        # Ground truth is only used for validation/testing, never for model updates
    
    def _serialize_features(self, features: StatisticalFeatures) -> Dict[str, Any]:
        """Serialize features for logging/analysis."""
        return {
            'total_param_norm': features.total_param_norm,
            'total_diff_norm': features.total_diff_norm,
            'relative_diff_norm': features.relative_diff_norm,
            'parameter_entropy': features.parameter_entropy,
            'norm_ratio_variance': features.norm_ratio_variance,
            'cosine_similarity_mean': features.cosine_similarity_stats['mean'],
            'cosine_similarity_std': features.cosine_similarity_stats['std'],
            'layer_count': len(features.layer_norm_distribution),
            'consistency_score': features.consistency_score
        }
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get comprehensive detection summary."""
        return {
            'node_id': self.node_id,
            'topology': self.topology,
            'total_neighbors': len(self.neighbor_features),
            'total_updates_analyzed': sum(self.update_count.values()),
            'total_detections': sum(self.detection_count.values()),
            'baseline_established': self.baseline_established,
            'detection_rate_by_neighbor': {
                neighbor_id: self.detection_count[neighbor_id] / max(self.update_count[neighbor_id], 1)
                for neighbor_id in self.update_count
            },
            'false_positive_estimates': dict(self.false_positive_estimates),
            'detection_thresholds': self.detection_thresholds.copy(),
            'method': 'robust_statistical'
        }
    
    def reset_neighbor_data(self, neighbor_id: str) -> None:
        """Reset all data for a specific neighbor."""
        self.neighbor_features.pop(neighbor_id, None)
        self.baseline_statistics.pop(neighbor_id, None)
        self.isolation_forests.pop(neighbor_id, None)
        self.update_count.pop(neighbor_id, None)
        self.detection_count.pop(neighbor_id, None)
        self.false_positive_estimates.pop(neighbor_id, None)
        
        self.logger.info(f"Reset statistical data for neighbor {neighbor_id}")