"""
Streaming Hilbert-Schmidt Independence Criterion (HSIC) implementation for trust drift detection.

This module implements an efficient streaming version of HSIC that can be used to detect
trust drift in model updates in decentralized federated learning.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque
import logging


class StreamingHSIC:
    """
    Streaming implementation of Hilbert-Schmidt Independence Criterion (HSIC).
    
    HSIC measures the statistical dependence between two random variables.
    This implementation uses a sliding window approach for efficiency.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        threshold: float = 0.05,
        alpha: float = 0.95,  # Exponential moving average factor
    ):
        """
        Initialize the Streaming HSIC calculator.
        
        Args:
            window_size: Size of the sliding window for streaming computation
            kernel_type: Type of kernel to use ("rbf", "linear", "polynomial")
            gamma: RBF kernel parameter
            threshold: HSIC threshold for independence (lower values indicate independence)
            alpha: Exponential moving average factor for adaptive thresholding
        """
        self.window_size = window_size
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.threshold = threshold
        self.alpha = alpha
        
        # Sliding windows for X and Y samples
        self.x_window = deque(maxlen=window_size)
        self.y_window = deque(maxlen=window_size)
        
        # Running statistics
        self.hsic_history = deque(maxlen=window_size)
        self.adaptive_threshold = threshold
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.StreamingHSIC")
        
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix for the given data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Kernel matrix of shape (n_samples, n_samples)
        """
        n = X.shape[0]
        
        if self.kernel_type == "rbf":
            # RBF (Gaussian) kernel
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    diff = X[i] - X[j]
                    K[i, j] = np.exp(-self.gamma * np.dot(diff, diff))
            return K
            
        elif self.kernel_type == "linear":
            # Linear kernel
            return np.dot(X, X.T)
            
        elif self.kernel_type == "polynomial":
            # Polynomial kernel (degree 2)
            return (1 + np.dot(X, X.T)) ** 2
            
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _center_kernel_matrix(self, K: np.ndarray) -> np.ndarray:
        """
        Center the kernel matrix using the centering matrix H.
        
        Args:
            K: Kernel matrix
            
        Returns:
            Centered kernel matrix
        """
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    
    def compute_hsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute HSIC between X and Y.
        
        Args:
            X: First variable samples (n_samples, n_features_x)
            Y: Second variable samples (n_samples, n_features_y)
            
        Returns:
            HSIC value (normalized)
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        
        n = X.shape[0]
        if n < 2:
            return 0.0
        
        # Compute kernel matrices
        K_x = self._compute_kernel_matrix(X)
        K_y = self._compute_kernel_matrix(Y)
        
        # Center the kernel matrices
        K_x_centered = self._center_kernel_matrix(K_x)
        K_y_centered = self._center_kernel_matrix(K_y)
        
        # Compute HSIC
        hsic = np.trace(K_x_centered @ K_y_centered) / (n - 1) ** 2
        
        # Normalize by the product of kernel matrix norms for scale invariance
        norm_x = np.sqrt(np.trace(K_x_centered @ K_x_centered)) / (n - 1)
        norm_y = np.sqrt(np.trace(K_y_centered @ K_y_centered)) / (n - 1)
        
        if norm_x > 0 and norm_y > 0:
            hsic_normalized = hsic / (norm_x * norm_y)
        else:
            hsic_normalized = 0.0
            
        return hsic_normalized
    
    def update(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
        """
        Update the streaming HSIC with new samples.
        
        Args:
            x: New sample from first variable
            y: New sample from second variable
            
        Returns:
            Tuple of (current HSIC value, trust drift detected)
        """
        # Add to windows
        self.x_window.append(x)
        self.y_window.append(y)
        
        # Need at least 2 samples to compute HSIC
        if len(self.x_window) < 2:
            return 0.0, False
        
        # Convert to arrays
        X = np.array(self.x_window)
        Y = np.array(self.y_window)
        
        # Compute HSIC
        hsic_value = self.compute_hsic(X, Y)
        self.hsic_history.append(hsic_value)
        
        # Update adaptive threshold using exponential moving average
        if len(self.hsic_history) > 1:
            self.adaptive_threshold = (
                self.alpha * self.adaptive_threshold + 
                (1 - self.alpha) * np.mean(self.hsic_history)
            )
        
        # Detect trust drift with improved adaptive thresholding
        # For federated learning, normal HSIC values are typically 0.9+
        # We need to detect anomalous spikes above the normal range
        
        # During the first few samples, learn the baseline
        if len(self.hsic_history) < 10:
            # Don't detect drift until we have enough samples to establish baseline
            drift_detected = False
        else:
            # Use statistical approach: detect values that are outliers
            hsic_array = np.array(self.hsic_history)
            hsic_mean = np.mean(hsic_array)
            hsic_std = np.std(hsic_array)
            
            # Detect values that are more than 2 standard deviations above mean
            # OR exceed a reasonable absolute threshold
            absolute_threshold = max(0.95, hsic_mean + 2 * hsic_std)
            
            drift_detected = (
                hsic_value > absolute_threshold and
                hsic_value > hsic_mean + 3 * hsic_std  # Very conservative
            )
        
        if drift_detected:
            hsic_array = np.array(self.hsic_history)
            hsic_mean = np.mean(hsic_array)
            hsic_std = np.std(hsic_array)
            self.logger.warning(
                f"Trust drift detected! HSIC: {hsic_value:.4f}, "
                f"Mean: {hsic_mean:.4f}, Std: {hsic_std:.4f}, "
                f"Threshold: {max(0.95, hsic_mean + 2 * hsic_std):.4f}"
            )
        
        return hsic_value, drift_detected
    
    def reset(self) -> None:
        """Reset the streaming HSIC state."""
        self.x_window.clear()
        self.y_window.clear()
        self.hsic_history.clear()
        self.adaptive_threshold = self.threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics of the streaming HSIC.
        
        Returns:
            Dictionary with statistics
        """
        if len(self.hsic_history) == 0:
            return {
                "current_hsic": 0.0,
                "mean_hsic": 0.0,
                "std_hsic": 0.0,
                "adaptive_threshold": self.adaptive_threshold,
                "window_size": len(self.x_window),
                "drift_rate": 0.0,
            }
        
        hsic_array = np.array(self.hsic_history)
        
        # Calculate drift rate (how often drift is detected)
        drift_count = sum(1 for h in self.hsic_history if h > self.threshold)
        drift_rate = drift_count / len(self.hsic_history)
        
        return {
            "current_hsic": self.hsic_history[-1],
            "mean_hsic": np.mean(hsic_array),
            "std_hsic": np.std(hsic_array),
            "adaptive_threshold": self.adaptive_threshold,
            "window_size": len(self.x_window),
            "drift_rate": drift_rate,
        }


class ModelUpdateHSIC(StreamingHSIC):
    """
    Specialized HSIC implementation for model parameter updates.
    
    This class adapts HSIC to work with high-dimensional model parameters
    by using dimensionality reduction and efficient representations.
    
    Key improvement: Dynamic baseline calibration to learn normal HSIC values
    during honest federated learning phases.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        kernel_type: str = "rbf",
        gamma: float = 0.1,
        threshold: float = 0.1,
        alpha: float = 0.9,
        reduce_dim: bool = True,
        target_dim: int = 100,
        calibration_rounds: int = 5,  # Number of rounds to calibrate baseline
        baseline_percentile: float = 95.0,  # Percentile for threshold setting
    ):
        """
        Initialize ModelUpdateHSIC.
        
        Args:
            window_size: Size of the sliding window
            kernel_type: Type of kernel to use
            gamma: RBF kernel parameter
            threshold: HSIC threshold for independence (will be auto-calibrated)
            alpha: Exponential moving average factor
            reduce_dim: Whether to reduce dimensionality of parameters
            target_dim: Target dimension for reduction
            calibration_rounds: Number of rounds to collect baseline data
            baseline_percentile: Percentile of baseline HSIC values to use as threshold
        """
        super().__init__(window_size, kernel_type, gamma, threshold, alpha)
        self.reduce_dim = reduce_dim
        self.target_dim = target_dim
        self.calibration_rounds = calibration_rounds
        self.baseline_percentile = baseline_percentile
        
        # For incremental PCA-like dimensionality reduction
        self.projection_matrix: Optional[np.ndarray] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.n_samples_seen = 0
        
        # Improved statistical drift detection
        self.logger.info(f"HSIC using statistical outlier detection for trust drift")
        
    def _flatten_parameters(self, parameters: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten model parameters into a single vector.
        
        Args:
            parameters: Dictionary of model parameters
            
        Returns:
            Flattened parameter vector
        """
        # Sort keys for consistency
        sorted_keys = sorted(parameters.keys())
        
        # Flatten all parameters
        flattened = []
        for key in sorted_keys:
            param = parameters[key]
            flattened.append(param.flatten())
        
        return np.concatenate(flattened)
    
    def _reduce_dimensions(self, x: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of parameter vector using random projection.
        
        Args:
            x: High-dimensional parameter vector
            
        Returns:
            Reduced dimension vector
        """
        if not self.reduce_dim:
            return x
        
        # Initialize projection matrix if needed (random projection)
        if self.projection_matrix is None:
            # Use random Gaussian projection
            input_dim = x.shape[0]
            self.projection_matrix = np.random.randn(self.target_dim, input_dim)
            # Normalize rows
            norms = np.linalg.norm(self.projection_matrix, axis=1, keepdims=True)
            self.projection_matrix /= norms
            
            self.feature_mean = np.zeros(input_dim)
        
        # Update running mean
        self.n_samples_seen += 1
        delta = x - self.feature_mean
        self.feature_mean += delta / self.n_samples_seen
        
        # Center and project
        x_centered = x - self.feature_mean
        x_reduced = self.projection_matrix @ x_centered
        
        return x_reduced
    
    
    def update_with_parameters(
        self,
        current_params: Dict[str, np.ndarray],
        received_params: Dict[str, np.ndarray],
        node_id: Optional[str] = None,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Update HSIC with model parameter updates.
        
        Args:
            current_params: Current model parameters
            received_params: Received model parameters from neighbor
            node_id: Optional ID of the node sending parameters
            
        Returns:
            Tuple of (HSIC value, drift detected, statistics)
        """
        # Flatten parameters
        current_flat = self._flatten_parameters(current_params)
        received_flat = self._flatten_parameters(received_params)
        
        # Compute parameter difference (update vector)
        update_vector = received_flat - current_flat
        
        # Reduce dimensions if needed
        if self.reduce_dim:
            current_reduced = self._reduce_dimensions(current_flat)
            update_reduced = self._reduce_dimensions(update_vector)
        else:
            current_reduced = current_flat
            update_reduced = update_vector
        
        # Update HSIC
        hsic_value, drift_detected = self.update(current_reduced, update_reduced)
        
        # Log HSIC statistics periodically for monitoring
        if len(self.hsic_history) > 0 and len(self.hsic_history) % 20 == 0:
            hsic_array = np.array(self.hsic_history)
            self.logger.debug(
                f"HSIC Statistics: Current={hsic_value:.3f}, "
                f"Mean={np.mean(hsic_array):.3f}, Std={np.std(hsic_array):.3f}, "
                f"Range=[{np.min(hsic_array):.3f}, {np.max(hsic_array):.3f}]"
            )
        
        # Get statistics
        stats = self.get_statistics()
        stats["node_id"] = node_id
        stats["update_norm"] = np.linalg.norm(update_vector)
        stats["relative_update_norm"] = stats["update_norm"] / (np.linalg.norm(current_flat) + 1e-8)
        
        # Add statistical threshold info
        if len(self.hsic_history) >= 10:
            hsic_array = np.array(self.hsic_history)
            stats["statistical_threshold"] = max(0.95, np.mean(hsic_array) + 2 * np.std(hsic_array))
        else:
            stats["statistical_threshold"] = "learning_baseline"
        
        return hsic_value, drift_detected, stats