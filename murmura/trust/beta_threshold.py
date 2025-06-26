"""
Beta distribution-based adaptive threshold for trust monitoring.

This module implements a Bayesian approach to threshold adaptation using
Beta distributions, which naturally model trust scores in [0,1].
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import logging
from collections import deque


@dataclass
class BetaThresholdConfig:
    """Configuration for Beta distribution-based thresholding."""
    
    # Prior parameters (weak priors for uniform distribution)
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    
    # Percentile for threshold (e.g., 0.95 = 95th percentile)
    base_percentile: float = 0.95
    
    # Context-specific percentile adjustments
    early_rounds_adjustment: float = -0.05  # More permissive in early rounds
    late_rounds_adjustment: float = 0.02    # Slightly stricter in late rounds
    
    # Learning rate for parameter updates
    learning_rate: float = 1.0
    
    # Minimum observations before using Beta threshold
    min_observations: int = 10
    
    # Window size for recent observations
    observation_window: int = 100
    
    # Enable context-specific models
    use_contextual_models: bool = True


class BetaThreshold:
    """
    Beta distribution-based adaptive threshold for trust scores.
    
    Uses Bayesian updating to learn the distribution of trust scores
    and set thresholds based on percentiles of the learned distribution.
    """
    
    def __init__(self, config: Optional[BetaThresholdConfig] = None):
        self.config = config or BetaThresholdConfig()
        self.logger = logging.getLogger(f"{__name__}.BetaThreshold")
        
        # Initialize Beta parameters
        self.alpha = self.config.alpha_prior
        self.beta = self.config.beta_prior
        
        # Track observations
        self.observations = deque(maxlen=self.config.observation_window)
        self.total_observations = 0
        
        # Statistics
        self.trust_score_history = deque(maxlen=100)
        self.threshold_history = deque(maxlen=100)
        
        self.logger.info(
            f"Initialized Beta threshold with α={self.alpha:.2f}, β={self.beta:.2f}, "
            f"percentile={self.config.base_percentile:.2f}"
        )
    
    def update(self, trust_score: float, is_malicious: bool) -> None:
        """
        Update Beta distribution parameters based on observation.
        
        Args:
            trust_score: Observed trust score [0, 1]
            is_malicious: Whether the node was actually malicious
        """
        # Validate input
        trust_score = np.clip(trust_score, 0.0, 1.0)
        
        # Store observation
        self.observations.append({
            'trust_score': trust_score,
            'is_malicious': is_malicious,
            'timestamp': self.total_observations
        })
        self.total_observations += 1
        self.trust_score_history.append(trust_score)
        
        # Update Beta parameters
        lr = self.config.learning_rate
        
        if is_malicious:
            # Malicious node: update beta (failures)
            # Weight by how wrong we were (low trust score = we got it right)
            self.beta += lr * (1 - trust_score)
        else:
            # Benign node: update alpha (successes)
            # Weight by trust score (high trust = we got it right)
            self.alpha += lr * trust_score
        
        self.logger.debug(
            f"Updated Beta params: α={self.alpha:.3f}, β={self.beta:.3f} "
            f"(trust={trust_score:.3f}, malicious={is_malicious})"
        )
    
    def get_threshold(self, percentile: Optional[float] = None) -> float:
        """
        Get current threshold at specified percentile.
        
        Args:
            percentile: Percentile for threshold (default: base_percentile)
            
        Returns:
            Threshold value
        """
        if self.total_observations < self.config.min_observations:
            # Not enough data: use very conservative default for FL
            return 0.99  # Very high threshold to avoid false positives in early stages
        
        percentile = percentile or self.config.base_percentile
        
        # Create Beta distribution
        dist = stats.beta(self.alpha, self.beta)
        
        # Get percentile threshold
        threshold = dist.ppf(percentile)
        
        # Store in history
        self.threshold_history.append(threshold)
        
        return threshold
    
    def get_confidence(self) -> float:
        """
        Get confidence level based on number of observations.
        
        Returns:
            Confidence score [0, 1]
        """
        # More observations = higher confidence
        # Asymptotic to 1 with exponential approach
        total_params = self.alpha + self.beta
        confidence = 1 - np.exp(-total_params / 50)
        return confidence
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics of the Beta distribution."""
        dist = stats.beta(self.alpha, self.beta)
        
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'mean': dist.mean(),
            'variance': dist.var(),
            'std': dist.std(),
            'mode': (self.alpha - 1) / (self.alpha + self.beta - 2) if self.alpha > 1 and self.beta > 1 else np.nan,
            'confidence': self.get_confidence(),
            'total_observations': self.total_observations,
            'current_threshold': self.get_threshold(),
        }
    
    def reset(self) -> None:
        """Reset to prior distribution."""
        self.alpha = self.config.alpha_prior
        self.beta = self.config.beta_prior
        self.observations.clear()
        self.total_observations = 0
        self.trust_score_history.clear()
        self.threshold_history.clear()
        self.logger.info("Reset Beta threshold to priors")


class ContextualBetaThreshold:
    """
    Context-aware Beta threshold that maintains separate distributions
    for different federated learning contexts.
    """
    
    def __init__(self, config: Optional[BetaThresholdConfig] = None):
        self.config = config or BetaThresholdConfig()
        self.logger = logging.getLogger(f"{__name__}.ContextualBetaThreshold")
        
        # Separate Beta models for different contexts
        self.models = {
            'early_rounds': BetaThreshold(config),
            'mid_rounds': BetaThreshold(config),
            'late_rounds': BetaThreshold(config),
            'high_accuracy': BetaThreshold(config),
            'low_accuracy': BetaThreshold(config),
        }
        
        # Global model (uses all observations)
        self.global_model = BetaThreshold(config)
        
        self.logger.info("Initialized contextual Beta threshold system")
    
    def _get_context_key(self, fl_round: int, total_rounds: int, accuracy: float) -> List[str]:
        """Determine which context models apply."""
        contexts = []
        
        # Round-based context
        progress = fl_round / max(1, total_rounds)
        if progress < 0.3:
            contexts.append('early_rounds')
        elif progress > 0.7:
            contexts.append('late_rounds')
        else:
            contexts.append('mid_rounds')
        
        # Accuracy-based context
        if accuracy > 0.9:
            contexts.append('high_accuracy')
        elif accuracy < 0.6:
            contexts.append('low_accuracy')
        
        return contexts
    
    def update(self, trust_score: float, is_malicious: bool, 
               fl_round: int, total_rounds: int, accuracy: float) -> None:
        """Update relevant context models."""
        # Always update global model
        self.global_model.update(trust_score, is_malicious)
        
        # Update context-specific models
        contexts = self._get_context_key(fl_round, total_rounds, accuracy)
        for context in contexts:
            if context in self.models:
                self.models[context].update(trust_score, is_malicious)
    
    def get_threshold(self, fl_round: int, total_rounds: int, accuracy: float) -> float:
        """Get context-aware threshold."""
        contexts = self._get_context_key(fl_round, total_rounds, accuracy)
        
        # Get percentile adjustment based on context
        progress = fl_round / max(1, total_rounds)
        if progress < 0.3:
            percentile_adjustment = self.config.early_rounds_adjustment
        elif progress > 0.7:
            percentile_adjustment = self.config.late_rounds_adjustment
        else:
            percentile_adjustment = 0.0
        
        adjusted_percentile = self.config.base_percentile + percentile_adjustment
        
        # Use context-specific model if available and confident
        thresholds = []
        weights = []
        
        for context in contexts:
            model = self.models.get(context)
            if model and model.get_confidence() > 0.5:
                thresholds.append(model.get_threshold(adjusted_percentile))
                weights.append(model.get_confidence())
        
        # Always include global model
        thresholds.append(self.global_model.get_threshold(adjusted_percentile))
        weights.append(self.global_model.get_confidence() * 0.5)  # Lower weight
        
        # Weighted average of thresholds
        if thresholds:
            weights = np.array(weights)
            weights = weights / weights.sum()
            threshold = np.average(thresholds, weights=weights)
        else:
            threshold = 0.99  # Very conservative default for FL
        
        # Ensure threshold is appropriate for FL (high values are normal)
        threshold = max(threshold, 0.85)  # Never go below 0.85 in FL context
        
        return threshold
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics for all models."""
        stats = {
            'global': self.global_model.get_statistics(),
            'contexts': {}
        }
        
        for name, model in self.models.items():
            if model.total_observations > 0:
                stats['contexts'][name] = model.get_statistics()
        
        return stats