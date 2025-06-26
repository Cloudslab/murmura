#!/usr/bin/env python3
"""
Test suite for Beta distribution-based adaptive thresholding.
"""

import numpy as np
import pytest
from scipy import stats

from murmura.trust.beta_threshold import (
    BetaThreshold,
    ContextualBetaThreshold,
    BetaThresholdConfig,
)


class TestBetaThreshold:
    """Test basic Beta threshold functionality."""
    
    def test_initialization(self):
        """Test Beta threshold initialization."""
        config = BetaThresholdConfig(alpha_prior=1.0, beta_prior=1.0)
        threshold = BetaThreshold(config)
        
        # Check initial parameters
        assert threshold.alpha == 1.0
        assert threshold.beta == 1.0
        assert threshold.total_observations == 0
        
        # Initial threshold should be conservative
        initial_threshold = threshold.get_threshold()
        assert 0.7 <= initial_threshold <= 0.9
    
    def test_update_honest_nodes(self):
        """Test updates with honest nodes."""
        threshold = BetaThreshold()
        
        # Simulate honest nodes with high trust scores
        for _ in range(10):
            trust_score = np.random.uniform(0.8, 0.95)
            threshold.update(trust_score, is_malicious=False)
        
        # Alpha should increase more than beta
        assert threshold.alpha > threshold.beta
        assert threshold.alpha > 1.0  # Should have increased from prior
        
        # Threshold should remain high (most nodes are trustworthy)
        current_threshold = threshold.get_threshold()
        assert current_threshold > 0.7
    
    def test_update_malicious_nodes(self):
        """Test updates with malicious nodes."""
        threshold = BetaThreshold()
        
        # Simulate malicious nodes with low trust scores
        for _ in range(5):
            trust_score = np.random.uniform(0.1, 0.3)
            threshold.update(trust_score, is_malicious=True)
        
        # Beta should increase
        assert threshold.beta > 1.0
        
        # Statistics should reflect the distribution
        stats = threshold.get_statistics()
        assert 'mean' in stats
        assert 'variance' in stats
        assert stats['total_observations'] == 5
    
    def test_confidence_calculation(self):
        """Test confidence increases with observations."""
        threshold = BetaThreshold()
        
        initial_confidence = threshold.get_confidence()
        assert initial_confidence < 0.2  # Low confidence initially
        
        # Add many observations
        for i in range(50):
            trust_score = np.random.uniform(0.7, 0.95)
            threshold.update(trust_score, is_malicious=False)
        
        final_confidence = threshold.get_confidence()
        assert final_confidence > 0.5  # Higher confidence after observations
        assert final_confidence > initial_confidence
    
    def test_threshold_convergence(self):
        """Test that threshold converges with consistent observations."""
        threshold = BetaThreshold()
        
        # Simulate consistent honest behavior (normal FL)
        thresholds = []
        for i in range(100):
            # Normal FL has high HSIC/trust scores
            trust_score = np.random.normal(0.92, 0.03)
            trust_score = np.clip(trust_score, 0, 1)
            threshold.update(trust_score, is_malicious=False)
            
            if i >= 10:  # After minimum observations
                thresholds.append(threshold.get_threshold())
        
        # Check convergence (variance should decrease)
        early_variance = np.var(thresholds[:20])
        late_variance = np.var(thresholds[-20:])
        assert late_variance < early_variance


class TestContextualBetaThreshold:
    """Test contextual Beta threshold functionality."""
    
    def test_context_separation(self):
        """Test that different contexts maintain separate models."""
        config = BetaThresholdConfig(use_contextual_models=True)
        contextual = ContextualBetaThreshold(config)
        
        # Update early rounds context
        for _ in range(10):
            contextual.update(
                trust_score=0.9,
                is_malicious=False,
                fl_round=2,
                total_rounds=20,
                accuracy=0.5
            )
        
        # Update late rounds context
        for _ in range(10):
            contextual.update(
                trust_score=0.95,
                is_malicious=False,
                fl_round=18,
                total_rounds=20,
                accuracy=0.9
            )
        
        # Get statistics
        stats = contextual.get_statistics()
        
        # Should have separate statistics for different contexts
        assert 'contexts' in stats
        assert 'early_rounds' in stats['contexts']
        assert 'late_rounds' in stats['contexts']
        
        # Different contexts should have different parameters
        early_stats = stats['contexts']['early_rounds']
        late_stats = stats['contexts']['late_rounds']
        assert early_stats['alpha'] != late_stats['alpha']
    
    def test_adaptive_percentile(self):
        """Test that percentile adjusts based on FL phase."""
        contextual = ContextualBetaThreshold()
        
        # Add some baseline observations
        for _ in range(20):
            contextual.update(0.9, False, 5, 20, 0.7)
        
        # Early rounds should have lower threshold (more permissive)
        early_threshold = contextual.get_threshold(
            fl_round=2, total_rounds=20, accuracy=0.5
        )
        
        # Late rounds should have higher threshold (slightly stricter)
        late_threshold = contextual.get_threshold(
            fl_round=18, total_rounds=20, accuracy=0.9
        )
        
        # The difference should be present but not extreme
        assert early_threshold < late_threshold
        assert abs(early_threshold - late_threshold) < 0.15


def test_integration_with_normal_fl():
    """Test Beta threshold with normal FL trust patterns."""
    threshold = BetaThreshold()
    
    # Simulate 50 rounds of normal FL
    false_positives = 0
    
    for round_num in range(50):
        # Normal FL has trust scores around 0.85-0.98
        trust_score = np.random.uniform(0.85, 0.98)
        
        # Get current threshold
        if round_num >= 5:  # After warm-up
            current_threshold = threshold.get_threshold()
            
            # Check if this would be a false positive
            if trust_score < current_threshold:
                false_positives += 1
        
        # Update with honest node
        threshold.update(trust_score, is_malicious=False)
    
    # Should have very few false positives
    false_positive_rate = false_positives / 45  # 45 rounds after warm-up
    assert false_positive_rate < 0.05  # Less than 5% false positives
    
    # Final threshold should be high (reflecting normal FL behavior)
    final_threshold = threshold.get_threshold()
    assert final_threshold > 0.85


if __name__ == "__main__":
    # Run basic tests
    print("Testing Beta threshold implementation...")
    
    # Test basic functionality
    test_beta = TestBetaThreshold()
    test_beta.test_initialization()
    test_beta.test_update_honest_nodes()
    test_beta.test_update_malicious_nodes()
    test_beta.test_confidence_calculation()
    test_beta.test_threshold_convergence()
    
    # Test contextual functionality
    test_contextual = TestContextualBetaThreshold()
    test_contextual.test_context_separation()
    test_contextual.test_adaptive_percentile()
    
    # Test integration
    test_integration_with_normal_fl()
    
    print("✅ All Beta threshold tests passed!")