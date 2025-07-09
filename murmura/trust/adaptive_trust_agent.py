"""
Adaptive Trust Agent using Meta-Learning/RL for dynamic threshold adjustment.

This module implements a context-aware trust detection system that learns
optimal trust decisions based on federated learning context rather than
just raw HSIC values.

Key Features:
- Context-aware decisions based on FL round, network state, etc.
- Dataset-independent learning from behavioral patterns
- FL-aware HSIC interpretation (recognizes 0.9+ as normal)
- Adaptive thresholding based on FL phase and risk tolerance
- Online learning with policy updates from feedback
- Explainable decisions with human-readable reasoning

The system addresses the core issue where traditional HSIC-based trust
systems produce false positives because they treat normal FL correlation
(HSIC 0.9+) as suspicious behavior.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time

from murmura.trust.beta_threshold import (
    BetaThreshold,
    ContextualBetaThreshold,
    BetaThresholdConfig,
)


@dataclass
class TrustContext:
    """Context information for adaptive threshold decisions - DECENTRALIZED FL ONLY."""
    
    # Real FL Context (measurable in decentralized setting)
    current_round: int
    total_rounds: int
    topology_type: str  # 'ring', 'complete', 'line'
    
    # Real Parameter Signals (directly measured)
    hsic_value: float
    update_magnitude: float
    update_direction_consistency: float
    neighbor_trust_scores: List[float]


class AdaptiveThresholdAgent:
    """
    Meta-learning agent for adaptive trust threshold adjustment.
    
    Key idea: Learn optimal trust decisions based on context,
    not just raw HSIC values. This addresses the core issue where
    normal FL has HSIC values of 0.9+ which are not indicative of attacks.
    """
    
    def __init__(self, use_beta_threshold: bool = True):
        self.logger = logging.getLogger(f"{__name__}.AdaptiveThresholdAgent")
        
        # Experience buffer for online learning
        self.experience_buffer = deque(maxlen=1000)
        self.reward_history = deque(maxlen=100)
        
        # Feature normalization stats
        self.feature_mean = None
        self.feature_std = None
        self.n_samples_seen = 0
        
        # Simple policy parameters (can be upgraded to neural network)
        self.policy_weights = self._initialize_policy()
        self.learning_rate = 0.01
        
        # Context tracking
        self.context_history = deque(maxlen=50)
        
        # BASELINE LEARNING: Learn normal FL behavior patterns
        self.baseline_learning_rounds = 3  # Learn baseline in first 3 rounds
        self.baseline_hsic_values = deque(maxlen=100)
        self.baseline_update_norms = deque(maxlen=100)
        self.baseline_learned = False
        self.hsic_baseline_mean = 0.9  # Default assumption
        self.hsic_baseline_std = 0.05
        self.update_norm_baseline_mean = 0.01
        self.update_norm_baseline_std = 0.005
        
        # Beta distribution-based thresholding (now secondary to baseline learning)
        self.use_beta_threshold = use_beta_threshold
        if use_beta_threshold:
            beta_config = BetaThresholdConfig(
                base_percentile=0.5,            # Moderate percentile since baseline learning is primary
                early_rounds_adjustment=-0.1,   # More permissive early rounds
                late_rounds_adjustment=0.05,    # Slightly stricter in late rounds
                min_observations=5,             # Wait for baseline learning to complete
                learning_rate=0.4,              # Moderate learning rate
            )
            self.beta_threshold = ContextualBetaThreshold(beta_config)
            self.logger.info("Using Beta distribution-based adaptive thresholding (secondary to baseline learning)")
        else:
            self.beta_threshold = None
        
        self.logger.info("Initialized Adaptive Trust Agent with meta-learning capabilities")
        
    def _initialize_policy(self) -> np.ndarray:
        """Initialize simple linear policy weights (used only during baseline learning)."""
        # 8 real features only (no fake/hardcoded features)  
        # Initialize with moderate weights since baseline learning is primary detection method
        weights = np.random.normal(0.0, 0.1, 8)  # Neutral initialization
        
        # Set specific weights for important features (moderate since baseline learning is primary)
        # Feature 1 is HSIC value - moderate negative weight
        weights[1] = -0.3  # Moderate negative weight for HSIC
        
        # Feature 2 is update norm - moderate positive weight
        weights[2] = 0.2   # Moderate positive weight for update norm
        
        # Feature 3 is consistency - moderate negative weight
        weights[3] = -0.2  # Moderate negative weight for inconsistency
        
        return weights
    
    def get_trust_decision(self, context: TrustContext) -> Tuple[bool, float, str]:
        """
        Make trust decision based on full context using baseline learning.
        
        Args:
            context: Full FL and network context
            
        Returns:
            (is_malicious, confidence, reasoning)
        """
        # Update baseline learning if still in learning phase
        self._update_baseline_learning(context)
        
        # Extract normalized features
        features = self._extract_features(context)
        
        # Use outlier detection if baseline is learned, otherwise use policy
        if self.baseline_learned:
            is_malicious, confidence, reasoning = self._detect_outlier_behavior(context)
        else:
            # During baseline learning, be very conservative
            malicious_prob = self._evaluate_policy(features) * 0.1  # Very conservative
            threshold = 0.9  # Very high threshold during learning
            is_malicious = malicious_prob > threshold
            confidence = abs(malicious_prob - threshold)
            reasoning = f"Baseline learning phase ({context.current_round}/{self.baseline_learning_rounds}): Conservative detection"
        
        # Store context for learning
        self.context_history.append({
            'context': context,
            'features': features,
            'decision': is_malicious,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        return is_malicious, confidence, reasoning
    
    def _extract_features(self, context: TrustContext) -> np.ndarray:
        """
        Extract normalized feature vector from context - ONLY REAL FEATURES FOR DECENTRALIZED FL.
        
        RESEARCH NOTE: This uses only 8 real, measurable features suitable for publication:
        1. FL Progress (0-1): current_round / total_rounds
        2. HSIC Value (0-1): Real parameter correlation measurement  
        3. Update Magnitude (>0): L2 norm of parameter update
        4. Update Direction Consistency (0-1): Temporal consistency of updates
        5. Neighbor Trust Mean (0-1): Average trust of neighboring nodes
        6. Neighbor Trust Std (≥0): Variance in neighbor trust scores  
        7. Neighbor Count (normalized): Number of neighbors / 10
        8. FL Phase (0,0.5,1): Early/Mid/Late phase indicator
        
        Removed fake/hardcoded features: global_accuracy, convergence_rate, 
        network_stability, communication_latency, attack_rate, false_positive_rate
        """
        
        # Only use features that are meaningful and measurable in decentralized FL
        features = np.array([
            # FL Progress - Real feature from actual round counting
            context.current_round / max(1, context.total_rounds),
            
            # HSIC Signal - Real correlation measurement between parameters
            context.hsic_value,
            
            # Parameter Update Properties - Real measurements
            context.update_magnitude,
            context.update_direction_consistency,
            
            # Neighbor Trust Statistics - Real trust scores from actual neighbors
            np.mean(context.neighbor_trust_scores) if context.neighbor_trust_scores else 0.5,
            np.std(context.neighbor_trust_scores) if len(context.neighbor_trust_scores) > 1 else 0.0,
            len(context.neighbor_trust_scores) / 10.0,  # Number of neighbors (normalized)
            
            # Phase indicator - Real FL phase based on actual rounds
            self._compute_phase_indicator(context),
        ])
        
        # Online feature normalization
        self._update_feature_stats(features)
        normalized_features = self._normalize_features(features)
        
        return normalized_features
    
    def _update_feature_stats(self, features: np.ndarray):
        """Update running mean and std for feature normalization."""
        self.n_samples_seen += 1
        
        if self.feature_mean is None:
            self.feature_mean = features.copy()
            self.feature_std = np.ones_like(features)
        else:
            # Online update of mean and variance
            delta = features - self.feature_mean
            self.feature_mean += delta / self.n_samples_seen
            
            # Simple running std approximation
            if self.n_samples_seen > 10:
                self.feature_std = 0.9 * self.feature_std + 0.1 * np.abs(delta)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics."""
        if self.feature_mean is None:
            return features
        
        normalized = (features - self.feature_mean) / (self.feature_std + 1e-8)
        return np.clip(normalized, -3, 3)  # Clip to prevent extreme values
    
    def _evaluate_policy(self, features: np.ndarray) -> float:
        """Evaluate policy to get malicious probability."""
        # Simple linear policy (can be upgraded to neural network)
        logit = np.dot(self.policy_weights, features)
        prob = 1.0 / (1.0 + np.exp(-logit))  # Sigmoid
        return prob
    
    def _get_adaptive_threshold(self, context: TrustContext) -> float:
        """
        Compute adaptive threshold based on FL phase and risk tolerance.
        
        Uses Beta distribution-based thresholding when available.
        """
        
        # Use Beta distribution threshold if enabled
        if self.use_beta_threshold and self.beta_threshold:
            # Get context-aware Beta threshold (round-based only)
            try:
                # Try contextual beta threshold first
                beta_threshold = self.beta_threshold.get_threshold(
                    fl_round=context.current_round,
                    total_rounds=context.total_rounds
                )
            except TypeError:
                # Fallback to simple beta threshold
                beta_threshold = self.beta_threshold.get_threshold()
            
            # HSIC-based adjustment (high HSIC is NORMAL in FL) - ONLY REAL FEATURE
            hsic_adjustment = 0.0
            if context.hsic_value > 0.995:
                hsic_adjustment = -0.02
            elif context.hsic_value < 0.8:
                hsic_adjustment = -0.05
            elif 0.9 <= context.hsic_value <= 0.98:
                hsic_adjustment = 0.05  # Reward normal behavior
            
            adaptive_threshold = beta_threshold + hsic_adjustment
            
            self.logger.debug(
                f"Beta threshold: {beta_threshold:.3f}, adjusted: {adaptive_threshold:.3f} "
                f"(hsic={hsic_adjustment:.3f})"
            )
            
            return np.clip(adaptive_threshold, 0.1, 0.95)
        
        # Fallback to manual threshold computation using ONLY REAL FEATURES
        base_threshold = 0.5  # Moderate threshold since baseline learning is primary
        
        # FL Phase adjustment - Real feature
        fl_progress = context.current_round / max(1, context.total_rounds)
        if fl_progress < 0.3:
            phase_adjustment = 0.2
        elif fl_progress > 0.8:
            phase_adjustment = -0.05
        else:
            phase_adjustment = 0.1
        
        # HSIC-based adjustment - Real feature
        hsic_adjustment = 0.0
        if context.hsic_value > 0.995:
            hsic_adjustment = -0.05
        elif context.hsic_value < 0.8:
            hsic_adjustment = -0.1
        elif 0.9 <= context.hsic_value <= 0.98:
            hsic_adjustment = 0.1
        
        adaptive_threshold = base_threshold + phase_adjustment + hsic_adjustment
        
        return np.clip(adaptive_threshold, 0.1, 0.9)
    
    def _compute_hsic_anomaly_score(self, context: TrustContext) -> float:
        """Compute HSIC-based anomaly score (real measurement)."""
        
        # In normal FL, HSIC typically ranges 0.90-0.99, with 0.94+ being very common
        # Only flag extreme deviations
        if context.hsic_value > 0.995:  # Extremely high (suspicious)
            return (context.hsic_value - 0.995) * 20  # Strong penalty for extreme values
        elif context.hsic_value < 0.85:  # Extremely low (suspicious)
            return (0.85 - context.hsic_value) * 10
        else:
            return 0.0  # Normal range - no anomaly
    
    def _compute_consistency_score(self, context: TrustContext) -> float:
        """Compute how consistent this node is with its neighbors."""
        if not context.neighbor_trust_scores:
            return 0.5
        
        neighbor_mean = np.mean(context.neighbor_trust_scores)
        neighbor_std = np.std(context.neighbor_trust_scores)
        
        # If neighbors have consistent trust, this node should too
        consistency = 1.0 - min(neighbor_std * 2, 1.0)
        
        return consistency
    
    def _compute_phase_indicator(self, context: TrustContext) -> float:
        """Compute FL phase indicator (early/mid/late)."""
        progress = context.current_round / max(1, context.total_rounds)
        
        if progress < 0.3:
            return 0.0  # Early phase
        elif progress < 0.7:
            return 0.5  # Mid phase  
        else:
            return 1.0  # Late phase
    
    def _generate_reasoning(self, context: TrustContext, prob: float, threshold: float) -> str:
        """Generate human-readable reasoning for the decision - ONLY REAL FEATURES."""
        
        factors = []
        
        # HSIC analysis with proper FL context
        if context.hsic_value > 0.995:
            factors.append(f"Extremely high HSIC ({context.hsic_value:.3f}) - suspicious")
        elif context.hsic_value >= 0.9:
            factors.append(f"Normal FL HSIC ({context.hsic_value:.3f}) - expected")
        elif context.hsic_value < 0.85:
            factors.append(f"Unusually low HSIC ({context.hsic_value:.3f}) - suspicious")
        
        if context.update_magnitude > 0.1:
            factors.append(f"Large update magnitude ({context.update_magnitude:.3f})")
        elif context.update_magnitude < 0.001:
            factors.append(f"Very small update magnitude ({context.update_magnitude:.3f})")
        
        if context.update_direction_consistency < 0.5:
            factors.append(f"Inconsistent update direction ({context.update_direction_consistency:.3f})")
        
        fl_progress = context.current_round / max(1, context.total_rounds)
        if fl_progress < 0.2:
            factors.append("Early FL phase (more permissive)")
        elif fl_progress > 0.8:
            factors.append("Late FL phase (more strict)")
        
        neighbor_count = len(context.neighbor_trust_scores)
        if neighbor_count > 0:
            neighbor_trust_mean = np.mean(context.neighbor_trust_scores)
            if neighbor_trust_mean < 0.5:
                factors.append(f"Low neighbor trust ({neighbor_trust_mean:.3f})")
            elif neighbor_trust_mean > 0.9:
                factors.append(f"High neighbor trust ({neighbor_trust_mean:.3f})")
        
        decision_str = 'MALICIOUS' if prob > threshold else 'HONEST'
        confidence = abs(prob - threshold)
        
        reasoning = f"Decision: {decision_str} (confidence: {confidence:.3f}). "
        reasoning += f"Factors: {', '.join(factors) if factors else 'Normal behavior pattern'}"
        
        return reasoning
    
    def _update_baseline_learning(self, context: TrustContext) -> None:
        """
        Update baseline learning with normal FL behavior patterns.
        
        Args:
            context: Current FL context
        """
        # Only learn baseline in early rounds when attacks are less likely
        if context.current_round <= self.baseline_learning_rounds:
            self.baseline_hsic_values.append(context.hsic_value)
            self.baseline_update_norms.append(context.update_magnitude)
            
            # Update running statistics
            if len(self.baseline_hsic_values) >= 5:  # Need some samples
                self.hsic_baseline_mean = np.mean(list(self.baseline_hsic_values))
                self.hsic_baseline_std = max(0.01, np.std(list(self.baseline_hsic_values)))
                
                self.update_norm_baseline_mean = np.mean(list(self.baseline_update_norms))
                self.update_norm_baseline_std = max(0.001, np.std(list(self.baseline_update_norms)))
                
                self.logger.debug(
                    f"Baseline update: HSIC={self.hsic_baseline_mean:.3f}±{self.hsic_baseline_std:.3f}, "
                    f"Update norm={self.update_norm_baseline_mean:.3f}±{self.update_norm_baseline_std:.3f}"
                )
        
        # Mark baseline as learned after collecting enough data
        if context.current_round > self.baseline_learning_rounds and len(self.baseline_hsic_values) >= 5:
            if not self.baseline_learned:
                self.baseline_learned = True
                self.logger.info(
                    f"Baseline learning complete: HSIC baseline={self.hsic_baseline_mean:.3f}±{self.hsic_baseline_std:.3f}, "
                    f"Update norm baseline={self.update_norm_baseline_mean:.3f}±{self.update_norm_baseline_std:.3f}"
                )
    
    def _detect_outlier_behavior(self, context: TrustContext) -> Tuple[bool, float, str]:
        """
        Detect malicious behavior using statistical outlier detection from learned baseline.
        
        Args:
            context: Current FL context
            
        Returns:
            (is_malicious, confidence, reasoning)
        """
        anomaly_scores = []
        reasoning_parts = []
        
        # 1. HSIC anomaly detection
        hsic_z_score = abs(context.hsic_value - self.hsic_baseline_mean) / self.hsic_baseline_std
        hsic_anomaly = hsic_z_score > 2.0  # 2-sigma rule
        anomaly_scores.append(hsic_z_score / 3.0)  # Normalize to [0,1] range
        
        if hsic_anomaly:
            reasoning_parts.append(f"HSIC outlier (z={hsic_z_score:.2f}, value={context.hsic_value:.3f} vs baseline={self.hsic_baseline_mean:.3f}±{self.hsic_baseline_std:.3f})")
        
        # 2. Update magnitude anomaly detection
        update_z_score = abs(context.update_magnitude - self.update_norm_baseline_mean) / self.update_norm_baseline_std
        update_anomaly = update_z_score > 2.5  # Slightly more tolerant for update norms
        anomaly_scores.append(update_z_score / 4.0)  # Normalize to [0,1] range
        
        if update_anomaly:
            reasoning_parts.append(f"Update magnitude outlier (z={update_z_score:.2f}, value={context.update_magnitude:.4f} vs baseline={self.update_norm_baseline_mean:.4f}±{self.update_norm_baseline_std:.4f})")
        
        # 3. Consistency with neighbors
        if context.neighbor_trust_scores:
            neighbor_mean = np.mean(context.neighbor_trust_scores)
            if neighbor_mean < 0.3:  # Very low neighbor trust
                anomaly_scores.append(0.8)
                reasoning_parts.append(f"Very low neighbor trust (mean={neighbor_mean:.3f})")
            elif neighbor_mean < 0.5:
                anomaly_scores.append(0.4)
                reasoning_parts.append(f"Low neighbor trust (mean={neighbor_mean:.3f})")
            else:
                anomaly_scores.append(0.0)
        else:
            anomaly_scores.append(0.0)  # No neighbors, can't judge
        
        # 4. Temporal consistency (if we have history)
        if len(self.context_history) > 2:
            recent_hsic = [ctx['context'].hsic_value for ctx in list(self.context_history)[-3:]]
            hsic_variance = np.var(recent_hsic)
            if hsic_variance > self.hsic_baseline_std ** 2 * 4:  # High variance
                anomaly_scores.append(0.6)
                reasoning_parts.append(f"High HSIC variance ({hsic_variance:.4f})")
            else:
                anomaly_scores.append(0.0)
        else:
            anomaly_scores.append(0.0)
        
        # Combine anomaly scores
        combined_anomaly_score = np.mean(anomaly_scores)
        
        # Dynamic threshold based on FL phase
        fl_progress = context.current_round / max(1, context.total_rounds)
        if fl_progress < 0.3:
            threshold = 0.7  # More permissive early
        elif fl_progress > 0.8:
            threshold = 0.4  # More strict late
        else:
            threshold = 0.5  # Moderate middle
        
        is_malicious = combined_anomaly_score > threshold
        confidence = combined_anomaly_score
        
        reasoning = f"Outlier detection: anomaly_score={combined_anomaly_score:.3f} vs threshold={threshold:.3f}. "
        reasoning += f"Factors: {'; '.join(reasoning_parts) if reasoning_parts else 'Normal behavior'}"
        
        return is_malicious, confidence, reasoning
    
    def update_from_feedback(self, 
                           context: TrustContext, 
                           decision: bool, 
                           actual_outcome: Optional[bool], 
                           reward: float):
        """Update the agent based on feedback (for online learning)."""
        
        experience = {
            'context': context,
            'decision': decision, 
            'actual': actual_outcome,
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.experience_buffer.append(experience)
        self.reward_history.append(reward)
        
        # Update Beta distribution if enabled
        if self.use_beta_threshold and self.beta_threshold and actual_outcome is not None:
            # Calculate trust score from the context
            # Higher trust score means less likely to be malicious
            trust_score = 1.0 - self._evaluate_policy(self._extract_features(context))
            
            # Update Beta distribution (round-based context only)
            self.beta_threshold.update(
                trust_score=trust_score,
                is_malicious=actual_outcome,
                fl_round=context.current_round,
                total_rounds=context.total_rounds
            )
            
            self.logger.debug(
                f"Updated Beta distribution: trust={trust_score:.3f}, "
                f"malicious={actual_outcome}, reward={reward:.3f}"
            )
        
        # Simple online policy update (gradient ascent)
        if len(self.experience_buffer) >= 10:
            self._update_policy()
        
        self.logger.debug(f"Updated policy with reward {reward:.3f}, buffer size: {len(self.experience_buffer)}")
    
    def _update_policy(self):
        """Update policy weights based on recent experience."""
        if len(self.experience_buffer) < 10:
            return
        
        # Simple gradient-based update using recent experiences
        recent_experiences = list(self.experience_buffer)[-10:]
        
        total_gradient = np.zeros_like(self.policy_weights)
        
        for exp in recent_experiences:
            context = exp['context']
            reward = exp['reward']
            decision = exp['decision']
            
            # Extract features
            features = self._extract_features(context)
            
            # Compute gradient (simplified policy gradient)
            prob = self._evaluate_policy(features)
            
            if decision:  # If we predicted malicious
                gradient = features * (1 - prob) * reward
            else:  # If we predicted honest
                gradient = -features * prob * reward
            
            total_gradient += gradient
        
        # Update weights
        self.policy_weights += self.learning_rate * total_gradient / len(recent_experiences)
        
        # Log update
        avg_reward = np.mean([exp['reward'] for exp in recent_experiences])
        self.logger.debug(f"Updated policy weights, avg reward: {avg_reward:.3f}")
    
    def get_beta_statistics(self) -> Optional[Dict[str, Any]]:
        """Get statistics from Beta distribution models."""
        if self.use_beta_threshold and self.beta_threshold:
            return self.beta_threshold.get_statistics()
        return None


class ContextTracker:
    """Tracks FL context for adaptive decision making - REAL FEATURES ONLY."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ContextTracker")
        self.fl_history = deque(maxlen=100)  # Only stores real measurements
        
    def build_context(self, node_id: str, update_data: Dict[str, Any]) -> TrustContext:
        """Build context for trust decision using ONLY REAL, MEASURABLE FEATURES."""
        
        # Extract only real, measurable context components for decentralized FL
        context = TrustContext(
            current_round=update_data.get('round', 0),
            total_rounds=update_data.get('total_rounds', 10),
            topology_type=update_data.get('topology', 'ring'),
            hsic_value=update_data.get('hsic', 0.9),
            update_magnitude=update_data.get('update_norm', 0.01),
            update_direction_consistency=update_data.get('consistency', 0.8),
            neighbor_trust_scores=update_data.get('neighbor_trusts', [])
        )
        
        # Store for history tracking (only real features)
        self.fl_history.append({
            'round': context.current_round,
            'hsic_value': context.hsic_value,
            'update_magnitude': context.update_magnitude,
            'timestamp': time.time()
        })
        
        return context
    
    # Removed record_decision_outcome method (relied on unknown ground truth)
    
    # Removed fake feature computation methods:
    # - _compute_convergence_rate (based on non-existent global accuracy)
    # - _compute_network_stability (hardcoded to 0.95)
    # - _compute_avg_latency (hardcoded to 0.1)
    # - _compute_recent_attack_rate (based on unreliable ground truth)
    # - _compute_false_positive_rate (based on unknown ground truth)


class DatasetIndependentTrustSystem:
    """
    Main trust system using adaptive thresholding.
    
    Key advantages:
    1. Dataset independent - learns from context, not absolute values
    2. Phase-aware - different behavior in early vs late FL rounds  
    3. Self-correcting - learns from false positives/negatives
    4. Explainable - provides reasoning for decisions
    """
    
    def __init__(self, use_beta_threshold: bool = True):
        self.logger = logging.getLogger(f"{__name__}.DatasetIndependentTrustSystem")
        self.agent = AdaptiveThresholdAgent(use_beta_threshold=use_beta_threshold)
        self.context_tracker = ContextTracker()
        
        beta_msg = "with Beta distribution thresholding" if use_beta_threshold else "with manual thresholding"
        self.logger.info(f"Initialized Dataset-Independent Trust System {beta_msg}")
        
    def assess_trust(self, node_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main trust assessment function."""
        
        # Build context from current state
        context = self.context_tracker.build_context(node_id, update_data)
        
        # Get adaptive decision
        is_malicious, confidence, reasoning = self.agent.get_trust_decision(context)
        
        result = {
            'malicious': is_malicious,
            'confidence': confidence,
            'reasoning': reasoning,
            'context': context,
            'adaptive_threshold': self.agent._get_adaptive_threshold(context),
            'hsic_value': context.hsic_value,
            'trust_score': 1.0 - confidence if not is_malicious else confidence
        }
        
        # Log decision for debugging
        self.logger.debug(
            f"Node {node_id}: {'MALICIOUS' if is_malicious else 'HONEST'} "
            f"(conf: {confidence:.3f}, HSIC: {context.hsic_value:.3f})"
        )
        
        return result
    
    def provide_feedback(self, node_id: str, context: TrustContext, 
                        decision: bool, actual_outcome: Optional[bool], reward: float):
        """Provide feedback for online learning."""
        self.agent.update_from_feedback(context, decision, actual_outcome, reward)
        self.context_tracker.record_decision_outcome(decision, actual_outcome)
    
    def set_beta_threshold(self, beta_threshold):
        """Set beta threshold for the adaptive agent."""
        if hasattr(self.agent, 'beta_threshold'):
            self.agent.beta_threshold = beta_threshold
            self.agent.use_beta_threshold = True
            self.logger.info("Beta threshold configured for adaptive trust system")
        else:
            self.logger.warning("Agent does not support beta threshold configuration")
    
    def configure_beta_threshold(self, beta_config):
        """Configure beta threshold with given config."""
        if hasattr(self.agent, 'beta_threshold') and self.agent.beta_threshold:
            # Update existing beta threshold config
            self.agent.beta_threshold.config = beta_config
            self.logger.info("Updated beta threshold configuration")
        else:
            self.logger.warning("No beta threshold system to configure")