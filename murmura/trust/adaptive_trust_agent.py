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


@dataclass
class TrustContext:
    """Context information for adaptive threshold decisions."""
    
    # Federated Learning Context
    current_round: int
    total_rounds: int
    convergence_rate: float
    global_accuracy: float
    
    # Network Context  
    topology_type: str
    network_stability: float
    communication_latency: float
    
    # Historical Context
    recent_attack_rate: float
    false_positive_rate: float
    
    # Current Signal
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
    
    def __init__(self):
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
        
        self.logger.info("Initialized Adaptive Trust Agent with meta-learning capabilities")
        
    def _initialize_policy(self) -> np.ndarray:
        """Initialize simple linear policy weights with FL-friendly bias."""
        # 15 features as defined in _extract_features
        # Initialize with negative bias to favor "honest" decisions
        weights = np.random.normal(-0.5, 0.1, 15)  # Negative bias
        
        # Set specific weights for important features
        # Feature 7 is HSIC value - should have minimal negative weight since high HSIC is normal in FL
        weights[7] = -0.1  # Very small negative weight for HSIC
        
        return weights
    
    def get_trust_decision(self, context: TrustContext) -> Tuple[bool, float, str]:
        """
        Make trust decision based on full context.
        
        Args:
            context: Full FL and network context
            
        Returns:
            (is_malicious, confidence, reasoning)
        """
        # Extract normalized features
        features = self._extract_features(context)
        
        # Get malicious probability from policy
        malicious_prob = self._evaluate_policy(features)
        
        # Get adaptive threshold based on context
        threshold = self._get_adaptive_threshold(context)
        
        # Make decision
        is_malicious = malicious_prob > threshold
        confidence = abs(malicious_prob - threshold)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(context, malicious_prob, threshold)
        
        # Store context for learning
        self.context_history.append({
            'context': context,
            'features': features,
            'prob': malicious_prob,
            'threshold': threshold,
            'decision': is_malicious,
            'timestamp': time.time()
        })
        
        return is_malicious, confidence, reasoning
    
    def _extract_features(self, context: TrustContext) -> np.ndarray:
        """Extract normalized feature vector from context."""
        
        # Raw features
        features = np.array([
            # FL Progress indicators (0-1 normalized)
            context.current_round / max(1, context.total_rounds),
            context.convergence_rate,
            context.global_accuracy,
            
            # Network health indicators  
            context.network_stability,
            1.0 / (1.0 + context.communication_latency),  # Inverse latency
            
            # Historical risk indicators
            context.recent_attack_rate,
            context.false_positive_rate,
            
            # Current signal strength
            context.hsic_value,
            context.update_magnitude,
            context.update_direction_consistency,
            np.mean(context.neighbor_trust_scores) if context.neighbor_trust_scores else 0.5,
            np.std(context.neighbor_trust_scores) if len(context.neighbor_trust_scores) > 1 else 0.0,
            
            # Composite indicators
            self._compute_anomaly_score(context),
            self._compute_consistency_score(context),
            
            # Phase indicator
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
        
        Key insight: Threshold should vary based on:
        - Early rounds: More permissive (higher threshold)
        - Late rounds: More strict (lower threshold)  
        - High-risk scenarios: More strict
        - Recent false positives: More permissive
        """
        
        base_threshold = 0.7  # Much higher base threshold to reduce false positives
        
        # FL Phase adjustment - be more permissive across all phases
        fl_progress = context.current_round / max(1, context.total_rounds)
        if fl_progress < 0.3:  # Early rounds - very permissive
            phase_adjustment = 0.2
        elif fl_progress > 0.8:  # Late rounds - only slightly more strict
            phase_adjustment = -0.05  # Much less strict than before
        else:  # Middle rounds - still permissive
            phase_adjustment = 0.1
        
        # Risk adjustment
        risk_adjustment = context.recent_attack_rate * 0.2
        
        # False positive adjustment (CRITICAL for addressing current issue)
        fp_adjustment = context.false_positive_rate * 0.4
        
        # Network stability adjustment
        stability_adjustment = (1.0 - context.network_stability) * 0.1
        
        # HSIC-based adjustment (key insight: high HSIC is NORMAL in FL)
        # HSIC values 0.85-0.99 are completely normal in FL
        hsic_adjustment = 0.0
        if context.hsic_value > 0.995:  # Only flag EXTREMELY high values (almost perfect correlation)
            hsic_adjustment = -0.05  # Very mild penalty
        elif context.hsic_value < 0.8:  # Suspiciously low HSIC (very rare in normal FL)
            hsic_adjustment = -0.1
        elif 0.9 <= context.hsic_value <= 0.98:  # Normal FL range - reward this
            hsic_adjustment = 0.1  # Positive adjustment for normal behavior
        
        adaptive_threshold = (base_threshold + 
                            phase_adjustment - 
                            risk_adjustment + 
                            fp_adjustment - 
                            stability_adjustment +
                            hsic_adjustment)
        
        return np.clip(adaptive_threshold, 0.1, 0.9)
    
    def _compute_anomaly_score(self, context: TrustContext) -> float:
        """Compute composite anomaly score from multiple signals."""
        
        # HSIC anomaly (high values are NORMAL in FL!)
        fl_progress = context.current_round / max(1, context.total_rounds)
        # In normal FL, HSIC typically ranges 0.90-0.99, with 0.94+ being very common
        expected_hsic = 0.94  # Constant expected value since HSIC doesn't change much in normal FL
        
        # Only flag extreme deviations
        if context.hsic_value > 0.995:  # Extremely high (suspicious)
            hsic_anomaly = (context.hsic_value - 0.995) * 20  # Strong penalty for extreme values
        elif context.hsic_value < 0.85:  # Extremely low (suspicious)
            hsic_anomaly = (0.85 - context.hsic_value) * 10
        else:
            hsic_anomaly = 0.0  # Normal range - no anomaly
        
        # Update magnitude anomaly (very large or very small updates suspicious)
        # Normal updates are typically in range 0.001-0.1
        if context.update_magnitude > 0.1:
            magnitude_anomaly = min(1.0, (context.update_magnitude - 0.1) * 10)
        elif context.update_magnitude < 0.001:
            magnitude_anomaly = min(1.0, (0.001 - context.update_magnitude) * 1000)
        else:
            magnitude_anomaly = 0.0
        
        # Consistency anomaly
        consistency_anomaly = 1.0 - context.update_direction_consistency
        
        return np.mean([hsic_anomaly, magnitude_anomaly, consistency_anomaly])
    
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
        """Generate human-readable reasoning for the decision."""
        
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
        
        if context.false_positive_rate > 0.1:
            factors.append(f"Recent false positives ({context.false_positive_rate:.3f})")
        
        decision_str = 'MALICIOUS' if prob > threshold else 'HONEST'
        confidence = abs(prob - threshold)
        
        reasoning = f"Decision: {decision_str} (confidence: {confidence:.3f}). "
        reasoning += f"Factors: {', '.join(factors) if factors else 'Normal behavior pattern'}"
        
        return reasoning
    
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


class ContextTracker:
    """Tracks FL context for adaptive decision making."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ContextTracker")
        self.fl_history = deque(maxlen=100)
        self.network_stats = {}
        self.attack_history = deque(maxlen=50)
        self.false_positive_history = deque(maxlen=50)
        
    def build_context(self, node_id: str, update_data: Dict[str, Any]) -> TrustContext:
        """Build comprehensive context for trust decision."""
        
        # Extract or compute context components
        context = TrustContext(
            current_round=update_data.get('round', 0),
            total_rounds=update_data.get('total_rounds', 10),
            convergence_rate=self._compute_convergence_rate(),
            global_accuracy=update_data.get('accuracy', 0.5),
            topology_type=update_data.get('topology', 'ring'),
            network_stability=self._compute_network_stability(),
            communication_latency=self._compute_avg_latency(),
            recent_attack_rate=self._compute_recent_attack_rate(),
            false_positive_rate=self._compute_false_positive_rate(),
            hsic_value=update_data.get('hsic', 0.9),
            update_magnitude=update_data.get('update_norm', 0.01),
            update_direction_consistency=update_data.get('consistency', 0.8),
            neighbor_trust_scores=update_data.get('neighbor_trusts', [])
        )
        
        # Store for history tracking
        self.fl_history.append({
            'round': context.current_round,
            'accuracy': context.global_accuracy,
            'convergence_rate': context.convergence_rate,
            'timestamp': time.time()
        })
        
        return context
    
    def record_decision_outcome(self, decision: bool, actual_outcome: Optional[bool]):
        """Record decision outcome for false positive tracking."""
        if actual_outcome is not None:
            is_false_positive = decision and not actual_outcome
            self.false_positive_history.append(is_false_positive)
    
    def _compute_convergence_rate(self) -> float:
        """Compute how fast the global model is converging."""
        if len(self.fl_history) < 2:
            return 0.5
        
        recent_accuracies = [h['accuracy'] for h in list(self.fl_history)[-5:]]
        if len(recent_accuracies) < 2:
            return 0.5
        
        # Simple convergence rate based on accuracy improvement
        accuracy_trend = recent_accuracies[-1] - recent_accuracies[0]
        convergence_rate = min(1.0, max(0.0, accuracy_trend * 10 + 0.5))
        
        return convergence_rate
    
    def _compute_network_stability(self) -> float:
        """Compute network stability score."""
        # For now, return high stability (can be improved with actual network metrics)
        return 0.95
    
    def _compute_avg_latency(self) -> float:
        """Compute average communication latency."""
        # For now, return low latency (can be improved with actual measurements)
        return 0.1
    
    def _compute_recent_attack_rate(self) -> float:
        """Compute rate of attacks detected in recent rounds."""
        if len(self.attack_history) == 0:
            return 0.0
        
        recent_attacks = sum(self.attack_history)
        return recent_attacks / len(self.attack_history)
    
    def _compute_false_positive_rate(self) -> float:
        """Compute rate of false positives in recent rounds."""
        if len(self.false_positive_history) == 0:
            return 0.0
        
        recent_fps = sum(self.false_positive_history)
        return recent_fps / len(self.false_positive_history)


class DatasetIndependentTrustSystem:
    """
    Main trust system using adaptive thresholding.
    
    Key advantages:
    1. Dataset independent - learns from context, not absolute values
    2. Phase-aware - different behavior in early vs late FL rounds  
    3. Self-correcting - learns from false positives/negatives
    4. Explainable - provides reasoning for decisions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DatasetIndependentTrustSystem")
        self.agent = AdaptiveThresholdAgent()
        self.context_tracker = ContextTracker()
        
        self.logger.info("Initialized Dataset-Independent Trust System with adaptive agent")
        
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