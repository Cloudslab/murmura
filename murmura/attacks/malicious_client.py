"""
Malicious Client Implementation for Gradual Label Flipping Attacks.

This module creates malicious FL clients that integrate seamlessly with the
federated learning process while applying gradual label flipping attacks.

Key considerations:
1. Malicious clients train on poisoned data (flipped labels)
2. Their trust monitors may incorrectly flag honest nodes as suspicious
3. They should behave normally in non-attack phases to avoid early detection
4. Attack intensity gradually increases to test trust monitor sensitivity
"""

import logging
import ray
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from murmura.attacks.gradual_label_flipping import GradualLabelFlippingAttack, AttackConfig
from murmura.attacks.gradual_model_poisoning import GradualModelPoisoningAttack, BackdoorConfig
from murmura.attacks.gradual_byzantine_gradient import GradualByzantineGradientAttack, ByzantineGradientConfig
from murmura.node.client_actor import VirtualClientActor


@ray.remote
class MaliciousClient(VirtualClientActor):
    """
    Malicious federated learning client that applies gradual attacks.
    
    This client extends the normal FL client with attack capabilities while
    maintaining the same interface to avoid detection. Supports multiple
    attack types including label flipping and model poisoning (backdoor).
    """
    
    def __init__(self, 
                 node_id: str,
                 attack_type: str,
                 attack_config: Any,
                 dataset_name: str = "mnist",
                 num_classes: int = 10,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 **kwargs):
        """
        Initialize malicious client.
        
        Args:
            node_id: Unique identifier for this malicious client
            attack_config: Configuration for the gradual attack
            dataset_name: Name of dataset (for attack configuration)
            num_classes: Number of classes in dataset
            **kwargs: Additional arguments for base FLNode
        """
        # Initialize base client actor
        super().__init__(client_id=node_id, **kwargs)
        
        # Attack components based on attack type
        self.attack_type = attack_type
        
        if attack_type == "label_flipping":
            self.attack = GradualLabelFlippingAttack(
                node_id=node_id,
                config=attack_config,
                num_classes=num_classes,
                dataset_name=dataset_name
            )
        elif attack_type == "model_poisoning":
            if input_shape is None:
                # Default shapes for common datasets
                input_shape = (28, 28) if dataset_name == "mnist" else (32, 32, 3)
            self.attack = GradualModelPoisoningAttack(
                node_id=node_id,
                config=attack_config,
                input_shape=input_shape,
                dataset_name=dataset_name
            )
        elif attack_type == "byzantine_gradient":
            self.attack = GradualByzantineGradientAttack(
                node_id=node_id,
                config=attack_config,
                dataset_name=dataset_name
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Malicious client state
        self.is_malicious = True
        self.attack_statistics = []
        self.detection_status = "undetected"  # "undetected", "suspected", "detected"
        
        # Training data cache (to apply consistent poisoning)
        self._poisoned_data_cache = {}
        self._original_data_cache = {}
        
        self.logger = logging.getLogger(f"murmura.attacks.MaliciousClient.{node_id}")
        self.logger.info(f"Initialized malicious client {node_id} with {attack_type} attack")
    
    def set_round_number(self, round_num: int) -> None:
        """
        Update for new FL round and attack progression.
        
        Args:
            round_num: Current federated learning round
        """
        # Update base class
        super().set_round_number(round_num)
        
        # Update attack for new round
        self.attack.update_round(round_num)
        
        # Clear cache for new round (attack intensity may have changed)
        self._poisoned_data_cache.clear()
        self._original_data_cache.clear()
        
        self.logger.debug(
            f"Round {round_num}: Attack phase = {self.attack.current_phase.value}, "
            f"intensity = {self.attack.current_intensity:.3f}"
        )
    
    def train_model(self, 
                   features: np.ndarray, 
                   labels: np.ndarray, 
                   **kwargs) -> Dict[str, Any]:
        """
        Train model on poisoned data during attack phases.
        
        Args:
            features: Training features
            labels: Training labels (will be poisoned)
            **kwargs: Additional training arguments
            
        Returns:
            Training results with attack statistics
        """
        # Cache original data for analysis
        data_key = f"round_{self.current_round}"
        if data_key not in self._original_data_cache:
            self._original_data_cache[data_key] = (features.copy(), labels.copy())
        
        # Apply attack based on type
        if self.attack.is_attacking():
            if self.attack_type == "label_flipping":
                poisoned_features, poisoned_labels, attack_stats = self.attack.poison_labels(
                    features, labels
                )
            elif self.attack_type == "model_poisoning":
                poisoned_features, poisoned_labels, attack_stats = self.attack.poison_data(
                    features, labels
                )
            elif self.attack_type == "byzantine_gradient":
                # For Byzantine gradient attacks, we train normally but manipulate parameters later
                poisoned_features, poisoned_labels = features, labels
                attack_stats = {
                    "attack_applied": True,
                    "phase": self.attack.current_phase.value,
                    "manipulation_deferred": True,  # Will manipulate parameters in get_model_parameters
                }
            
            # Cache poisoned data
            self._poisoned_data_cache[data_key] = (poisoned_features, poisoned_labels)
            
            # Store attack statistics
            self.attack_statistics.append(attack_stats)
            
            # Log attack application
            if attack_stats["labels_flipped"] > 0:
                self.logger.info(
                    f"Round {self.current_round}: Applied {attack_stats['phase']} attack, "
                    f"flipped {attack_stats['labels_flipped']} labels "
                    f"({attack_stats['flip_rate']:.1%})"
                )
            
            # Train on poisoned data
            training_results = super().train_model(poisoned_features, poisoned_labels, **kwargs)
            
        else:
            # No attack - train normally (dormant phase)
            attack_stats = {
                "attack_applied": False,
                "phase": self.attack.current_phase.value,
                "intensity": 0.0,
                "labels_flipped": 0,
                "flip_rate": 0.0,
            }
            self.attack_statistics.append(attack_stats)
            
            training_results = super().train_model(features, labels, **kwargs)
        
        # Add attack information to training results
        training_results["attack_stats"] = attack_stats
        training_results["is_malicious"] = True
        training_results["detection_status"] = self.detection_status
        
        return training_results
    
    def get_model_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get model parameters (potentially corrupted by attack).
        
        Returns:
            Model parameters dictionary
        """
        # Get parameters from base class
        parameters = super().get_model_parameters()
        
        # Apply Byzantine gradient attack to parameters if applicable
        if (self.attack_type == "byzantine_gradient" and 
            hasattr(self.attack, 'manipulate_parameters') and 
            self.attack.is_attacking()):
            
            parameters, manipulation_stats = self.attack.manipulate_parameters(parameters)
            
            # Log manipulation
            if manipulation_stats.get("parameters_manipulated", 0) > 0:
                self.logger.info(
                    f"Round {self.current_round}: Manipulated {manipulation_stats['parameters_manipulated']} parameters "
                    f"({manipulation_stats['phase']})"
                )
        
        # Add metadata about corruption
        if hasattr(self, '_parameter_metadata'):
            self._parameter_metadata.update({
                "corrupted_by_attack": self.attack.is_attacking(),
                "attack_phase": self.attack.current_phase.value,
                "attack_type": self.attack_type,
            })
            
            # Add type-specific metadata
            if self.attack_type in ["label_flipping", "model_poisoning"]:
                if hasattr(self.attack, 'total_labels_flipped'):
                    self._parameter_metadata["cumulative_poison_rate"] = (
                        self.attack.total_labels_flipped / 
                        max(1, self.attack.total_samples_processed)
                    )
                elif hasattr(self.attack, 'total_samples_poisoned'):
                    self._parameter_metadata["cumulative_poison_rate"] = (
                        self.attack.total_samples_poisoned / 
                        max(1, self.attack.total_samples_processed)
                    )
            elif self.attack_type == "byzantine_gradient":
                if hasattr(self.attack, 'total_parameters_manipulated'):
                    self._parameter_metadata["cumulative_manipulation_rate"] = (
                        self.attack.total_parameters_manipulated / 
                        max(1, self.attack.total_updates_processed)
                    )
        else:
            self._parameter_metadata = {
                "corrupted_by_attack": self.attack.is_attacking(),
                "attack_phase": self.attack.current_phase.value,
                "attack_type": self.attack_type,
            }
        
        return parameters
    
    def receive_model_update(self, 
                            neighbor_id: str, 
                            neighbor_parameters: Dict[str, np.ndarray],
                            **kwargs) -> None:
        """
        Receive model update from neighbor (may incorrectly assess honest nodes).
        
        This is where the key insight comes in: malicious nodes may flag honest
        nodes as suspicious because their corrupted models see honest updates as anomalous.
        
        Args:
            neighbor_id: ID of neighbor sending update
            neighbor_parameters: Neighbor's model parameters
            **kwargs: Additional arguments
        """
        # Process update normally (this may trigger trust assessment)
        super().receive_model_update(neighbor_id, neighbor_parameters, **kwargs)
        
        # Log potential false flags (for research analysis)
        if self.attack.is_attacking() and hasattr(self, 'trust_monitor'):
            # The malicious node's trust monitor may incorrectly flag honest nodes
            # This is expected behavior and demonstrates why malicious nodes' 
            # trust assessments should not be trusted
            self.logger.debug(
                f"Malicious node {self.node_id} processing update from {neighbor_id} "
                f"(may incorrectly assess honest node due to attack corruption)"
            )
    
    def get_attack_report(self) -> Dict[str, Any]:
        """
        Get comprehensive attack report for analysis.
        
        Returns:
            Dictionary with attack statistics and analysis
        """
        attack_stats = self.attack.get_attack_statistics()
        
        # Add client-specific information
        report = {
            **attack_stats,
            "detection_status": self.detection_status,
            "attack_statistics_per_round": self.attack_statistics.copy(),
            "training_rounds_completed": len(self.attack_statistics),
            "data_poisoning_summary": {
                "total_rounds_attacked": len([s for s in self.attack_statistics if s["attack_applied"]]),
                "max_flip_rate_achieved": max([s["flip_rate"] for s in self.attack_statistics], default=0),
                "average_flip_rate": np.mean([s["flip_rate"] for s in self.attack_statistics if s["attack_applied"]]) if any(s["attack_applied"] for s in self.attack_statistics) else 0,
            }
        }
        
        # Detection analysis
        if self.detection_status != "undetected":
            report["detection_analysis"] = {
                "expected_detection_round": self.attack.get_expected_detection_round(),
                "actual_detection_round": getattr(self, "_detection_round", None),
                "detection_efficiency": "early" if getattr(self, "_detection_round", float('inf')) < self.attack.get_expected_detection_round() else "late"
            }
        
        return report
    
    def mark_as_detected(self, detection_round: int, detection_confidence: float = 1.0) -> None:
        """
        Mark this malicious client as detected by trust monitoring.
        
        Args:
            detection_round: Round when detection occurred
            detection_confidence: Confidence level of detection
        """
        self.detection_status = "detected"
        self._detection_round = detection_round
        self._detection_confidence = detection_confidence
        
        self.logger.warning(
            f"Malicious client {self.node_id} detected at round {detection_round} "
            f"(confidence: {detection_confidence:.3f})"
        )
    
    def mark_as_suspected(self, suspicion_round: int, suspicion_level: float) -> None:
        """
        Mark this malicious client as suspected.
        
        Args:
            suspicion_round: Round when suspicion arose
            suspicion_level: Level of suspicion
        """
        if self.detection_status == "undetected":
            self.detection_status = "suspected"
            self._suspicion_round = suspicion_round
            self._suspicion_level = suspicion_level
            
            self.logger.info(
                f"Malicious client {self.node_id} under suspicion at round {suspicion_round} "
                f"(level: {suspicion_level:.3f})"
            )


def create_mixed_actors(num_actors: int,
                       malicious_fraction: float,
                       attack_type: str,
                       attack_config: Any,
                       dataset_name: str = "mnist",
                       num_classes: int = 10,
                       input_shape: Optional[Tuple[int, ...]] = None,
                       random_seed: int = 42,
                       **kwargs) -> List[ray.ObjectRef]:
    """
    Create a mixed population of honest and malicious actors.
    
    Args:
        num_actors: Total number of actors to create
        malicious_fraction: Fraction of actors that should be malicious (0.0 to 1.0)
        attack_type: Type of attack ("label_flipping" or "model_poisoning")
        attack_config: Configuration for malicious actors
        dataset_name: Dataset name
        num_classes: Number of classes in dataset
        input_shape: Shape of input data (for model poisoning)
        random_seed: Random seed for reproducible malicious actor selection
        **kwargs: Additional arguments for actor creation
        
    Returns:
        List of Ray actor references (mix of honest and malicious)
    """
    np.random.seed(random_seed)
    
    # Determine which actors should be malicious
    num_malicious = int(num_actors * malicious_fraction)
    malicious_indices = np.random.choice(num_actors, num_malicious, replace=False)
    
    actors = []
    
    for i in range(num_actors):
        node_id = f"node_{i}"
        
        if i in malicious_indices:
            # Create malicious actor
            actor = MaliciousClient.remote(
                node_id=node_id,
                attack_type=attack_type,
                attack_config=attack_config,
                dataset_name=dataset_name,
                num_classes=num_classes,
                input_shape=input_shape,
                **kwargs
            )
            logging.info(f"Created malicious actor: {node_id} ({attack_type})")
        else:
            # Create honest actor
            actor = VirtualClientActor.remote(client_id=node_id, **kwargs)
            logging.info(f"Created honest actor: {node_id}")
        
        actors.append(actor)
    
    logging.info(
        f"Created mixed population: {num_malicious} malicious, "
        f"{num_actors - num_malicious} honest actors"
    )
    
    return actors


def analyze_attack_effectiveness(attack_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze effectiveness of gradual label flipping attacks.
    
    Args:
        attack_reports: List of attack reports from malicious clients
        
    Returns:
        Analysis summary
    """
    if not attack_reports:
        return {"error": "No attack reports provided"}
    
    # Detection analysis
    detected_count = len([r for r in attack_reports if r["detection_status"] == "detected"])
    suspected_count = len([r for r in attack_reports if r["detection_status"] == "suspected"])
    undetected_count = len([r for r in attack_reports if r["detection_status"] == "undetected"])
    
    # Attack progression analysis
    max_intensity_reached = max([r["current_intensity"] for r in attack_reports])
    avg_poison_rate = np.mean([
        r["data_poisoning_summary"]["average_flip_rate"] 
        for r in attack_reports
    ])
    
    # Detection timing analysis
    detection_rounds = [
        r.get("detection_analysis", {}).get("actual_detection_round")
        for r in attack_reports 
        if r["detection_status"] == "detected"
    ]
    detection_rounds = [r for r in detection_rounds if r is not None]
    
    analysis = {
        "attack_summary": {
            "total_malicious_clients": len(attack_reports),
            "max_intensity_reached": max_intensity_reached,
            "average_poison_rate": avg_poison_rate,
        },
        "detection_summary": {
            "detected_clients": detected_count,
            "suspected_clients": suspected_count, 
            "undetected_clients": undetected_count,
            "detection_rate": detected_count / len(attack_reports),
            "false_negative_rate": undetected_count / len(attack_reports),
        },
        "detection_timing": {
            "average_detection_round": np.mean(detection_rounds) if detection_rounds else None,
            "earliest_detection": min(detection_rounds) if detection_rounds else None,
            "latest_detection": max(detection_rounds) if detection_rounds else None,
        },
        "attack_progression": {
            "clients_reached_moderate_phase": len([
                r for r in attack_reports 
                if "moderate" in [phase for phase in r["phase_history"]]
            ]),
            "clients_reached_aggressive_phase": len([
                r for r in attack_reports
                if "aggressive" in [phase for phase in r["phase_history"]]
            ]),
        }
    }
    
    return analysis