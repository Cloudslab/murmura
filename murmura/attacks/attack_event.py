"""
Attack event classes for logging and monitoring model poisoning attacks.
"""

from typing import List, Dict, Optional, Any
from murmura.visualization.training_event import TrainingEvent


class AttackEvent(TrainingEvent):
    """Base class for attack events"""

    def __init__(
        self,
        round_num: int,
        attack_type: str,
        malicious_clients: List[int],
        attack_intensity: float,
        attack_config: Dict[str, Any]
    ):
        """
        Args:
            round_num (int): The current round number.
            attack_type (str): Type of attack being performed.
            malicious_clients (List[int]): List of malicious client indices.
            attack_intensity (float): Current attack intensity (0.0 to 1.0).
            attack_config (Dict[str, Any]): Attack configuration parameters.
        """
        super().__init__(round_num, f"attack_{attack_type}")
        self.attack_type = attack_type
        self.malicious_clients = malicious_clients
        self.attack_intensity = attack_intensity
        self.attack_config = attack_config
        self.num_malicious_clients = len(malicious_clients)


class LabelFlippingEvent(AttackEvent):
    """Event for label flipping attacks"""

    def __init__(
        self,
        round_num: int,
        malicious_clients: List[int],
        attack_intensity: float,
        attack_config: Dict[str, Any],
        samples_poisoned: Dict[int, int],  # client_id -> num_samples_poisoned
        target_label: Optional[int] = None,
        source_label: Optional[int] = None
    ):
        """
        Args:
            round_num (int): The current round number.
            malicious_clients (List[int]): List of malicious client indices.
            attack_intensity (float): Current attack intensity (0.0 to 1.0).
            attack_config (Dict[str, Any]): Attack configuration parameters.
            samples_poisoned (Dict[int, int]): Number of samples poisoned per client.
            target_label (Optional[int]): Target label for flipping.
            source_label (Optional[int]): Source label for flipping.
        """
        super().__init__(round_num, "label_flipping", malicious_clients, attack_intensity, attack_config)
        self.samples_poisoned = samples_poisoned
        self.target_label = target_label
        self.source_label = source_label
        self.total_samples_poisoned = sum(samples_poisoned.values())


class GradientManipulationEvent(AttackEvent):
    """Event for gradient manipulation attacks"""

    def __init__(
        self,
        round_num: int,
        malicious_clients: List[int],
        attack_intensity: float,
        attack_config: Dict[str, Any],
        parameters_modified: Dict[int, List[str]],  # client_id -> list of parameter names
        noise_scale: float,
        sign_flip_prob: float
    ):
        """
        Args:
            round_num (int): The current round number.
            malicious_clients (List[int]): List of malicious client indices.
            attack_intensity (float): Current attack intensity (0.0 to 1.0).
            attack_config (Dict[str, Any]): Attack configuration parameters.
            parameters_modified (Dict[int, List[str]]): Parameters modified per client.
            noise_scale (float): Scale factor for gradient noise.
            sign_flip_prob (float): Probability of flipping gradient signs.
        """
        super().__init__(round_num, "gradient_manipulation", malicious_clients, attack_intensity, attack_config)
        self.parameters_modified = parameters_modified
        self.noise_scale = noise_scale
        self.sign_flip_prob = sign_flip_prob
        self.total_parameters_modified = sum(len(params) for params in parameters_modified.values())


class AttackSummaryEvent(TrainingEvent):
    """Event for attack summary statistics"""

    def __init__(
        self,
        round_num: int,
        attack_statistics: Dict[str, Any],
        global_attack_metrics: Dict[str, Any]
    ):
        """
        Args:
            round_num (int): The current round number.
            attack_statistics (Dict[str, Any]): Attack statistics from all malicious clients.
            global_attack_metrics (Dict[str, Any]): Global attack metrics and summaries.
        """
        super().__init__(round_num, "attack_summary")
        self.attack_statistics = attack_statistics
        self.global_attack_metrics = global_attack_metrics


class AttackDetectionEvent(TrainingEvent):
    """Event for attack detection mechanisms"""

    def __init__(
        self,
        round_num: int,
        suspected_malicious_clients: List[int],
        detection_metrics: Dict[str, Any],
        detection_method: str,
        confidence_scores: Dict[int, float]  # client_id -> confidence score
    ):
        """
        Args:
            round_num (int): The current round number.
            suspected_malicious_clients (List[int]): Clients suspected of being malicious.
            detection_metrics (Dict[str, Any]): Metrics from detection algorithms.
            detection_method (str): Method used for detection.
            confidence_scores (Dict[int, float]): Confidence scores for suspected clients.
        """
        super().__init__(round_num, "attack_detection")
        self.suspected_malicious_clients = suspected_malicious_clients
        self.detection_metrics = detection_metrics
        self.detection_method = detection_method
        self.confidence_scores = confidence_scores