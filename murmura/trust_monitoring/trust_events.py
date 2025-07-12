"""
Trust monitoring events for visualization and logging.
"""

from typing import Dict, Any
from murmura.visualization.training_event import TrainingEvent


class TrustEvent(TrainingEvent):
    """Base class for trust monitoring events."""
    
    def __init__(self, node_id: str, round_num: int, trust_scores: Dict[str, float]):
        super().__init__(round_num, "trust_monitoring")
        self.node_id = node_id
        self.trust_scores = trust_scores
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "trust_event",
            "node_id": self.node_id,
            "round_num": self.round_num,
            "trust_scores": self.trust_scores,
            "timestamp": self.timestamp,
            "step_name": self.step_name
        }


class TrustAnomalyEvent(TrustEvent):
    """Event triggered when anomalous behavior is detected."""
    
    def __init__(self, node_id: str, round_num: int, trust_scores: Dict[str, float],
                 suspected_neighbor: str, anomaly_type: str, anomaly_score: float, 
                 evidence: Dict[str, Any]):
        super().__init__(node_id, round_num, trust_scores)
        self.step_name = "trust_anomaly"
        self.suspected_neighbor = suspected_neighbor
        self.anomaly_type = anomaly_type
        self.anomaly_score = anomaly_score
        self.evidence = evidence
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "event_type": "trust_anomaly_event",
            "suspected_neighbor": self.suspected_neighbor,
            "anomaly_type": self.anomaly_type,
            "anomaly_score": self.anomaly_score,
            "evidence": self.evidence
        })
        return base_dict


class TrustScoreEvent(TrustEvent):
    """Event for tracking trust score changes over time."""
    
    def __init__(self, node_id: str, round_num: int, trust_scores: Dict[str, float],
                 score_changes: Dict[str, float], detection_method: str):
        super().__init__(node_id, round_num, trust_scores)
        self.step_name = "trust_score_update"
        self.score_changes = score_changes
        self.detection_method = detection_method
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "event_type": "trust_score_event", 
            "score_changes": self.score_changes,
            "detection_method": self.detection_method
        })
        return base_dict