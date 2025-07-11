"""
Configuration for trust monitoring in decentralized federated learning.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class TrustMonitorConfig(BaseModel):
    """Configuration for trust monitoring system."""
    
    # Core monitoring settings
    enable_trust_monitoring: bool = Field(
        default=False,
        description="Enable trust monitoring for malicious behavior detection"
    )
    
    # Historical data management
    history_window_size: int = Field(
        default=10,
        ge=3,
        description="Number of recent rounds to maintain for analysis"
    )
    
    # Detection sensitivity
    anomaly_detection_method: Literal["cusum", "zscore", "iqr"] = Field(
        default="cusum",
        description="Statistical method for anomaly detection"
    )
    
    # Neighbor comparison settings
    min_neighbors_for_consensus: int = Field(
        default=2,
        ge=1,
        description="Minimum neighbors required for consensus-based detection"
    )
    
    # Trust score settings
    initial_trust_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Initial trust score for new neighbors"
    )
    
    trust_decay_factor: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Factor for trust score decay when anomalies detected"
    )
    
    trust_recovery_factor: float = Field(
        default=1.02,
        ge=1.0,
        le=1.1,
        description="Factor for trust score recovery when behavior is normal"
    )
    
    # Alert thresholds (relative, not absolute)
    suspicion_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Trust score threshold for marking node as suspicious"
    )
    
    # Logging and monitoring
    log_trust_events: bool = Field(
        default=True,
        description="Enable detailed logging of trust events"
    )
    
    export_trust_metrics: bool = Field(
        default=True,
        description="Export trust metrics for analysis"
    )
    
    # Trust-weighted aggregation settings
    enable_trust_weighted_aggregation: bool = Field(
        default=True,
        description="Apply trust scores as weights during parameter aggregation"
    )
    
    trust_weight_exponent: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Exponent to apply to trust scores for weight calculation (higher = more aggressive)"
    )