"""
Configuration for trust monitoring in decentralized federated learning.
"""

from typing import Literal
from pydantic import BaseModel, Field


class TrustMonitorConfig(BaseModel):
    """Configuration for trust monitoring system."""

    # Core monitoring settings
    enable_trust_monitoring: bool = Field(
        default=False,
        description="Enable trust monitoring for malicious behavior detection",
    )

    # Historical data management
    history_window_size: int = Field(
        default=10, ge=3, description="Number of recent rounds to maintain for analysis"
    )

    # Detection sensitivity
    anomaly_detection_method: Literal["cusum", "zscore", "iqr"] = Field(
        default="cusum", description="Statistical method for anomaly detection"
    )

    # Neighbor comparison settings
    min_neighbors_for_consensus: int = Field(
        default=2,
        ge=1,
        description="Minimum neighbors required for consensus-based detection",
    )

    # Trust score settings
    initial_trust_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Initial trust score for new neighbors"
    )

    trust_decay_factor: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Factor for trust score decay when anomalies detected",
    )

    trust_recovery_factor: float = Field(
        default=1.02,
        ge=1.0,
        le=1.1,
        description="Factor for trust score recovery when behavior is normal",
    )

    # Enhanced recovery options
    enable_polynomial_recovery: bool = Field(
        default=True,
        description="Use polynomial (quadratic) recovery instead of linear for faster trust restoration",
    )

    polynomial_recovery_power: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Power for polynomial recovery (0.5 = square root, 1.0 = linear)",
    )

    # Alert thresholds (relative, not absolute)
    suspicion_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Trust score threshold for marking node as suspicious",
    )

    # Logging and monitoring
    log_trust_events: bool = Field(
        default=True, description="Enable detailed logging of trust events"
    )

    export_trust_metrics: bool = Field(
        default=True, description="Export trust metrics for analysis"
    )

    # Trust-weighted aggregation settings
    enable_trust_weighted_aggregation: bool = Field(
        default=True,
        description="Apply trust scores as weights during parameter aggregation",
    )

    trust_weight_exponent: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Exponent to apply to trust scores for weight calculation (higher = more aggressive)",
    )

    # Aggressive trust penalty options
    enable_exponential_decay: bool = Field(
        default=False,
        description="Use exponential decay instead of linear for repeated violations",
    )

    exponential_decay_base: float = Field(
        default=0.8,
        ge=0.1,
        le=0.95,
        description="Base for exponential decay (lower = more aggressive)",
    )

    trust_scaling_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Scaling factor for trust score to weight conversion (lower = more aggressive penalty)",
    )

    weight_normalization_method: str = Field(
        default="sigmoid",
        description="Method for normalizing trust weights: 'minmax', 'sigmoid', 'tanh', 'soft'",
    )

    sigmoid_steepness: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Steepness parameter for sigmoid normalization (higher = sharper transition)",
    )
