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

    # Escalating penalty trust decay options (formerly "polynomial decay")
    enable_escalating_penalty_decay: bool = Field(
        default=True,
        description="Use escalating penalty decay where punishment increases with repeated violations",
    )

    escalating_penalty_power: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Power for escalating penalty (higher = more aggressive penalty for repeated violations)",
    )

    escalating_penalty_base_factor: float = Field(
        default=0.95,
        ge=0.7,
        le=0.99,
        description="Base factor for escalating penalty decay scaling",
    )

    # Backward compatibility aliases (deprecated - use escalating_penalty_* instead)
    enable_polynomial_decay: bool = Field(
        default=None,
        description="DEPRECATED: Use enable_escalating_penalty_decay instead",
    )
    polynomial_decay_power: float = Field(
        default=None,
        description="DEPRECATED: Use escalating_penalty_power instead",
    )
    polynomial_decay_base_factor: float = Field(
        default=None,
        description="DEPRECATED: Use escalating_penalty_base_factor instead",
    )

    # Legacy decay options (kept for compatibility)
    enable_exponential_decay: bool = Field(
        default=False,
        description="Use exponential decay instead of polynomial for repeated violations",
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

    # Loss spoofing detection settings
    enable_loss_spoofing_detection: bool = Field(
        default=True,
        description="Enable detection of loss spoofing attacks using local validation",
    )

    local_validation_split_ratio: float = Field(
        default=0.1,
        ge=0.05,
        le=0.3,
        description="Ratio of training data to reserve for local validation (loss spoofing detection)",
    )

    spoofing_detection_threshold: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="Threshold for flagging loss spoofing based on validation discrepancy",
    )

    baseline_calibration_rounds: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of initial rounds to use for establishing distribution-aware baseline",
    )

    loss_ratio_tolerance: float = Field(
        default=3.0,
        ge=1.5,
        le=10.0,
        description="Maximum acceptable ratio between reported and validated loss before flagging",
    )

    min_loss_ratio_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.7,
        description="Minimum acceptable ratio between reported and validated loss (detects deflated reports)",
    )

    # Trust monitor resource monitoring
    enable_trust_resource_monitoring: bool = Field(
        default=False,
        description="Enable resource usage tracking specifically for trust monitor operations",
    )

    trust_resource_sampling_interval: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Sampling interval for trust monitor resource monitoring (seconds)",
    )

    max_trust_resource_history: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum number of resource measurements to keep in memory per trust monitor",
    )
