"""
Configuration for trust monitoring in decentralized federated learning.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class StatisticalConfig:
    """Configuration for robust statistical trust detection."""
    
    window_size: int = 20
    min_samples_for_detection: int = 5
    outlier_contamination: float = 0.1
    enable_adaptive_thresholds: bool = True
    # Detection thresholds for gradual attacks (balanced sensitivity)
    relative_difference_threshold: float = 1.0
    cosine_similarity_threshold: float = 0.9
    parameter_magnitude_threshold: float = 1.5
    statistical_outlier_threshold: float = -0.02
    temporal_consistency_threshold: float = 0.75
    
    # Adaptive detector settings
    use_adaptive_detector: bool = True
    learning_rate: float = 0.1
    warmup_rounds: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "min_samples_for_detection": self.min_samples_for_detection,
            "outlier_contamination": self.outlier_contamination,
            "enable_adaptive_thresholds": self.enable_adaptive_thresholds,
            "relative_difference_threshold": self.relative_difference_threshold,
            "cosine_similarity_threshold": self.cosine_similarity_threshold,
            "parameter_magnitude_threshold": self.parameter_magnitude_threshold,
            "statistical_outlier_threshold": self.statistical_outlier_threshold,
            "temporal_consistency_threshold": self.temporal_consistency_threshold,
            "use_adaptive_detector": self.use_adaptive_detector,
            "learning_rate": self.learning_rate,
            "warmup_rounds": self.warmup_rounds,
        }


@dataclass
class HSICConfig:
    """Configuration for HSIC algorithm with dynamic baseline calibration."""
    
    window_size: int = 50
    kernel_type: str = "rbf"
    gamma: float = 0.1
    threshold: float = 0.1  # Will be auto-calibrated
    alpha: float = 0.9
    reduce_dim: bool = True
    target_dim: int = 100
    calibration_rounds: int = 5  # Number of rounds to collect baseline data
    baseline_percentile: float = 95.0  # Percentile for threshold setting
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "kernel_type": self.kernel_type,
            "gamma": self.gamma,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "reduce_dim": self.reduce_dim,
            "target_dim": self.target_dim,
            "calibration_rounds": self.calibration_rounds,
            "baseline_percentile": self.baseline_percentile,
        }


@dataclass
class TrustPolicyConfig:
    """Configuration for trust policies."""
    
    warn_threshold: float = 0.15
    downgrade_threshold: float = 0.3
    exclude_threshold: float = 0.5
    reputation_window: int = 100
    min_samples_for_action: int = 10
    weight_reduction_factor: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "warn_threshold": self.warn_threshold,
            "downgrade_threshold": self.downgrade_threshold,
            "exclude_threshold": self.exclude_threshold,
            "reputation_window": self.reputation_window,
            "min_samples_for_action": self.min_samples_for_action,
            "weight_reduction_factor": self.weight_reduction_factor,
        }


@dataclass
class TrustMonitoringConfig:
    """Complete configuration for trust monitoring."""
    
    enabled: bool = False
    statistical_config: Optional[StatisticalConfig] = field(default_factory=StatisticalConfig)
    trust_policy_config: TrustPolicyConfig = field(default_factory=TrustPolicyConfig)
    log_trust_metrics: bool = True
    trust_report_interval: int = 10  # Report every N rounds
    persist_trust_state: bool = False
    trust_state_path: Optional[str] = None
    topology: str = "ring"  # Network topology for optimization
    
    # Ensemble detection configuration
    enable_ensemble_detection: bool = False
    num_classes: int = 10  # Number of classes in the dataset
    ensemble_weights: Optional[Dict[str, float]] = None  # Custom signal weights
    
    # Legacy HSIC support (deprecated)
    hsic_config: Optional[HSICConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "enabled": self.enabled,
            "trust_policy_config": self.trust_policy_config.to_dict(),
            "log_trust_metrics": self.log_trust_metrics,
            "trust_report_interval": self.trust_report_interval,
            "persist_trust_state": self.persist_trust_state,
            "trust_state_path": self.trust_state_path,
            "topology": self.topology,
            "enable_ensemble_detection": self.enable_ensemble_detection,
            "num_classes": self.num_classes,
            "ensemble_weights": self.ensemble_weights,
        }
        
        # Add statistical config if available
        if self.statistical_config is not None:
            result["statistical_config"] = self.statistical_config.to_dict()
        
        # Add legacy HSIC config if available (deprecated)
        if self.hsic_config is not None:
            result["hsic_config"] = self.hsic_config.to_dict()
            
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrustMonitoringConfig":
        """Create from dictionary."""
        # Extract nested configs
        statistical_dict = config_dict.get("statistical_config", {})
        trust_policy_dict = config_dict.get("trust_policy_config", {})
        hsic_dict = config_dict.get("hsic_config", {})  # Legacy support
        
        # Create config objects
        statistical_config = StatisticalConfig(**statistical_dict) if statistical_dict else None
        trust_policy_config = TrustPolicyConfig(**trust_policy_dict)
        hsic_config = HSICConfig(**hsic_dict) if hsic_dict else None
        
        # Create main config
        return cls(
            enabled=config_dict.get("enabled", False),
            statistical_config=statistical_config,
            trust_policy_config=trust_policy_config,
            log_trust_metrics=config_dict.get("log_trust_metrics", True),
            trust_report_interval=config_dict.get("trust_report_interval", 10),
            persist_trust_state=config_dict.get("persist_trust_state", False),
            trust_state_path=config_dict.get("trust_state_path"),
            topology=config_dict.get("topology", "ring"),
            enable_ensemble_detection=config_dict.get("enable_ensemble_detection", False),
            num_classes=config_dict.get("num_classes", 10),
            ensemble_weights=config_dict.get("ensemble_weights"),
            hsic_config=hsic_config,
        )


def create_default_trust_config() -> TrustMonitoringConfig:
    """Create default trust monitoring configuration with statistical detection."""
    return TrustMonitoringConfig(
        enabled=True,
        statistical_config=StatisticalConfig(),
        trust_policy_config=TrustPolicyConfig(),
    )


def create_strict_trust_config() -> TrustMonitoringConfig:
    """Create strict trust monitoring configuration with lower thresholds."""
    return TrustMonitoringConfig(
        enabled=True,
        statistical_config=StatisticalConfig(
            window_size=15,
            min_samples_for_detection=3,
            outlier_contamination=0.05,
            relative_difference_threshold=2.0,
            cosine_similarity_threshold=0.8,
        ),
        trust_policy_config=TrustPolicyConfig(
            warn_threshold=0.2,
            downgrade_threshold=0.4,
            exclude_threshold=0.6,
            min_samples_for_action=5,
            weight_reduction_factor=0.4,
        ),
    )


def create_relaxed_trust_config() -> TrustMonitoringConfig:
    """Create relaxed trust monitoring configuration with higher thresholds."""
    return TrustMonitoringConfig(
        enabled=True,
        statistical_config=StatisticalConfig(
            window_size=25,
            min_samples_for_detection=8,
            outlier_contamination=0.15,
            relative_difference_threshold=3.0,
            cosine_similarity_threshold=0.6,
        ),
        trust_policy_config=TrustPolicyConfig(
            warn_threshold=0.4,
            downgrade_threshold=0.6,
            exclude_threshold=0.8,
            min_samples_for_action=10,
            weight_reduction_factor=0.8,
        ),
    )