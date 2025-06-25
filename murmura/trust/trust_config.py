"""
Configuration for trust monitoring in decentralized federated learning.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class HSICConfig:
    """Configuration for HSIC algorithm."""
    
    window_size: int = 50
    kernel_type: str = "rbf"
    gamma: float = 0.1
    threshold: float = 0.1
    alpha: float = 0.9
    reduce_dim: bool = True
    target_dim: int = 100
    
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
    hsic_config: HSICConfig = field(default_factory=HSICConfig)
    trust_policy_config: TrustPolicyConfig = field(default_factory=TrustPolicyConfig)
    log_trust_metrics: bool = True
    trust_report_interval: int = 10  # Report every N rounds
    persist_trust_state: bool = False
    trust_state_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "hsic_config": self.hsic_config.to_dict(),
            "trust_policy_config": self.trust_policy_config.to_dict(),
            "log_trust_metrics": self.log_trust_metrics,
            "trust_report_interval": self.trust_report_interval,
            "persist_trust_state": self.persist_trust_state,
            "trust_state_path": self.trust_state_path,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrustMonitoringConfig":
        """Create from dictionary."""
        # Extract nested configs
        hsic_dict = config_dict.get("hsic_config", {})
        trust_policy_dict = config_dict.get("trust_policy_config", {})
        
        # Create config objects
        hsic_config = HSICConfig(**hsic_dict)
        trust_policy_config = TrustPolicyConfig(**trust_policy_dict)
        
        # Create main config
        return cls(
            enabled=config_dict.get("enabled", False),
            hsic_config=hsic_config,
            trust_policy_config=trust_policy_config,
            log_trust_metrics=config_dict.get("log_trust_metrics", True),
            trust_report_interval=config_dict.get("trust_report_interval", 10),
            persist_trust_state=config_dict.get("persist_trust_state", False),
            trust_state_path=config_dict.get("trust_state_path"),
        )


def create_default_trust_config() -> TrustMonitoringConfig:
    """Create default trust monitoring configuration."""
    return TrustMonitoringConfig(
        enabled=True,
        hsic_config=HSICConfig(),
        trust_policy_config=TrustPolicyConfig(),
    )


def create_strict_trust_config() -> TrustMonitoringConfig:
    """Create strict trust monitoring configuration with lower thresholds."""
    return TrustMonitoringConfig(
        enabled=True,
        hsic_config=HSICConfig(
            window_size=30,
            threshold=0.05,
            alpha=0.95,
        ),
        trust_policy_config=TrustPolicyConfig(
            warn_threshold=0.1,
            downgrade_threshold=0.2,
            exclude_threshold=0.3,
            min_samples_for_action=5,
            weight_reduction_factor=0.3,
        ),
    )


def create_relaxed_trust_config() -> TrustMonitoringConfig:
    """Create relaxed trust monitoring configuration with higher thresholds."""
    return TrustMonitoringConfig(
        enabled=True,
        hsic_config=HSICConfig(
            window_size=100,
            threshold=0.2,
            alpha=0.8,
        ),
        trust_policy_config=TrustPolicyConfig(
            warn_threshold=0.3,
            downgrade_threshold=0.5,
            exclude_threshold=0.7,
            min_samples_for_action=20,
            weight_reduction_factor=0.7,
        ),
    )