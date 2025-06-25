"""
Trust monitoring components for decentralized federated learning.

This module provides HSIC-based trust drift detection to identify
and mitigate malicious behavior in federated learning systems.
"""

from murmura.trust.hsic import StreamingHSIC, ModelUpdateHSIC
from murmura.trust.trust_monitor import TrustMonitor, TrustAction, TrustLevel
from murmura.trust.trust_config import (
    HSICConfig,
    TrustPolicyConfig,
    TrustMonitoringConfig,
    create_default_trust_config,
    create_strict_trust_config,
    create_relaxed_trust_config,
)

__all__ = [
    # HSIC components
    "StreamingHSIC",
    "ModelUpdateHSIC",
    
    # Trust monitor
    "TrustMonitor",
    "TrustAction",
    "TrustLevel",
    
    # Configuration
    "HSICConfig",
    "TrustPolicyConfig",
    "TrustMonitoringConfig",
    "create_default_trust_config",
    "create_strict_trust_config",
    "create_relaxed_trust_config",
]