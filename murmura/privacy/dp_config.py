from enum import Enum
from typing import Optional, Dict

from pydantic import BaseModel, Field, model_validator


class DPMechanism(str, Enum):
    """Available differential privacy mechanisms."""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    DISCRETE_GAUSSIAN = "discrete_gaussian"


class DPAccountant(str, Enum):
    """Privacy accounting methods."""
    RDP = "rdp"  # Rényi Differential Privacy
    ZCDP = "zcdp"  # Zero-Concentrated Differential Privacy
    PLD = "pld"  # Privacy Loss Distribution


class ClippingStrategy(str, Enum):
    """Gradient/parameter clipping strategies."""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    QUANTILE = "quantile"


class NoiseApplication(str, Enum):
    """Where to apply differential privacy noise."""
    SERVER_SIDE = "server_side"  # Central DP - noise added after aggregation
    CLIENT_SIDE = "client_side"  # Local DP - noise added before transmission


class DifferentialPrivacyConfig(BaseModel):
    """
    Configuration for differential privacy in federated/decentralized learning.

    This configuration supports both Central DP (server-side noise) and
    Local DP (client-side noise) following industry best practices.
    """

    # Core privacy parameters
    epsilon: float = Field(
        default=1.0,
        gt=0.0,
        description="Privacy budget (ε). Lower values = stronger privacy. "
                    "Common values: 0.1-1.0 (strong), 1.0-10.0 (moderate), >10.0 (weak)"
    )

    delta: Optional[float] = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Failure probability (δ). Should be < 1/n where n is dataset size. "
                    "If None, uses pure ε-DP. Common: 1e-5 to 1e-8"
    )

    # Mechanism configuration
    mechanism: DPMechanism = Field(
        default=DPMechanism.GAUSSIAN,
        description="Noise mechanism. Gaussian for (ε,δ)-DP, Laplace for pure ε-DP"
    )

    noise_application: NoiseApplication = Field(
        default=NoiseApplication.SERVER_SIDE,
        description="Where to apply noise. SERVER_SIDE for central DP, CLIENT_SIDE for local DP"
    )

    # Clipping configuration
    clipping_strategy: ClippingStrategy = Field(
        default=ClippingStrategy.ADAPTIVE,
        description="How to clip gradients/parameters before adding noise"
    )

    clipping_norm: Optional[float] = Field(
        default=1.0,
        gt=0.0,
        description="L2 clipping threshold. Ignored if using adaptive/quantile clipping"
    )

    target_quantile: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Target quantile for adaptive clipping (typically 0.5)"
    )

    # Advanced parameters
    noise_multiplier: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Noise multiplier (σ). If None, computed from ε, δ. "
                    "Gaussian noise std = noise_multiplier * clipping_norm"
    )

    accountant: DPAccountant = Field(
        default=DPAccountant.RDP,
        description="Privacy accounting method. RDP recommended for iterative algorithms"
    )

    # Client sampling (for privacy amplification)
    client_sampling_rate: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Fraction of clients sampled per round. Enables privacy amplification"
    )

    # Per-client privacy (user-level)
    per_client_clipping: bool = Field(
        default=False,
        description="Enable per-client clipping for user-level privacy"
    )

    max_clients_per_user: int = Field(
        default=1,
        ge=1,
        description="Maximum clients per user (for user-level privacy)"
    )

    # Monitoring and debugging
    enable_privacy_monitoring: bool = Field(
        default=True,
        description="Enable real-time privacy budget tracking"
    )

    privacy_budget_warning_threshold: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Warn when privacy budget consumption exceeds this fraction"
    )

    # Advanced composition
    total_rounds: Optional[int] = Field(
        default=None,
        gt=0,
        description="Total training rounds for budget allocation. Required for composition"
    )

    # Implementation parameters
    rdp_orders: Optional[list[float]] = Field(
        default=None,
        description="RDP orders for accounting. If None, uses default range [1.1, 2, 3, ..., 64]"
    )

    secure_aggregation: bool = Field(
        default=False,
        description="Enable secure aggregation protocol (requires additional setup)"
    )

    @model_validator(mode="after")
    def validate_dp_config(self) -> "DifferentialPrivacyConfig":
        """Validate differential privacy configuration parameters."""

        # Delta validation for Gaussian mechanism
        if self.mechanism == DPMechanism.GAUSSIAN and self.delta is None:
            raise ValueError(
                "Gaussian mechanism requires delta parameter. "
                "Use delta=1e-5 to 1e-8 for typical applications."
            )

        if self.mechanism == DPMechanism.LAPLACE and self.delta is not None:
            raise ValueError(
                "Laplace mechanism provides pure ε-DP and should not use delta. "
                "Set delta=None for Laplace mechanism."
            )

        # Noise multiplier computation
        if self.noise_multiplier is None:
            if self.mechanism == DPMechanism.GAUSSIAN and self.delta is not None:
                # Compute noise multiplier from ε, δ using standard conversion
                # This is a simplified version - production should use privacy accountants
                import math
                self.noise_multiplier = math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
            elif self.mechanism == DPMechanism.LAPLACE:
                # For Laplace: noise scale = sensitivity / epsilon
                self.noise_multiplier = 1.0 / self.epsilon
            else:
                raise ValueError(
                    "Cannot compute noise_multiplier. Please specify explicitly or provide valid ε, δ."
                )

        # Adaptive clipping validation
        if self.clipping_strategy != ClippingStrategy.FIXED and self.clipping_norm is None:
            self.clipping_norm = 1.0  # Default for adaptive strategies

        # Client sampling validation
        if self.client_sampling_rate is not None and self.noise_application == NoiseApplication.CLIENT_SIDE:
            # Client sampling with local DP doesn't provide privacy amplification
            import warnings
            warnings.warn(
                "Client sampling with LOCAL DP (client-side noise) does not provide "
                "privacy amplification. Consider using server-side noise for better utility.",
                UserWarning
            )

        # RDP orders default
        if self.rdp_orders is None:
            self.rdp_orders = [1.1] + list(range(2, 65))

        # Total rounds validation for composition
        if self.accountant in [DPAccountant.RDP, DPAccountant.ZCDP] and self.total_rounds is None:
            import warnings
            warnings.warn(
                "total_rounds not specified. Privacy composition will be computed "
                "incrementally, which may be less tight than pre-allocating budget.",
                UserWarning
            )

        return self

    def get_noise_scale(self) -> float:
        """Get the noise scale for the configured mechanism."""
        if self.clipping_norm is None:
            raise ValueError("clipping_norm must be set to compute noise scale")
        return self.noise_multiplier * self.clipping_norm

    def is_central_dp(self) -> bool:
        """Check if this configuration uses Central Differential Privacy."""
        return self.noise_application == NoiseApplication.SERVER_SIDE

    def is_local_dp(self) -> bool:
        """Check if this configuration uses Local Differential Privacy."""
        return self.noise_application == NoiseApplication.CLIENT_SIDE

    def is_client_side(self) -> bool:
        """Check if this configuration uses client-side noise application (Local DP)."""
        return self.noise_application == NoiseApplication.CLIENT_SIDE

    def is_server_side(self) -> bool:
        """Check if this configuration uses server-side noise application (Central DP)."""
        return self.noise_application == NoiseApplication.SERVER_SIDE

    def get_privacy_description(self) -> str:
        """Get human-readable description of privacy guarantees."""
        if self.delta is None:
            return f"Pure {self.epsilon}-differential privacy"
        else:
            return f"({self.epsilon}, {self.delta})-differential privacy"

    def estimate_utility_impact(self) -> Dict[str, str]:
        """Estimate the impact on model utility (rough guidelines)."""
        impact = {}

        # Privacy level assessment
        if self.epsilon <= 0.1:
            impact["privacy_level"] = "Very Strong"
            impact["utility_impact"] = "High - significant accuracy degradation expected"
        elif self.epsilon <= 1.0:
            impact["privacy_level"] = "Strong"
            impact["utility_impact"] = "Moderate - noticeable but manageable accuracy loss"
        elif self.epsilon <= 10.0:
            impact["privacy_level"] = "Moderate"
            impact["utility_impact"] = "Low to Moderate - minor accuracy loss"
        else:
            impact["privacy_level"] = "Weak"
            impact["utility_impact"] = "Minimal - slight accuracy loss"

        # Noise application impact
        if self.is_local_dp():
            impact["communication"] = "Higher noise required for same privacy level"
        else:
            impact["communication"] = "Efficient - server-side noise addition"

        return impact
