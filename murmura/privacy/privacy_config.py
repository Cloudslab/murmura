from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, model_validator


class PrivacyMechanismType(str, Enum):
    """
    Enumeration of privacy mechanism types.
    """

    GAUSSIAN = "gaussian"
    NONE = "none"


class PrivacyMode(str, Enum):
    """
    Enumeration of privacy modes.
    """

    LOCAL = "local"
    CENTRAL = "central"
    NONE = "none"


class PrivacyConfig(BaseModel):
    """
    Improved configuration object for differential privacy in distributed learning.
    """

    enabled: bool = Field(
        default=False,
        description="Enable differential privacy for the model.",
    )
    mechanism_type: PrivacyMechanismType = Field(
        default=PrivacyMechanismType.NONE,
        description="Type of differential privacy mechanism to use.",
    )

    privacy_mode: PrivacyMode = Field(
        default=PrivacyMode.NONE,
        description="Mode of differential privacy to use.",
    )

    target_epsilon: float = Field(
        default=8.0,  # Industry standard range is often 1-10, with 8 being reasonable.
        description="Target epsilon value for differential privacy.",
        gt=0,
    )

    target_delta: float = Field(
        default=1e-5,
        description="Target delta value for differential privacy.",
        gt=0,
        lt=1,
    )

    noise_multiplier: Optional[float] = Field(
        default=None,
        description="Noise multiplier for the differential privacy mechanism. If None, calculated adaptively.",
    )

    clipping_norm: Optional[float] = Field(
        default=None,
        description="L2 norm to clip gradients/parameters (if None, use adaptive clipping).",
    )

    adaptive_clipping_quantile: float = Field(
        default=0.9,
        description="Quantile for adaptive clipping (if adaptive clipping is used).",
        ge=0.0,
        le=1.0,
    )

    microbatches: int = Field(
        default=1,
        description="Number of microbatches for client-side processing",
        gt=0,
    )

    per_layer_clipping: bool = Field(
        default=True,
        description="Whether to clip each layer separately (True) or globally (False).",
    )

    accounting_mode: str = Field(
        default="rdp",
        description="Privacy accounting mode to use. (rdp=Renyi Differential Privacy)",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping (starting value for adaptive clipping).",
        gt=0,
    )

    # Flag for adaptive noise multiplier
    adaptive_noise: bool = Field(
        default=True,
        description="Whether to adaptively calibrate noise to meet target epsilon",
    )

    # Early stopping when privacy budget reached
    early_stopping: bool = Field(
        default=False,
        description="Whether to stop training early if privacy budget is exhausted",
    )

    # New field for privacy monitoring frequency
    monitor_frequency: int = Field(
        default=1,
        description="How often to recompute privacy budget (in rounds)",
        ge=1,
    )

    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Mechanism-specific parameters for differential privacy.",
    )

    @model_validator(mode="after")
    def validate_privacy_config(self) -> "PrivacyConfig":
        """
        Validate the privacy configuration with improved checks for parameter compatibility.
        """
        if self.params is None:
            self.params = {}

        if self.enabled:
            if self.mechanism_type == PrivacyMechanismType.NONE:
                self.mechanism_type = PrivacyMechanismType.GAUSSIAN

            if self.privacy_mode == PrivacyMode.NONE:
                raise ValueError(
                    "Privacy mode must be specified when differential privacy is enabled."
                )

            # Check noise and clipping norm compatibility
            if self.noise_multiplier is not None:
                # If noise is specified, validate it's in a reasonable range
                if self.noise_multiplier < 0.1:
                    print(
                        f"WARNING: Very low noise multiplier ({self.noise_multiplier}) may provide insufficient privacy."
                    )

                # If both noise and clipping are specified, check sensible ratio
                if self.clipping_norm is not None:
                    noise_clip_ratio = self.noise_multiplier / self.clipping_norm
                    if noise_clip_ratio < 0.1:
                        print(
                            f"WARNING: Noise-to-clipping ratio ({noise_clip_ratio:.4f}) is very low. "
                            + "This may not provide enough privacy relative to utility."
                        )
                    elif noise_clip_ratio > 10:
                        print(
                            f"WARNING: Noise-to-clipping ratio ({noise_clip_ratio:.4f}) is very high. "
                            + "This may destroy utility without providing meaningful additional privacy."
                        )

        if self.mechanism_type == PrivacyMechanismType.GAUSSIAN:
            # If no specific noise multiplier, set a reasonable default
            if self.noise_multiplier is None and not self.adaptive_noise:
                # Default to a reasonable value based on target epsilon
                if self.target_epsilon <= 1.0:
                    self.noise_multiplier = 2.0
                elif self.target_epsilon <= 3.0:
                    self.noise_multiplier = 1.2
                elif self.target_epsilon <= 8.0:
                    self.noise_multiplier = 0.8
                else:
                    self.noise_multiplier = 0.5

                print(
                    f"Setting default noise multiplier to {self.noise_multiplier} for target ε={self.target_epsilon}"
                )

        # Store rounds and epochs for proper accounting
        if "rounds" in self.params:
            print(f"Privacy config includes {self.params['rounds']} rounds")

        return self
