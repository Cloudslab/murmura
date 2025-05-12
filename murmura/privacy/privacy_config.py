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
    Configuration object for differential privacy in distributed learning.
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
        description="Noise multiplier for the differential privacy mechanism.",
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

    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Mechanism-specific parameters for differential privacy.",
    )

    @model_validator(mode="after")
    def validate_privacy_config(self) -> "PrivacyConfig":
        """
        Validate the privacy configuration.
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

        if self.mechanism_type == PrivacyMechanismType.GAUSSIAN:
            if self.noise_multiplier is None:
                self.noise_multiplier = 1.0

        return self
