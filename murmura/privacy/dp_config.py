from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DPMechanism(str, Enum):
    """Supported differential privacy mechanisms"""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    UNIFORM = "uniform"


class DPAccountingMethod(str, Enum):
    """Differential privacy accounting methods"""
    RDP = "rdp"  # Rényi Differential Privacy (Opacus default)
    GDP = "gdp"  # Gaussian Differential Privacy
    PRIV_ACCOUNTANT = "prv"  # TensorFlow Privacy Accountant (if available)


class DPConfig(BaseModel):
    """
    Configuration for differential privacy settings.
    
    This configuration defines privacy parameters that work well with both
    MNIST and skin lesion classification while maintaining reasonable utility.
    """
    
    # Core DP parameters
    target_epsilon: float = Field(
        default=8.0,
        description="Target privacy budget (epsilon). Lower = more private",
        ge=0.1,
        le=50.0
    )
    
    target_delta: Optional[float] = Field(
        default=1e-5,
        description="Target privacy parameter (delta). Should be ~1/dataset_size",
        ge=1e-8,
        le=1e-3
    )
    
    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum L2 norm for gradient clipping",
        ge=0.1,
        le=10.0
    )
    
    noise_multiplier: Optional[float] = Field(
        default=None,
        description="Noise multiplier for DP. If None, computed automatically",
        ge=0.1,
        le=10.0
    )
    
    # Advanced parameters
    mechanism: DPMechanism = Field(
        default=DPMechanism.GAUSSIAN,
        description="Noise mechanism to use"
    )
    
    accounting_method: DPAccountingMethod = Field(
        default=DPAccountingMethod.RDP,
        description="Privacy accounting method"
    )
    
    secure_mode: bool = Field(
        default=False,
        description="Enable secure RNG and additional safety checks (requires torchcsprng)"
    )
    
    # Training-specific parameters
    sample_rate: Optional[float] = Field(
        default=None,
        description="Subsampling rate. If None, computed from batch_size/dataset_size",
        ge=0.001,
        le=1.0
    )
    
    epochs: Optional[int] = Field(
        default=None,
        description="Number of epochs for privacy accounting. If None, uses training config"
    )
    
    # Federated learning specific
    enable_client_dp: bool = Field(
        default=True,
        description="Enable DP at client level (local DP)"
    )
    
    enable_central_dp: bool = Field(
        default=False,
        description="Enable DP at central aggregation level"
    )
    
    # Advanced tuning
    alphas: Optional[list[float]] = Field(
        default=None,
        description="RDP orders for accounting. If None, uses Opacus defaults"
    )
    
    auto_tune_noise: bool = Field(
        default=True,
        description="Automatically tune noise multiplier to achieve target epsilon"
    )
    
    # Safety features
    strict_privacy_check: bool = Field(
        default=True,
        description="Perform strict privacy validation before training"
    )
    
    @model_validator(mode="after")
    def validate_dp_config(self) -> "DPConfig":
        """Validate DP configuration parameters"""
        
        # Basic parameter validation
        if self.target_delta is not None and self.target_delta >= 1.0:
            raise ValueError("Delta must be < 1.0 for meaningful privacy")
            
        if self.noise_multiplier is not None and self.noise_multiplier <= 0:
            raise ValueError("Noise multiplier must be positive")
            
        if self.sample_rate is not None and (self.sample_rate <= 0 or self.sample_rate > 1):
            raise ValueError("Sample rate must be in (0, 1]")
            
        # Federated learning validation
        if not self.enable_client_dp and not self.enable_central_dp:
            raise ValueError("At least one of client_dp or central_dp must be enabled")
            
        return self

    def get_privacy_spent_message(self, epsilon: float, delta: float) -> str:
        """Generate a human-readable privacy spent message"""
        return (
            f"Privacy spent: (ε={epsilon:.3f}, δ={delta:.2e}) "
            f"with target (ε={self.target_epsilon}, δ={self.target_delta})"
        )
    
    def is_privacy_exhausted(self, current_epsilon: float) -> bool:
        """Check if privacy budget is exhausted"""
        return current_epsilon >= self.target_epsilon
    
    @classmethod
    def create_for_mnist(cls) -> "DPConfig":
        """Create DP config optimized for MNIST (60k samples)"""
        return cls(
            target_epsilon=8.0,  # Reasonable privacy for MNIST
            target_delta=1e-5,   # ~1/60000
            max_grad_norm=1.0,
            auto_tune_noise=True,
            enable_client_dp=True,
            enable_central_dp=False,
        )
    
    @classmethod
    def create_for_skin_lesion(cls) -> "DPConfig":
        """Create DP config optimized for skin lesion datasets (~10k samples)"""
        return cls(
            target_epsilon=10.0,  # Slightly higher for smaller dataset
            target_delta=1e-4,    # ~1/10000
            max_grad_norm=1.2,    # Slightly higher for medical images
            auto_tune_noise=True,
            enable_client_dp=True,
            enable_central_dp=False,
        )
    
    @classmethod
    def create_high_privacy(cls) -> "DPConfig":
        """Create high privacy config (lower epsilon)"""
        return cls(
            target_epsilon=1.0,   # Very private
            target_delta=1e-6,
            max_grad_norm=0.8,
            auto_tune_noise=True,
            enable_client_dp=True,
            enable_central_dp=False,
        )