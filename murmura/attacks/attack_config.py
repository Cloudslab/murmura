"""
Configuration classes for model poisoning attacks in federated learning research.
"""

from typing import Literal, Optional, List, Tuple
from pydantic import BaseModel, Field, model_validator


class AttackConfig(BaseModel):
    """Configuration for model poisoning attacks in federated learning research."""
    
    # Core attack parameters
    malicious_clients_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of clients to make malicious (0.0 = no attacks)"
    )
    
    attack_type: Optional[Literal["label_flipping", "gradient_manipulation", "fgsm", "pgd", "uap"]] = Field(
        default=None,
        description="Type of poisoning attack to perform"
    )
    
    # Attack intensity and progression
    attack_intensity_start: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Initial attack intensity (0.0 = no effect, 1.0 = maximum)"
    )
    
    attack_intensity_end: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Final attack intensity"
    )
    
    intensity_progression: Literal["linear", "exponential", "step"] = Field(
        default="linear",
        description="How attack intensity increases over rounds"
    )
    
    # Label flipping attack parameters
    label_flip_target: Optional[int] = Field(
        default=None,
        description="Target label for label flipping (if None, random flipping)"
    )
    
    label_flip_source: Optional[int] = Field(
        default=None,
        description="Source label to flip from (if None, flip any label)"
    )
    
    # Gradient manipulation parameters
    gradient_noise_scale: float = Field(
        default=5.0,
        ge=0.0,
        description="Scale factor for gradient noise injection"
    )
    
    gradient_sign_flip_prob: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability of flipping gradient signs"
    )
    
    # Attack targeting
    target_layers: Optional[List[str]] = Field(
        default=None,
        description="Specific model layers to target (if None, target all)"
    )
    
    # FGSM attack parameters
    fgsm_epsilon: float = Field(
        default=2.0,
        ge=0.0,
        description="Epsilon parameter for FGSM attack (perturbation magnitude)"
    )
    
    fgsm_targeted: bool = Field(
        default=False,
        description="Whether FGSM attack is targeted or untargeted"
    )
    
    fgsm_target_class: int = Field(
        default=0,
        ge=0,
        description="Target class for targeted FGSM attack"
    )
    
    fgsm_clip_values: Tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Min and max values for clipping FGSM adversarial examples"
    )
    
    fgsm_attack_mode: Literal["gradient", "weight"] = Field(
        default="gradient",
        description="Whether FGSM applies to gradients or weights"
    )
    
    # PGD attack parameters
    pgd_epsilon: float = Field(
        default=2.0,
        ge=0.0,
        description="Epsilon parameter for PGD attack (perturbation magnitude)"
    )
    
    pgd_alpha: float = Field(
        default=0.2,
        ge=0.0,
        description="Step size for PGD attack iterations"
    )
    
    pgd_num_iterations: int = Field(
        default=10,
        ge=1,
        description="Number of iterations for PGD attack"
    )
    
    pgd_random_start: bool = Field(
        default=True,
        description="Whether to use random start for PGD attack"
    )
    
    pgd_targeted: bool = Field(
        default=False,
        description="Whether PGD attack is targeted or untargeted"
    )
    
    pgd_target_class: int = Field(
        default=0,
        ge=0,
        description="Target class for targeted PGD attack"
    )
    
    pgd_clip_values: Tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Min and max values for clipping PGD adversarial examples"
    )
    
    pgd_norm_type: Literal["inf", "2", "1"] = Field(
        default="inf",
        description="Norm type for PGD attack (inf, 2, or 1)"
    )
    
    pgd_attack_mode: Literal["gradient", "weight"] = Field(
        default="gradient",
        description="Whether PGD applies to gradients or weights"
    )
    
    # UAP attack parameters
    uap_epsilon: float = Field(
        default=1.0,
        ge=0.0,
        description="Epsilon parameter for UAP attack (perturbation magnitude)"
    )
    
    uap_max_iterations: int = Field(
        default=50,
        ge=1,
        description="Maximum iterations for UAP computation"
    )
    
    uap_step_size: float = Field(
        default=0.1,
        ge=0.0,
        description="Step size for UAP optimization"
    )
    
    uap_fooling_rate: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Target fooling rate for UAP attack"
    )
    
    uap_norm_type: Literal["inf", "2", "1"] = Field(
        default="inf",
        description="Norm type for UAP attack (inf, 2, or 1)"
    )
    
    uap_clip_values: Tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Min and max values for clipping UAP adversarial examples"
    )
    
    uap_update_frequency: int = Field(
        default=10,
        ge=1,
        description="Update frequency for UAP (every N rounds)"
    )
    
    uap_max_samples: int = Field(
        default=100,
        ge=10,
        description="Maximum samples to accumulate for UAP computation"
    )
    
    uap_attack_mode: Literal["gradient", "weight"] = Field(
        default="gradient",
        description="Whether UAP applies to gradients or weights"
    )
    
    # Attack logging and monitoring
    log_attack_details: bool = Field(
        default=True,
        description="Enable detailed logging of attack actions"
    )
    
    attack_start_round: int = Field(
        default=1,
        ge=1,
        description="Round to start attacks (allows for baseline establishment)"
    )
    
    # Reproducibility
    malicious_node_seed: Optional[int] = Field(
        default=None,
        description="Seed for reproducible malicious node selection (None = random)"
    )
    
    @model_validator(mode="after")
    def validate_attack_config(self) -> "AttackConfig":
        """Validate attack configuration parameters."""
        
        # If attacks are enabled, require attack type
        if self.malicious_clients_ratio > 0.0 and self.attack_type is None:
            raise ValueError(
                "attack_type must be specified when malicious_clients_ratio > 0"
            )
        
        # Validate intensity progression
        if self.attack_intensity_start > self.attack_intensity_end:
            raise ValueError(
                "attack_intensity_start cannot be greater than attack_intensity_end"
            )
        
        # Validate label flipping parameters
        if self.attack_type == "label_flipping":
            if self.label_flip_target is not None and self.label_flip_target < 0:
                raise ValueError("label_flip_target must be non-negative")
            if self.label_flip_source is not None and self.label_flip_source < 0:
                raise ValueError("label_flip_source must be non-negative")
        
        # Validate gradient-based attack parameters
        gradient_attacks = ["fgsm", "pgd", "uap", "gradient_manipulation"]
        if self.attack_type in gradient_attacks:
            # Validate PGD parameters
            if self.attack_type == "pgd" and self.pgd_alpha > self.pgd_epsilon:
                raise ValueError("pgd_alpha (step size) should not exceed pgd_epsilon")
            
            # Validate clip values
            for attack in ["fgsm", "pgd", "uap"]:
                clip_attr = f"{attack}_clip_values"
                if hasattr(self, clip_attr):
                    clip_values = getattr(self, clip_attr)
                    if clip_values[0] >= clip_values[1]:
                        raise ValueError(f"{clip_attr} min must be less than max")
        
        return self
    
    def get_attack_intensity(self, current_round: int, total_rounds: int) -> float:
        """Calculate attack intensity for the current round."""
        
        if current_round < self.attack_start_round:
            return 0.0
        
        if self.malicious_clients_ratio == 0.0:
            return 0.0
        
        # Calculate progress through the attack phase
        attack_rounds = total_rounds - self.attack_start_round + 1
        if attack_rounds <= 1:
            return self.attack_intensity_start
        
        progress = (current_round - self.attack_start_round) / (attack_rounds - 1)
        progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
        
        # Apply progression curve
        if self.intensity_progression == "linear":
            intensity = self.attack_intensity_start + progress * (
                self.attack_intensity_end - self.attack_intensity_start
            )
        elif self.intensity_progression == "exponential":
            intensity = self.attack_intensity_start * (
                (self.attack_intensity_end / self.attack_intensity_start) ** progress
            )
        elif self.intensity_progression == "step":
            # Step function at 50% through the attack phase
            if progress < 0.5:
                intensity = self.attack_intensity_start
            else:
                intensity = self.attack_intensity_end
        else:
            intensity = self.attack_intensity_start
        
        return intensity
    
    def is_attack_active(self, current_round: int) -> bool:
        """Check if attacks should be active in the current round."""
        return (
            self.malicious_clients_ratio > 0.0
            and current_round >= self.attack_start_round
            and self.attack_type is not None
        )