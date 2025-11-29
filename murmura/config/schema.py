"""Configuration schema using Pydantic."""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Experiment-level configuration."""
    name: str = Field(description="Experiment name")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    rounds: int = Field(default=20, description="Number of training rounds")
    verbose: bool = Field(default=False, description="Enable verbose logging")


class TopologyConfig(BaseModel):
    """Network topology configuration."""
    type: Literal["ring", "fully", "erdos", "k-regular"] = Field(
        description="Topology type"
    )
    num_nodes: int = Field(description="Number of nodes in the network")
    p: Optional[float] = Field(default=None, description="Edge probability for Erdos-Renyi")
    k: Optional[int] = Field(default=None, description="Degree for k-regular graphs")
    seed: int = Field(default=12345, description="Random seed for topology generation")


class AggregationConfig(BaseModel):
    """Aggregation algorithm configuration."""
    algorithm: Literal["fedavg", "krum", "balance", "sketchguard", "ubar"] = Field(
        description="Aggregation algorithm"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm-specific parameters"
    )


class AttackConfig(BaseModel):
    """Attack configuration."""
    enabled: bool = Field(default=False, description="Enable Byzantine attacks")
    type: Optional[Literal["gaussian", "directed_deviation"]] = Field(
        default=None, description="Attack type"
    )
    percentage: float = Field(default=0.0, description="Fraction of nodes to compromise")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Attack-specific parameters"
    )


class TrainingConfig(BaseModel):
    """Training configuration."""
    local_epochs: int = Field(default=1, description="Local training epochs per round")
    batch_size: int = Field(default=64, description="Training batch size")
    lr: float = Field(default=0.01, description="Learning rate")
    max_samples: Optional[int] = Field(
        default=None,
        description="Maximum samples per client (None for all data)"
    )


class DataConfig(BaseModel):
    """Data configuration."""
    adapter: str = Field(description="Dataset adapter identifier (e.g., 'leaf.femnist')")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset-specific parameters"
    )


class ModelConfig(BaseModel):
    """Model configuration."""
    factory: str = Field(description="Model factory function or class path")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific parameters"
    )


class Config(BaseModel):
    """Main configuration object."""
    experiment: ExperimentConfig
    topology: TopologyConfig
    aggregation: AggregationConfig
    attack: AttackConfig = Field(default_factory=AttackConfig)
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields
