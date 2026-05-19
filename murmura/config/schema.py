"""Configuration schema using Pydantic."""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field


class DistributedConfig(BaseModel):
    """Configuration for the ZeroMQ-based distributed backend."""

    transport: Literal["ipc", "tcp"] = Field(
        default="ipc",
        description="Transport layer: ipc (single machine) or tcp (multi-machine)",
    )
    ipc_dir: str = Field(
        default="/tmp/murmura",
        description="Base directory for IPC socket files (ipc transport only)",
    )
    # TCP transport settings
    host: str = Field(
        default="127.0.0.1",
        description="Coordinator host (tcp transport: address nodes connect to)",
    )
    coordinator_pub_port: int = Field(
        default=5500,
        description="Coordinator PUB socket port (broadcasts round signals)",
    )
    coordinator_pull_port: int = Field(
        default=5501,
        description="Coordinator PULL socket port (receives node metrics)",
    )
    base_port: int = Field(
        default=5550,
        description="Base port for node PULL sockets; node i uses base_port + i",
    )
    node_hosts: Optional[Dict[int, str]] = Field(
        default=None,
        description="Per-node host overrides for tcp transport: {node_id: host}",
    )
    # Timing
    round_duration_s: float = Field(
        default=60.0,
        description=(
            "Wall-clock budget per round in seconds. Must be long enough for "
            "local training + model exchange + evaluation on the slowest node."
        ),
    )
    startup_grace_s: float = Field(
        default=5.0,
        description="Seconds between process launch and the first round start, "
                    "giving all nodes time to bind sockets and connect.",
    )


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
    algorithm: Literal[
        "fedavg", "krum", "balance", "sketchguard", "ubar", "evidential_trust"
    ] = Field(description="Aggregation algorithm")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm-specific parameters"
    )


class AttackConfig(BaseModel):
    """Attack configuration."""
    enabled: bool = Field(default=False, description="Enable Byzantine attacks")
    type: Optional[Literal["gaussian", "directed_deviation", "topology_liar"]] = Field(
        default=None, description="Attack type"
    )
    percentage: float = Field(default=0.0, description="Fraction of nodes to compromise")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Attack-specific parameters"
    )


class MobilityConfig(BaseModel):
    """Random-walk mobility model for time-varying topology G^t."""
    area_size: float = Field(default=100.0, description="2-D arena side length")
    comm_range: float = Field(
        default=30.0,
        description="Communication range; edge (i,j) ∈ G^t iff dist(r_i^t, r_j^t) < comm_range",
    )
    max_speed: float = Field(
        default=5.0, description="Max node displacement per round (random walk step bound)"
    )
    seed: int = Field(default=42, description="RNG seed for initial positions and movement")
    ensure_connected: bool = Field(
        default=True,
        description="If True, connect isolated nodes to their nearest peer each round",
    )


class DMTTConfig(BaseModel):
    """DMTT protocol hyper-parameters (Section 4 of the DMTT paper)."""
    # Collaboration budget
    budget_B: int = Field(default=5, description="Max collaborators per round |C_i^t| ≤ B")

    # Link reliability EMA  (Algorithm 1, line: ĉ update)
    rho: float = Field(default=0.1, description="EMA smoothing factor ρ ∈ (0,1)")

    # Source trust (Algorithm 4: Beta evidence parameters)
    lambda_forget: float = Field(default=0.9, description="Forgetting factor λ ∈ (0,1]")
    w_d: float = Field(default=1.0, description="Weight for direct confirmation evidence w_d")
    w_c: float = Field(default=0.5, description="Weight for corroboration evidence w_c")
    w_x: float = Field(default=1.0, description="Weight for contradiction evidence w_x")
    # Topology trust T_ij^topo formula
    tau_U: float = Field(default=0.3, description="Uncertainty tolerance threshold τ_U")
    eta: float = Field(default=5.0, description="Uncertainty penalty scale η")

    # Model compatibility score (Section 4.3)
    w_a: float = Field(default=0.7, description="Accuracy weight in model score w_a ∈ [0,1]")
    tau_u: float = Field(default=0.5, description="Uncertainty threshold for model score τ_u")

    # Collaboration score weights (Section 4.4)
    lambda1: float = Field(default=0.4, description="Model compatibility weight λ_1")
    lambda2: float = Field(default=0.3, description="Topology trust weight λ_2")
    lambda3: float = Field(default=0.2, description="Link reliability weight λ_3")
    lambda4: float = Field(default=0.1, description="Communication cost weight λ_4")


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
    # Distributed backend (defaults to simulation for backward compatibility)
    backend: Literal["simulation", "distributed"] = Field(
        default="simulation",
        description="Execution backend: simulation (in-process) or distributed (ZMQ)",
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="ZMQ distributed backend settings (only used when backend=distributed)",
    )
    # Dynamic topology (optional; when present, activates mobility model in distributed mode)
    mobility: Optional[MobilityConfig] = Field(
        default=None,
        description="Mobility model settings; if set, topology varies per round via G^t",
    )
    # DMTT trust protocol (optional; when present, activates DMTTNodeProcess)
    dmtt: Optional[DMTTConfig] = Field(
        default=None,
        description="DMTT protocol settings; requires mobility to also be set",
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields
