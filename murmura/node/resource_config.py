from typing import Optional, Literal, Dict, Any

from pydantic import BaseModel, Field


class ResourceConfig(BaseModel):
    """Configuration for resource allocation in multi-node clusters"""

    actors_per_node: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of virtual actors to create per physical node. If None, distributes evenly.",
    )
    gpus_per_actor: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="GPU resources per actor. If None, auto-calculated based on available resources.",
    )
    cpus_per_actor: Optional[float] = Field(
        default=1.0, gt=0.0, description="CPU resources per actor"
    )
    memory_per_actor: Optional[int] = Field(
        default=None,
        gt=0,
        description="Memory (MB) per actor. If None, uses Ray defaults.",
    )
    placement_strategy: Literal["spread", "pack", "strict_spread", "strict_pack"] = (
        Field(default="spread", description="Actor placement strategy across nodes")
    )
    node_affinity: Optional[Dict[str, Any]] = Field(
        default=None, description="Node affinity constraints for actor placement"
    )


class RayClusterConfig(BaseModel):
    """Configuration for Ray cluster setup"""

    address: Optional[str] = Field(
        default=None,
        description="Ray cluster address. If None, starts local cluster or connects to existing.",
    )
    namespace: str = Field(default="murmura", description="Ray namespace for isolation")
    include_dashboard: bool = Field(
        default=False, description="Whether to include Ray dashboard"
    )
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level for Ray workers"
    )
    runtime_env: Optional[Dict[str, Any]] = Field(
        default=None, description="Runtime environment configuration for Ray"
    )
    auto_detect_cluster: bool = Field(
        default=True, description="Whether to auto-detect multi-node cluster setup"
    )
