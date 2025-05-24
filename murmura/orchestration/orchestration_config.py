from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

from murmura.aggregation.aggregation_config import AggregationConfig
from murmura.network_management.topology import TopologyConfig
from murmura.node.resource_config import RayClusterConfig, ResourceConfig


class OrchestrationConfig(BaseModel):
    """
    Enhanced configuration object for learning orchestration with multi-node support
    """

    # Core configuration
    num_actors: int = Field(
        default=10, gt=0, description="Total number of virtual clients"
    )
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)

    # Dataset configuration
    dataset_name: str = Field(default="mnist", description="Dataset name")
    partition_strategy: Literal["dirichlet", "iid"] = Field(
        default="dirichlet", description="Data Partitioning strategy"
    )
    alpha: float = Field(
        default=0.5,
        gt=0,
        description="Dirichlet concentration parameter (only for dirichlet strategy)",
    )
    min_partition_size: int = Field(
        default=100,
        gt=0,
        description="Minimum samples per partition (only for dirichlet strategy)",
    )
    split: str = Field(default="train", description="Dataset split")

    # Multi-node Ray cluster configuration
    ray_cluster: RayClusterConfig = Field(
        default_factory=RayClusterConfig, description="Ray cluster configuration"
    )
    resources: ResourceConfig = Field(
        default_factory=ResourceConfig, description="Resource allocation configuration"
    )

    @model_validator(mode="after")
    def validate_multinode_config(self) -> "OrchestrationConfig":
        """Validate multi-node configuration consistency"""

        # If actors_per_node is specified, ensure it's compatible with total num_actors
        if self.resources.actors_per_node is not None:
            if self.num_actors < self.resources.actors_per_node:
                raise ValueError(
                    f"num_actors ({self.num_actors}) cannot be less than "
                    f"actors_per_node ({self.resources.actors_per_node})"
                )

        # Validate placement strategy
        valid_strategies = ["spread", "pack", "strict_spread", "strict_pack"]
        if self.resources.placement_strategy not in valid_strategies:
            raise ValueError(f"placement_strategy must be one of {valid_strategies}")

        return self

    @property
    def ray_address(self) -> Optional[str]:
        """Backward compatibility property"""
        return self.ray_cluster.address

    def get_resource_requirements(self) -> Dict[str, Any]:
        """Get resource requirements for actor creation"""
        resources = {}

        if self.resources.cpus_per_actor is not None:
            resources["num_cpus"] = self.resources.cpus_per_actor

        if self.resources.gpus_per_actor is not None:
            resources["num_gpus"] = self.resources.gpus_per_actor

        if self.resources.memory_per_actor is not None:
            resources["memory"] = (
                self.resources.memory_per_actor * 1024 * 1024
            )  # Convert MB to bytes

        return resources

    def get_placement_group_strategy(self) -> str:
        """Get Ray placement group strategy from our placement strategy"""
        strategy_mapping = {
            "spread": "SPREAD",
            "pack": "PACK",
            "strict_spread": "STRICT_SPREAD",
            "strict_pack": "STRICT_PACK",
        }
        return strategy_mapping[self.resources.placement_strategy]
