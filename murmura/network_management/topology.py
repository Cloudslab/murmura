from enum import Enum
from typing import Optional, Dict, List

from pydantic import BaseModel, Field, model_validator


class TopologyType(str, Enum):
    STAR = "star"
    RING = "ring"
    COMPLETE = "complete"
    LINE = "line"
    CUSTOM = "custom"


class TopologyConfig(BaseModel):
    """
    Configuration Model for Network Topology
    """

    topology_type: TopologyType = Field(
        default=TopologyType.COMPLETE,
        description="Type of network topology between clients",
    )
    hub_index: int = Field(
        default=0, description="Index of hub node (only for star topology)"
    )
    adjacency_list: Optional[Dict[int, List[int]]] = Field(
        default=None, description="Custom adjacency list for CUSTOM topology"
    )

    @model_validator(mode="after")
    def validate_topology(self) -> "TopologyConfig":
        if self.topology_type == TopologyType.STAR and self.hub_index < 0:
            raise ValueError("Hub index cannot be negative")

        if self.topology_type == TopologyType.CUSTOM:
            if not self.adjacency_list:
                raise ValueError("Adjacency list required for custom topology")
            for node, neighbors in self.adjacency_list.items():
                if any(n < 0 for n in neighbors):
                    raise ValueError("Negative node indices not allowed")

        return self
