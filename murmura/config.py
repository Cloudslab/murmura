from pydantic import BaseModel, Field
from typing import Literal, Optional


class OrchestrationConfig(BaseModel):
    """
    Configuration object for learning orchestration
    """

    num_actors: int = Field(default=10, gt=0, description="Number of virtual clients")
    ray_address: Optional[str] = Field(default=None, description="Ray cluster address")
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
