from pydantic import BaseModel
from typing import Literal


class OrchestrationConfig(BaseModel):
    """
    Configuration object for learning orchestration
    """

    num_actors: int = 10
    dataset_name: str = "mnist"
    partition_strategy: Literal["dirichlet", "iid"] = "dirichlet"
    alpha: float = 0.5
    min_partition_size: int = 100
    split: str = "train"
