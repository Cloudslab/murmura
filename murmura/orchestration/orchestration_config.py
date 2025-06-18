from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field, model_validator

from murmura.aggregation.aggregation_config import AggregationConfig
from murmura.network_management.topology import TopologyConfig
from murmura.node.resource_config import RayClusterConfig, ResourceConfig
from murmura.defenses.defense_config import DefenseConfig


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
    dataset_name: str = Field(default="unknown", description="Dataset name")
    partition_strategy: Literal[
        "dirichlet",
        "iid",
        "sensitive_groups",
        "topology_correlated",
        "imbalanced_sensitive",
    ] = Field(default="dirichlet", description="Data Partitioning strategy")
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

    # Generic column configuration - NO DEFAULTS, must be specified by user
    feature_columns: Optional[List[str]] = Field(
        default=None,
        description="List of feature column names (must be specified for each dataset)",
    )
    label_column: Optional[str] = Field(
        default=None,
        description="Name of the label column (must be specified for each dataset)",
    )

    # Multi-node Ray cluster configuration
    ray_cluster: RayClusterConfig = Field(
        default_factory=RayClusterConfig, description="Ray cluster configuration"
    )
    resources: ResourceConfig = Field(
        default_factory=ResourceConfig, description="Resource allocation configuration"
    )

    # ADDED: Training parameters
    rounds: int = Field(
        default=5, gt=0, description="Number of federated learning rounds"
    )

    epochs: int = Field(default=1, gt=0, description="Number of local epochs per round")

    batch_size: int = Field(
        default=32, gt=0, description="Batch size for local training"
    )

    learning_rate: float = Field(
        default=0.001, gt=0.0, description="Learning rate for local training"
    )

    # ADDED: Test split parameter
    test_split: str = Field(default="test", description="Test dataset split name")

    # ADDED: Monitoring parameters
    monitor_resources: bool = Field(
        default=False, description="Enable resource usage monitoring"
    )

    health_check_interval: int = Field(
        default=5, gt=0, description="Interval (rounds) for actor health checks"
    )

    # ADDED: Subsampling parameters for privacy amplification
    client_sampling_rate: float = Field(
        default=1.0,
        ge=0.01,
        le=1.0,
        description="Fraction of clients to sample per round (l parameter from DP-FL papers)",
    )

    data_sampling_rate: float = Field(
        default=1.0,
        ge=0.01,
        le=1.0,
        description="Fraction of local data to sample per client per round (s parameter from DP-FL papers)",
    )

    enable_subsampling_amplification: bool = Field(
        default=False,
        description="Enable privacy amplification by subsampling in DP accounting",
    )

    # ADDED: Defense mechanism configuration
    defenses: DefenseConfig = Field(
        default_factory=DefenseConfig,
        description="Defense mechanism configuration (PSR, DCS, ATO)",
    )

    @model_validator(mode="after")
    def validate_multinode_config(self) -> "OrchestrationConfig":
        """Enhanced validation for multi-node configuration and dataset columns"""

        # Validate actors and nodes
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

        # CRITICAL: Validate dataset column configuration
        if self.feature_columns is None:
            raise ValueError(
                "feature_columns must be specified for the dataset. "
                "This is required for generic data handling across different datasets. "
                "Examples:\n"
                "  - For image datasets: feature_columns=['image']\n"
                "  - For text datasets: feature_columns=['text'] or ['input_ids']\n"
                "  - For tabular datasets: feature_columns=['feature1', 'feature2', ...]\n"
                "  - For multi-modal: feature_columns=['image', 'text']"
            )

        if self.label_column is None:
            raise ValueError(
                "label_column must be specified for the dataset. "
                "This is required for supervised learning tasks. "
                "Examples:\n"
                "  - Most datasets: label_column='label'\n"
                "  - Medical datasets: label_column='dx' or label_column='diagnosis'\n"
                "  - Text classification: label_column='sentiment' or label_column='category'"
            )

        # Validate that feature_columns is a non-empty list
        if not isinstance(self.feature_columns, list) or len(self.feature_columns) == 0:
            raise ValueError(
                "feature_columns must be a non-empty list of column names. "
                f"Got: {self.feature_columns} (type: {type(self.feature_columns)})"
            )

        # Validate that label_column is a non-empty string
        if (
            not isinstance(self.label_column, str)
            or len(self.label_column.strip()) == 0
        ):
            raise ValueError(
                "label_column must be a non-empty string. "
                f"Got: {self.label_column} (type: {type(self.label_column)})"
            )

        # Validate that feature columns don't contain the label column
        if self.label_column in self.feature_columns:
            raise ValueError(
                f"label_column '{self.label_column}' cannot be included in feature_columns {self.feature_columns}. "
                "Features and labels must be separate columns."
            )

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
