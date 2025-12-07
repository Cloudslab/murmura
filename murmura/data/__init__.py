"""Data adapters for federated datasets."""

from murmura.data.base import DatasetProtocol
from murmura.data.adapters import DatasetAdapter, TorchDatasetAdapter
from murmura.data.partitioners import (
    dirichlet_partition,
    iid_partition,
    natural_partition,
    combine_partitions_with_dirichlet,
)

__all__ = [
    "DatasetProtocol",
    "DatasetAdapter",
    "TorchDatasetAdapter",
    "dirichlet_partition",
    "iid_partition",
    "natural_partition",
    "combine_partitions_with_dirichlet",
]
