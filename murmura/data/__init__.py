"""Data adapters for federated datasets."""

from murmura.data.base import DatasetProtocol
from murmura.data.adapters import DatasetAdapter, TorchDatasetAdapter

__all__ = ["DatasetProtocol", "DatasetAdapter", "TorchDatasetAdapter"]
