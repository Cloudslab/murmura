"""Base protocol for federated datasets."""

from typing import Protocol, List, runtime_checkable
from torch.utils.data import Dataset


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for federated datasets.

    Federated datasets must provide client-partitioned data access.
    """

    def get_client_data(self, client_id: int) -> Dataset:
        """Get dataset for a specific client.

        Args:
            client_id: Client index

        Returns:
            PyTorch Dataset containing this client's data
        """
        ...

    def get_num_clients(self) -> int:
        """Get total number of clients.

        Returns:
            Number of clients in the federated dataset
        """
        ...

    def get_client_partitions(self) -> List[List[int]]:
        """Get data partition indices for all clients.

        Returns:
            List where each element is a list of dataset indices for that client
        """
        ...
