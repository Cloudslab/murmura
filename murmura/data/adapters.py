"""Dataset adapters for converting standard datasets to federated format."""

from typing import List, Optional
from torch.utils.data import Dataset, Subset


class DatasetAdapter:
    """Adapter to convert a standard PyTorch dataset to federated format.

    This class wraps a PyTorch Dataset and provides client-partitioned access.
    """

    def __init__(
        self,
        dataset: Dataset,
        client_partitions: List[List[int]],
    ):
        """Initialize dataset adapter.

        Args:
            dataset: PyTorch Dataset to wrap
            client_partitions: List of lists, where client_partitions[i] contains
                             the dataset indices for client i
        """
        self.dataset = dataset
        self.client_partitions = client_partitions
        self.num_clients = len(client_partitions)

    def get_client_data(self, client_id: int) -> Dataset:
        """Get dataset for a specific client.

        Args:
            client_id: Client index

        Returns:
            Subset of the dataset containing this client's data
        """
        if not 0 <= client_id < self.num_clients:
            raise ValueError(
                f"client_id {client_id} out of range [0, {self.num_clients})"
            )

        indices = self.client_partitions[client_id]
        return Subset(self.dataset, indices)

    def get_num_clients(self) -> int:
        """Get total number of clients."""
        return self.num_clients

    def get_client_partitions(self) -> List[List[int]]:
        """Get data partition indices for all clients."""
        return self.client_partitions


class TorchDatasetAdapter(DatasetAdapter):
    """Convenience alias for DatasetAdapter."""
    pass
