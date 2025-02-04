from typing import Dict, Any, Optional, List

import ray


@ray.remote
class VirtualClientActor:
    """Ray remote actor representing a virtual client in federated learning."""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.data_partition: Optional[List[int]] = None
        self.metadata: Dict[str, Any] = {}

    def receive_data(
        self, data_partition: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Receive a data partition and metadata dictionary.

        :param data_partition: DataPartition instance
        :param metadata: Metadata dictionary
        """
        self.data_partition = data_partition
        self.metadata = metadata if metadata is not None else {}
        return f"Client {self.client_id} received {len(data_partition)} samples"

    def get_data_info(self) -> Dict[str, Any]:
        """
        Return Information about stored data partition.
        :return:
        """
        return {
            "client_id": self.client_id,
            "data_size": len(self.data_partition) if self.data_partition else 0,
            "metadata": self.metadata,
        }
