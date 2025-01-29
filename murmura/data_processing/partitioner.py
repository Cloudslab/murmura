from abc import ABC, abstractmethod
from typing import List

from murmura.data_processing.dataset import MDataset


class Partitioner(ABC):
    @abstractmethod
    def partition(
            self, dataset: MDataset, num_partitions: int, seed: int) -> List[MDataset]:
        """
        Partitions the dataset into multiple subsets.

        :param dataset: MDataset to be partitioned
        :param num_partitions: Number of partitions
        :param seed: Optional seed for reproducibility
        :return: List of Partitions
        """
        pass
