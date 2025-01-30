from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, cast

import numpy as np
from numpy.typing import NDArray

from murmura.data_processing.dataset import MDataset


class Partitioner(ABC):
    """
    Abstract class that defines the interface for partitioner.
    """

    def __init__(self, num_partitions: int, seed: Optional[int] = 42):
        self.num_partitions = num_partitions
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.partitions: Dict[int, List[int]] = {}

    @abstractmethod
    def partition(
        self, dataset: MDataset, split_name: str, partition_by: Optional[str] = None
    ) -> None:
        """
        Partition the dataset into a specified number of partitions.

        :param dataset: Dataset to partition.
        :param split_name: Which split to partition.
        :param partition_by: Class to partition the data by
        """
        pass


class DirichletPartitioner(Partitioner):
    """
    Partitions data using Dirichlet distribution across classes.

    This implementation is inspired by the DirichletPartitioner in Flower
    Thank you for making FL accessible to the masses
    https://github.com/adap/flower/blob/main/datasets/flwr_datasets/partitioner/dirichlet_partitioner.py
    """

    def __init__(
        self,
        num_partitions: int,
        alpha: Union[float, List[float], NDArray[np.float64]],
        partition_by: str,
        min_partition_size: int = 10,
        self_balancing: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ):
        super().__init__(num_partitions, seed)
        self.alpha = self._validate_alpha(alpha)
        self.partition_by = partition_by
        self.min_partition_size = min_partition_size
        self.self_balancing = self_balancing
        self.shuffle = shuffle
        self.avg_samples_per_partition: Optional[float] = None

    def _validate_alpha(
        self, alpha: Union[float, List[float], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Validate and convert the alpha parameter to numpy array

        :param alpha: Alpha parameter.
        :return: Numpy array with alpha parameter.
        """
        if isinstance(alpha, (float, int)):
            return np.full(self.num_partitions, float(alpha), dtype=np.float64)
        if isinstance(alpha, (list, np.ndarray)):
            arr = np.array(alpha, dtype=np.float64)
            if len(arr) != self.num_partitions:
                raise ValueError("Alpha length must match number of partitions")
            if not np.all(arr > 0):
                raise ValueError("All alpha values must be > 0")
            return arr
        raise TypeError("Alpha must be float, list, or numpy array")

    def partition(
        self, dataset: MDataset, split_name: str, partition_by: Optional[str] = None
    ) -> None:
        """
        Partition the dataset into a specified number of partitions using Dirichlet distribution across classes.
        :param dataset: Dataset to partition.
        :param split_name: Split to partition within the dataset
        :param partition_by: class to partition the data by
        """
        partition_by = partition_by or self.partition_by

        split_dataset = dataset.get_split(split_name)
        targets = np.array(split_dataset[partition_by])
        unique_classes = np.unique(targets)
        self.avg_samples_per_partition = len(split_dataset) / self.num_partitions

        for attempt in range(10):
            self.partitions = {i: [] for i in range(self.num_partitions)}

            for cls in unique_classes:
                cls_indices = np.where(targets == cls)[0]
                n_class_samples = len(cls_indices)

                # Generate proportions and handle types explicitly
                proportions = self.rng.dirichlet(self.alpha)
                pid_to_prop = {
                    pid: float(proportions[pid]) for pid in range(self.num_partitions)
                }

                # Self-balancing logic
                if self.self_balancing:
                    for pid in pid_to_prop.copy():
                        if len(self.partitions[pid]) > self.avg_samples_per_partition:
                            pid_to_prop[pid] = 0.0

                    total = sum(pid_to_prop.values())
                    if total > 0:
                        pid_to_prop = {
                            pid: val / total for pid, val in pid_to_prop.items()
                        }
                    else:
                        equal_share = 1.0 / self.num_partitions
                        pid_to_prop = {pid: equal_share for pid in pid_to_prop}

                # Calculate split points safely
                cumulative_props = np.cumsum(list(pid_to_prop.values()))
                split_points = (cumulative_props[:-1] * n_class_samples).astype(int)

                # Handle splits with explicit type casting
                try:
                    splits = np.split(cls_indices, split_points)
                except ValueError:
                    splits = [
                        cls_indices,
                        *[
                            np.array([], dtype=np.int64)
                            for _ in range(self.num_partitions - 1)
                        ],
                    ]

                # Distribute indices with type-safe conversion
                for pid, indices_arr in enumerate(splits):
                    indices_list = cast(List[int], indices_arr.tolist())
                    self.partitions[pid].extend(indices_list)

            if min(len(v) for v in self.partitions.values()) >= self.min_partition_size:
                break
        else:
            raise RuntimeError(
                "Failed to meet minimum partition size after 10 attempts"
            )

        if self.shuffle:
            for pid in self.partitions:
                self.rng.shuffle(self.partitions[pid])

        dataset.add_partitions(split_name, cast(Dict[int, List[int]], self.partitions))


class IIDPartitioner(Partitioner):
    """Creates IID partitions by random sampling."""

    def __init__(
        self, num_partitions: int, shuffle: bool = True, seed: Optional[int] = 42
    ):
        super().__init__(num_partitions, seed)
        self.shuffle = shuffle

    def partition(
        self, dataset: MDataset, split_name: str, partition_by: Optional[str] = None
    ) -> None:
        """
        Partition the dataset into a specified number of equal, random partitions.

        :param dataset: Dataset to partition.
        :param split_name: Split to partition within the dataset
        :param partition_by: Class to partition the data by (Not required for IID)
        """
        hf_dataset = dataset.get_split(split_name)
        indices = np.arange(len(hf_dataset), dtype=np.int64)

        if self.shuffle:
            self.rng.shuffle(indices)

        split_arrays = np.array_split(indices, self.num_partitions)
        self.partitions = cast(Dict[int, List[int]], {pid: arr.tolist() for pid, arr in enumerate(split_arrays)})

        dataset.add_partitions(split_name, self.partitions)
