---
title: Dataset Partitioner
---
# Introduction

This document will walk you through the implementation of the "Dataset Partitioner" in the murmura data processing module. The partitioner is designed to split datasets into multiple partitions using different strategies. We will explore the design decisions and the rationale behind the implementation.

We will cover:

1. Why an abstract base class is used for partitioning.
2. How the Dirichlet partitioning strategy is implemented.
3. How the IID partitioning strategy is implemented.

# Abstract base class for partitioning

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="10">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="10:2:2" line-data="class Partitioner(ABC):">`Partitioner`</SwmToken> class serves as an abstract base class, defining a common interface for all partitioning strategies. This ensures that any specific partitioner must implement the <SwmToken path="/murmura/data_processing/partitioner.py" pos="22:3:3" line-data="    def partition(">`partition`</SwmToken> method, providing flexibility to extend with different partitioning strategies.

```
class Partitioner(ABC):
    """
    Abstract class that defines the interface for partitioner.
    """

    def __init__(self, num_partitions: int, seed: Optional[int] = 42):
        self.num_partitions = num_partitions
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.partitions: Dict[int, List[int]] = {}
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="21">

---

The abstract method <SwmToken path="/murmura/data_processing/partitioner.py" pos="22:3:3" line-data="    def partition(">`partition`</SwmToken> is defined to enforce implementation in subclasses. This method is crucial as it dictates how datasets should be partitioned.

```
    @abstractmethod
    def partition(
        self, dataset: MDataset, split_name: str, partition_by: Optional[str] = None
    ) -> None:
        """
        Partition the dataset into a specified number of partitions.
```

---

</SwmSnippet>

# Dirichlet partitioning strategy

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="35">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="35:2:2" line-data="class DirichletPartitioner(Partitioner):">`DirichletPartitioner`</SwmToken> class extends the <SwmToken path="/murmura/data_processing/partitioner.py" pos="35:4:4" line-data="class DirichletPartitioner(Partitioner):">`Partitioner`</SwmToken> class to implement a partitioning strategy based on the Dirichlet distribution. This approach allows for non-IID (non-independent and identically distributed) partitioning, which is useful in federated learning scenarios.

```
class DirichletPartitioner(Partitioner):
    """
    Partitions data using Dirichlet distribution across classes.

    This implementation is inspired by the DirichletPartitioner in Flower
    Thank you for making FL accessible to the masses
    https://github.com/adap/flower/blob/main/datasets/flwr_datasets/partitioner/dirichlet_partitioner.py
    """
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="44">

---

The constructor initializes parameters specific to the Dirichlet partitioning, such as <SwmToken path="/murmura/data_processing/partitioner.py" pos="47:1:1" line-data="        alpha: Union[float, List[float], NDArray[np.float64]],">`alpha`</SwmToken>, which controls the concentration of the distribution, and <SwmToken path="/murmura/data_processing/partitioner.py" pos="50:1:1" line-data="        self_balancing: bool = True,">`self_balancing`</SwmToken>, which ensures partitions are balanced.

```
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
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="62">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="62:3:3" line-data="    def _validate_alpha(">`_validate_alpha`</SwmToken> method ensures that the <SwmToken path="/murmura/data_processing/partitioner.py" pos="63:4:4" line-data="        self, alpha: Union[float, List[float], NDArray[np.float64]]">`alpha`</SwmToken> parameter is correctly formatted and valid, which is essential for generating the Dirichlet distribution.

```
    def _validate_alpha(
        self, alpha: Union[float, List[float], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Validate and convert the alpha parameter to numpy array
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="82">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="82:3:3" line-data="    def partition(">`partition`</SwmToken> method in <SwmToken path="/murmura/data_processing/partitioner.py" pos="35:2:2" line-data="class DirichletPartitioner(Partitioner):">`DirichletPartitioner`</SwmToken> uses the Dirichlet distribution to allocate data points to partitions. It handles class distribution and self-balancing to maintain partition size constraints.

```
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
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="98">

---

The method iterates over classes, generating proportions for each partition and adjusting them to ensure balance and minimum partition size.

```
        for attempt in range(10):
            self.partitions = {i: [] for i in range(self.num_partitions)}

            for cls in unique_classes:
                cls_indices = np.where(targets == cls)[0]
                n_class_samples = len(cls_indices)
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="154">

---

Finally, the method shuffles the partitions if required and adds them to the dataset.

```
        if self.shuffle:
            for pid in self.partitions:
                self.rng.shuffle(self.partitions[pid])
```

---

</SwmSnippet>

# IID partitioning strategy

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="161">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="161:2:2" line-data="class IIDPartitioner(Partitioner):">`IIDPartitioner`</SwmToken> class provides a simple IID (independent and identically distributed) partitioning strategy by randomly sampling data points into partitions. This is useful for scenarios where uniform distribution is desired.

```
class IIDPartitioner(Partitioner):
    """Creates IID partitions by random sampling."""

    def __init__(
        self, num_partitions: int, shuffle: bool = True, seed: Optional[int] = 42
    ):
        super().__init__(num_partitions, seed)
        self.shuffle = shuffle
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="170">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="170:3:3" line-data="    def partition(">`partition`</SwmToken> method in <SwmToken path="/murmura/data_processing/partitioner.py" pos="161:2:2" line-data="class IIDPartitioner(Partitioner):">`IIDPartitioner`</SwmToken> randomly shuffles the dataset indices and splits them into equal partitions, ensuring each partition is of similar size.

```
    def partition(
        self, dataset: MDataset, split_name: str, partition_by: Optional[str] = None
    ) -> None:
        """
        Partition the dataset into a specified number of equal, random partitions.
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="192">

---

The partitions are then added to the dataset, completing the IID partitioning process.

```
        dataset.add_partitions(split_name, self.partitions)
```

---

</SwmSnippet>

This document has outlined the key design decisions and implementation details of the dataset partitioner, focusing on the abstract base class and the specific partitioning strategies.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBbXVybXVyYSUzQSUzQW11cnRhemFocg==" repo-name="murmura"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
