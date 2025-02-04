---
title: Data Pre-Processing
---
# Introduction

This document will walk you through the data pre-processing implementation in the murmura project. The purpose of this implementation is to provide a unified interface for loading and managing datasets, as well as partitioning them for various machine learning tasks.

We will cover:

1. How datasets are represented and managed.
2. The mechanism for loading datasets from different sources.
3. The partitioning strategies implemented and their configurations.

# Dataset representation and management

<SwmSnippet path="/murmura/data_processing/dataset.py" line="21">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> class is central to our dataset management. It provides a unified representation of datasets with split management capabilities.

```
class MDataset:
    def __init__(self, splits: DatasetDict):
        """
        Unified dataset representation with split management

        Args:
            splits: Hugging Face DatasetDict containing all splits
        """
        self._splits = splits
        self._partitions: Dict[str, Dict[int, List[int]]] = {}
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/dataset.py" line="32">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="33:3:3" line-data="    def available_splits(self) -&gt; List[str]:">`available_splits`</SwmToken> property allows us to retrieve the names of all available splits within the dataset, which is crucial for managing different data subsets.

```
    @property
    def available_splits(self) -> List[str]:
        """Get list of available split names"""
        return list(self._splits.keys())
```

---

</SwmSnippet>

# Loading datasets from different sources

<SwmSnippet path="/murmura/data_processing/dataset.py" line="11">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="11:2:2" line-data="class DatasetSource(Enum):">`DatasetSource`</SwmToken> enum defines the various data sources from which datasets can be loaded. This includes formats like CSV, JSON, and others.

```
class DatasetSource(Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    PANDAS = "pandas"
    DICT = "dict"
    LIST = "list"
    HUGGING_FACE = "hugging_face"
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/dataset.py" line="42">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="43:3:3" line-data="    def load(">`load`</SwmToken> method in the <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> class provides a consistent interface for loading data from these sources. It uses a dictionary to map each source type to its corresponding loading function.

```
    @classmethod
    def load(
        cls: type[T],
        source: DatasetSource,
        split: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Load data from specified source with consistent interface
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/dataset.py" line="70">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="70:3:3" line-data="    def _load_csv(">`_load_csv`</SwmToken> method is an example of how data is loaded from a CSV file. It reads the file into a pandas <SwmToken path="/murmura/data_processing/dataset.py" pos="92:15:15" line-data="        cls: type[T], data: pd.DataFrame, split: Optional[str] = None, **_: Any">`DataFrame`</SwmToken> and then creates an <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> instance from it.

```
    def _load_csv(
        cls: type[T], path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ) -> T:
        df = pd.read_csv(path, **kwargs)
        return cls._create_from_data(df, split)
```

---

</SwmSnippet>

# Partitioning strategies

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="10">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="10:2:2" line-data="class Partitioner(ABC):">`Partitioner`</SwmToken> abstract class defines the interface for partitioning datasets. It includes methods that must be implemented by any subclass to partition data.

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

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="35">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="35:2:2" line-data="class DirichletPartitioner(Partitioner):">`DirichletPartitioner`</SwmToken> class partitions data using a Dirichlet distribution. This is useful for creating non-IID partitions across classes.

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

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="82">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="82:3:3" line-data="    def partition(">`partition`</SwmToken> method in <SwmToken path="/murmura/data_processing/partitioner.py" pos="35:2:2" line-data="class DirichletPartitioner(Partitioner):">`DirichletPartitioner`</SwmToken> handles the logic for distributing data indices into partitions based on the Dirichlet distribution.

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

<SwmSnippet path="/murmura/data_processing/partitioner.py" line="161">

---

The <SwmToken path="/murmura/data_processing/partitioner.py" pos="161:2:2" line-data="class IIDPartitioner(Partitioner):">`IIDPartitioner`</SwmToken> class provides a simpler partitioning strategy by creating IID partitions through random sampling.

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

# Partitioner factory

<SwmSnippet path="/murmura/data_processing/partitioner_factory.py" line="9">

---

The <SwmToken path="/murmura/data_processing/partitioner_factory.py" pos="9:2:2" line-data="class PartitionerFactory:">`PartitionerFactory`</SwmToken> class is responsible for creating partitioner instances based on a given configuration. It supports both Dirichlet and IID partitioning strategies.

```
class PartitionerFactory:
    """
    Factory for creating partitioner instances based on configuration.
    """

    @staticmethod
    def create(config: OrchestrationConfig) -> Partitioner:
        """
        Create a partitioner instance based on the configuration.
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/data_processing/partitioner_factory.py" line="19">

---

The <SwmToken path="/murmura/data_processing/partitioner_factory.py" pos="15:3:3" line-data="    def create(config: OrchestrationConfig) -&gt; Partitioner:">`create`</SwmToken> method in <SwmToken path="/murmura/data_processing/partitioner_factory.py" pos="9:2:2" line-data="class PartitionerFactory:">`PartitionerFactory`</SwmToken> uses the configuration to determine which partitioner to instantiate, ensuring flexibility in partitioning strategy selection.

```
        :param config: Orchestration configuration containing partition strategy
        :return: Initialized partitioner instance
        """
        if config.partition_strategy == "dirichlet":
            return DirichletPartitioner(
                num_partitions=config.num_actors,
                alpha=config.alpha,
                partition_by="label",
                min_partition_size=config.min_partition_size,
            )
        elif config.partition_strategy == "iid":
            return IIDPartitioner(num_partitions=config.num_actors, shuffle=True)
        else:
            raise ValueError(
                f"Unsupported partition strategy {config.partition_strategy}"
            )
```

---

</SwmSnippet>

This walkthrough highlights the key components and design decisions in the data pre-processing implementation, focusing on dataset management and partitioning strategies.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBbXVybXVyYSUzQSUzQW11cnRhemFocg==" repo-name="murmura"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
