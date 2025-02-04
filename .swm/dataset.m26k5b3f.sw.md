---
title: 'Dataset '
---
# Introduction

This document will walk you through the implementation of the <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> class in the <SwmPath>[murmura/data_processing/dataset.py](/murmura/data_processing/dataset.py)</SwmPath> file. The <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> class provides a unified interface for managing datasets with various data sources and split management capabilities.

We will cover:

1. The rationale behind using an enumeration for dataset sources.
2. The design of the <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> class and its split management.
3. The loading mechanism for different data sources.
4. The partitioning functionality within the dataset.

# Dataset source enumeration

<SwmSnippet path="/murmura/data_processing/dataset.py" line="11">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="11:2:2" line-data="class DatasetSource(Enum):">`DatasetSource`</SwmToken> enumeration defines the supported data sources for loading datasets. This allows for a consistent interface when dealing with different data formats.

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

# <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> class design

<SwmSnippet path="/murmura/data_processing/dataset.py" line="21">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> class is designed to handle datasets with multiple splits, providing a unified representation and management of these splits.

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

# Available splits

<SwmSnippet path="/murmura/data_processing/dataset.py" line="32">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="33:3:3" line-data="    def available_splits(self) -&gt; List[str]:">`available_splits`</SwmToken> property provides a list of all available dataset splits, allowing users to easily query which splits are present.

```
    @property
    def available_splits(self) -> List[str]:
        """Get list of available split names"""
        return list(self._splits.keys())
```

---

</SwmSnippet>

# Partitions management

<SwmSnippet path="/murmura/data_processing/dataset.py" line="37">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="38:3:3" line-data="    def partitions(self) -&gt; Dict[str, Dict[int, List[int]]]:">`partitions`</SwmToken> property and related methods manage partitions within each split, enabling more granular control over dataset subsets.

```
    @property
    def partitions(self) -> Dict[str, Dict[int, List[int]]]:
        """Get all partitions across all splits"""
        return self._partitions
```

---

</SwmSnippet>

# Loading datasets

<SwmSnippet path="/murmura/data_processing/dataset.py" line="42">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="43:3:3" line-data="    def load(">`load`</SwmToken> class method provides a consistent interface for loading datasets from various sources, utilizing the <SwmToken path="/murmura/data_processing/dataset.py" pos="45:4:4" line-data="        source: DatasetSource,">`DatasetSource`</SwmToken> enumeration to determine the appropriate loading method.

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

<SwmSnippet path="/murmura/data_processing/dataset.py" line="52">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="54:11:11" line-data="            split: Split(s) to load (format depends on source)">`load`</SwmToken> method uses a dictionary to map each <SwmToken path="/murmura/data_processing/dataset.py" pos="57:6:6" line-data="        loader: Dict[DatasetSource, Callable[..., T]] = {">`DatasetSource`</SwmToken> to its corresponding loading function, ensuring that the correct method is called based on the source type.

```
        Args:
            source: Data source type (from Source enum)
            split: Split(s) to load (format depends on source)
            **kwargs: Source-specific parameters
        """
        loader: Dict[DatasetSource, Callable[..., T]] = {
            DatasetSource.CSV: cls._load_csv,
            DatasetSource.JSON: cls._load_json,
            DatasetSource.PARQUET: cls._load_parquet,
            DatasetSource.PANDAS: cls._load_pandas,
            DatasetSource.DICT: cls._load_dict,
            DatasetSource.LIST: cls._load_list,
            DatasetSource.HUGGING_FACE: cls._load_hf,
        }
```

---

</SwmSnippet>

# Loading from CSV

<SwmSnippet path="/murmura/data_processing/dataset.py" line="70">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="70:3:3" line-data="    def _load_csv(">`_load_csv`</SwmToken> method demonstrates how to load a dataset from a CSV file, converting it into the internal dataset representation.

```
    def _load_csv(
        cls: type[T], path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ) -> T:
        df = pd.read_csv(path, **kwargs)
        return cls._create_from_data(df, split)
```

---

</SwmSnippet>

# Loading from JSON

<SwmSnippet path="/murmura/data_processing/dataset.py" line="76">

---

Similarly, the <SwmToken path="/murmura/data_processing/dataset.py" pos="77:3:3" line-data="    def _load_json(">`_load_json`</SwmToken> method handles loading datasets from JSON files.

```
    @classmethod
    def _load_json(
        cls: type[T], path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ) -> T:
        df = pd.read_json(path, **kwargs)
        return cls._create_from_data(df, split)
```

---

</SwmSnippet>

# Loading from Parquet

<SwmSnippet path="/murmura/data_processing/dataset.py" line="83">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="84:3:3" line-data="    def _load_parquet(">`_load_parquet`</SwmToken> method is responsible for loading datasets from Parquet files.

```
    @classmethod
    def _load_parquet(
        cls: type[T], path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ) -> T:
        df = pd.read_parquet(path, **kwargs)
        return cls._create_from_data(df, split)
```

---

</SwmSnippet>

# Loading from Pandas <SwmToken path="/murmura/data_processing/dataset.py" pos="92:15:15" line-data="        cls: type[T], data: pd.DataFrame, split: Optional[str] = None, **_: Any">`DataFrame`</SwmToken>

<SwmSnippet path="/murmura/data_processing/dataset.py" line="90">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="91:3:3" line-data="    def _load_pandas(">`_load_pandas`</SwmToken> method allows loading datasets directly from a Pandas <SwmToken path="/murmura/data_processing/dataset.py" pos="92:15:15" line-data="        cls: type[T], data: pd.DataFrame, split: Optional[str] = None, **_: Any">`DataFrame`</SwmToken>.

```
    @classmethod
    def _load_pandas(
        cls: type[T], data: pd.DataFrame, split: Optional[str] = None, **_: Any
    ) -> T:
        return cls._create_from_data(data, split)
```

---

</SwmSnippet>

# Loading from dictionary

<SwmSnippet path="/murmura/data_processing/dataset.py" line="96">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="97:3:3" line-data="    def _load_dict(">`_load_dict`</SwmToken> method supports loading datasets from a dictionary structure.

```
    @classmethod
    def _load_dict(
        cls: type[T], data: Dict, split: Optional[str] = None, **_: Any
    ) -> T:
        return cls._create_from_data(data, split)
```

---

</SwmSnippet>

# Loading from list

<SwmSnippet path="/murmura/data_processing/dataset.py" line="102">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="103:3:3" line-data="    def _load_list(">`_load_list`</SwmToken> method enables loading datasets from a list.

```
    @classmethod
    def _load_list(
        cls: type[T], data: List, split: Optional[str] = None, **_: Any
    ) -> T:
        return cls._create_from_data(data, split)
```

---

</SwmSnippet>

# Loading from Hugging Face datasets

<SwmSnippet path="/murmura/data_processing/dataset.py" line="108">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="109:3:3" line-data="    def _load_hf(">`_load_hf`</SwmToken> method provides functionality to load datasets from the Hugging Face datasets library, handling both single and multiple splits.

```
    @classmethod
    def _load_hf(
        cls: type[T],
        dataset_name: str,
        split: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> T:
        try:
            if split is None:
                raise ValueError("split is not specified")
            dataset = load_dataset(dataset_name, split=split, **kwargs)
            if isinstance(dataset, list) and isinstance(split, list):
                dataset_dict = DatasetDict()
                for i, s in enumerate(split):
                    dataset_dict[s] = dataset[i]
                return cls(dataset_dict)
            return cls(DatasetDict({split or "train": dataset}))
        except ValueError as e:
            if "split" in str(e):
                full_dataset = load_dataset(dataset_name, **kwargs)
                return cls(full_dataset)
            raise
```

---

</SwmSnippet>

# Creating datasets from data

<SwmSnippet path="/murmura/data_processing/dataset.py" line="131">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="132:3:3" line-data="    def _create_from_data(">`_create_from_data`</SwmToken> method is a utility function that converts <SwmToken path="/murmura/data_processing/dataset.py" pos="135:10:12" line-data="        &quot;&quot;&quot;Create MDataset from in-memory data structures&quot;&quot;&quot;">`in-memory`</SwmToken> data structures into the internal dataset representation.

```
    @classmethod
    def _create_from_data(
        cls: type[T], data: Union[pd.DataFrame, Dict, List], split: Optional[str] = None
    ) -> T:
        """Create MDataset from in-memory data structures"""
        if isinstance(data, pd.DataFrame):
            dataset = Dataset.from_pandas(data)
        elif isinstance(data, dict):
            dataset = Dataset.from_dict(data)
        elif isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
```

---

</SwmSnippet>

# Accessing specific splits

<SwmSnippet path="/murmura/data_processing/dataset.py" line="148">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="148:3:3" line-data="    def get_split(self, split: str = &quot;train&quot;) -&gt; Dataset:">`get_split`</SwmToken> method allows users to access a specific dataset split, returning it as a Hugging Face <SwmToken path="/murmura/data_processing/dataset.py" pos="148:22:22" line-data="    def get_split(self, split: str = &quot;train&quot;) -&gt; Dataset:">`Dataset`</SwmToken>.

```
    def get_split(self, split: str = "train") -> Dataset:
        """Access specific split as Hugging Face Dataset"""
        return self._splits[split]
```

---

</SwmSnippet>

# Train-test split

<SwmSnippet path="/murmura/data_processing/dataset.py" line="152">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="152:3:3" line-data="    def train_test_split(">`train_test_split`</SwmToken> method facilitates creating new train and test splits from existing data, providing flexibility in dataset management.

```
    def train_test_split(
        self,
        source_split: str = "train",
        test_size: float = 0.2,
        seed: int = 42,
        new_split_names: tuple[str, str] = ("train", "test"),
    ) -> "MDataset":
        """Create new splits from existing data"""
        base_dataset = self._splits[source_split]
        splits = base_dataset.train_test_split(test_size=test_size, seed=seed)
        return MDataset(
            DatasetDict(
                {
                    **self._splits,
                    new_split_names[0]: splits["train"],
                    new_split_names[1]: splits["test"],
                }
            )
        )
```

---

</SwmSnippet>

# Managing partitions

<SwmSnippet path="/murmura/data_processing/dataset.py" line="172">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="172:3:3" line-data="    def get_partitions(self, split_name: str) -&gt; Dict[int, List[int]]:">`get_partitions`</SwmToken>, <SwmToken path="/murmura/data_processing/dataset.py" pos="177:3:3" line-data="    def add_partitions(self, split_name: str, partitions: Dict[int, List[int]]) -&gt; None:">`add_partitions`</SwmToken>, <SwmToken path="/murmura/data_processing/dataset.py" pos="181:3:3" line-data="    def list_partitioned_splits(self) -&gt; List[str]:">`list_partitioned_splits`</SwmToken>, and <SwmToken path="/murmura/data_processing/dataset.py" pos="185:3:3" line-data="    def clear_partitions(self, split_name: Optional[str] = None) -&gt; None:">`clear_partitions`</SwmToken> methods provide comprehensive functionality for managing partitions within dataset splits.

```
    def get_partitions(self, split_name: str) -> Dict[int, List[int]]:
        if split_name not in self._splits:
            raise KeyError(f"Split {split_name} does not exist")
        return self._partitions.get(split_name, {})

    def add_partitions(self, split_name: str, partitions: Dict[int, List[int]]) -> None:
        """Store partitions for a specific split"""
        self._partitions[split_name] = partitions

    def list_partitioned_splits(self) -> List[str]:
        """Get list of splits with existing partitions"""
        return list(self._partitions.keys())

    def clear_partitions(self, split_name: Optional[str] = None) -> None:
        """Remove partitions for a split or all splits"""
        if split_name:
            self._partitions.pop(split_name, None)
        else:
            self._partitions.clear()
```

---

</SwmSnippet>

# String representation

<SwmSnippet path="/murmura/data_processing/dataset.py" line="192">

---

The <SwmToken path="/murmura/data_processing/dataset.py" pos="192:3:3" line-data="    def __repr__(self) -&gt; str:">`__repr__`</SwmToken> method offers a string representation of the <SwmToken path="/murmura/data_processing/dataset.py" pos="193:7:7" line-data="        base = f&quot;MDataset(splits={self.available_splits}&quot;">`MDataset`</SwmToken> instance, including information about available splits and partitions.

```
    def __repr__(self) -> str:
        base = f"MDataset(splits={self.available_splits}"
        if self._partitions:
            partitioned_splits = len(self._partitions)
            return f"{base}, partitions={partitioned_splits} split{'s' if partitioned_splits > 1 else ''})"
        return f"{base})"
```

---

</SwmSnippet>

This concludes the walkthrough of the <SwmToken path="/murmura/data_processing/dataset.py" pos="21:2:2" line-data="class MDataset:">`MDataset`</SwmToken> class implementation, highlighting the key design decisions and functionalities.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBbXVybXVyYSUzQSUzQW11cnRhemFocg==" repo-name="murmura"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
