from enum import Enum
from typing import Union, Optional, Dict, List, Callable, TypeVar, Any
from pathlib import Path
import pandas as pd

from datasets import Dataset, DatasetDict, load_dataset  # type: ignore

T = TypeVar("T", bound="MDataset")


class DatasetSource(Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    PANDAS = "pandas"
    DICT = "dict"
    LIST = "list"
    HUGGING_FACE = "hugging_face"


class MDataset:
    def __init__(self, splits: DatasetDict):
        """
        Unified dataset representation with split management

        Args:
            splits: Hugging Face DatasetDict containing all splits
        """
        self._splits = splits
        self._partitions: Dict[str, Dict[int, List[int]]] = {}

    @property
    def available_splits(self) -> List[str]:
        """Get list of available split names"""
        return list(self._splits.keys())

    @property
    def partitions(self) -> Dict[str, Dict[int, List[int]]]:
        """Get all partitions across all splits"""
        return self._partitions

    @classmethod
    def load(
        cls: type[T],
        source: DatasetSource,
        split: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Load data from specified source with consistent interface

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

        return loader[source](split=split, **kwargs)

    @classmethod
    def _load_csv(
        cls: type[T], path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ) -> T:
        df = pd.read_csv(path, **kwargs)
        return cls._create_from_data(df, split)

    @classmethod
    def _load_json(
        cls: type[T], path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ) -> T:
        df = pd.read_json(path, **kwargs)
        return cls._create_from_data(df, split)

    @classmethod
    def _load_parquet(
        cls: type[T], path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ) -> T:
        df = pd.read_parquet(path, **kwargs)
        return cls._create_from_data(df, split)

    @classmethod
    def _load_pandas(
        cls: type[T], data: pd.DataFrame, split: Optional[str] = None, **_: Any
    ) -> T:
        return cls._create_from_data(data, split)

    @classmethod
    def _load_dict(
        cls: type[T], data: Dict, split: Optional[str] = None, **_: Any
    ) -> T:
        return cls._create_from_data(data, split)

    @classmethod
    def _load_list(
        cls: type[T], data: List, split: Optional[str] = None, **_: Any
    ) -> T:
        return cls._create_from_data(data, split)

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

        split_name = split or "train"
        return cls(DatasetDict({split_name: dataset}))

    def get_split(self, split: str = "train") -> Dataset:
        """Access specific split as Hugging Face Dataset"""
        return self._splits[split]

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

    def __repr__(self) -> str:
        base = f"MDataset(splits={self.available_splits}"
        if self._partitions:
            partitioned_splits = len(self._partitions)
            return f"{base}, partitions={partitioned_splits} split{'s' if partitioned_splits > 1 else ''})"
        return f"{base})"
