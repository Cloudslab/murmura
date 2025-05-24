import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Dict, List, Callable, TypeVar, Any

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
    def __init__(self, splits: DatasetDict, dataset_metadata: Optional[Dict[str, Any]] = None):
        """
        Unified dataset representation with split management and multi-node support

        Args:
            splits: Hugging Face DatasetDict containing all splits
            dataset_metadata: Metadata for recreating dataset in multi-node environments
        """
        self._splits = splits
        self._partitions: Dict[str, Dict[int, List[int]]] = {}
        self._dataset_metadata = dataset_metadata or {}
        self._logger = logging.getLogger("murmura.dataset")

    @property
    def available_splits(self) -> List[str]:
        """Get list of available split names"""
        return list(self._splits.keys())

    @property
    def partitions(self) -> Dict[str, Dict[int, List[int]]]:
        """Get all partitions across all splits"""
        return self._partitions

    @property
    def dataset_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata for multi-node reconstruction"""
        return self._dataset_metadata

    def is_serializable_for_multinode(self) -> bool:
        """Check if dataset can be safely serialized across nodes"""
        try:
            # Check if this is a HuggingFace dataset that might have file references
            if self._dataset_metadata.get("source") == DatasetSource.HUGGING_FACE:
                return True  # We'll handle HF datasets specially

            # For other datasets, try a small serialization test
            test_split = list(self._splits.keys())[0]
            test_sample = self._splits[test_split][:1]  # Just one sample
            pickle.dumps(test_sample)
            return True
        except Exception as e:
            self._logger.warning(f"Dataset not serializable for multi-node: {e}")
            return False

    def prepare_for_multinode_distribution(self) -> "MDataset":
        """
        Prepare dataset for distribution across multiple nodes.
        This converts memory-mapped datasets to serializable format.
        """
        if self.is_serializable_for_multinode() and self._dataset_metadata.get("source") != DatasetSource.HUGGING_FACE:
            return self  # Already safe to distribute

        self._logger.info("Converting dataset to multi-node compatible format...")

        # Convert each split to in-memory format
        serializable_splits = DatasetDict()

        for split_name, split_dataset in self._splits.items():
            self._logger.debug(f"Converting split '{split_name}' to serializable format...")

            # Convert to pandas then back to dataset to break file dependencies
            try:
                df = split_dataset.to_pandas()
                serializable_splits[split_name] = Dataset.from_pandas(df)
            except Exception as e:
                # Fallback: convert to dict then back to dataset
                self._logger.debug(f"Pandas conversion failed, using dict method: {e}")
                data_dict = {}
                for key in split_dataset.column_names:
                    data_dict[key] = split_dataset[key]
                serializable_splits[split_name] = Dataset.from_dict(data_dict)

        self._logger.info("Dataset conversion completed")

        # Create new MDataset with serializable data
        new_dataset = MDataset(serializable_splits, self._dataset_metadata.copy())
        new_dataset._partitions = self._partitions.copy()

        return new_dataset

    @classmethod
    def load_dataset_with_multinode_support(cls: type[T], source, dataset_name, split) -> T:
        """
        Backward compatible helper that users can drop into existing code
        """
        try:
            # Try the enhanced multinode approach first
            return MDataset.create_multinode_compatible(
                source, dataset_name=dataset_name, split=split
            )
        except Exception:
            # Fallback to original approach
            return MDataset.load(source, dataset_name=dataset_name, split=split)

    @classmethod
    def create_multinode_compatible(
            cls: type[T],
            source: DatasetSource,
            split: Optional[Union[str, List[str]]] = None,
            force_serializable: bool = True,
            **kwargs: Any,
    ) -> T:
        """
        Create a dataset that's optimized for multi-node distribution.
        
        Args:
            source: Data source type
            split: Split(s) to load
            force_serializable: Whether to force conversion to serializable format
            **kwargs: Source-specific parameters
        """
        # First load normally
        dataset = cls.load(source, split, **kwargs)

        # Store metadata for potential reconstruction
        dataset._dataset_metadata.update({
            "source": source,
            "split": split,
            "kwargs": kwargs,
            "multinode_compatible": True
        })

        # Convert to serializable format if needed
        if force_serializable and source == DatasetSource.HUGGING_FACE:
            dataset = dataset.prepare_for_multinode_distribution()

        return dataset

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
                result = cls(dataset_dict)
            else:
                result = cls(DatasetDict({split or "train": dataset}))

            # Store metadata for reconstruction
            result._dataset_metadata = {
                "source": DatasetSource.HUGGING_FACE,
                "dataset_name": dataset_name,
                "split": split,
                "kwargs": kwargs
            }

            return result
        except ValueError as e:
            if "split" in str(e):
                full_dataset = load_dataset(dataset_name, **kwargs)
                result = cls(full_dataset)
                result._dataset_metadata = {
                    "source": DatasetSource.HUGGING_FACE,
                    "dataset_name": dataset_name,
                    "split": None,
                    "kwargs": kwargs
                }
                return result
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
            ),
            self._dataset_metadata.copy()
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

    def merge_split(self, other_dataset: "MDataset", split: str) -> None:
        """
        Merge a split from another dataset into this dataset.

        :param other_dataset: The source dataset to copy the split from
        :param split: The name of the split to merge

        :raises KeyError: If the split is not found in the source dataset
        """
        if split not in other_dataset.available_splits:
            raise KeyError(f"Split '{split}' not found in source dataset")

        # Add the split if it doesn't already exist
        if split not in self.available_splits:
            self._splits[split] = other_dataset.get_split(split)

    def merge_splits(
            self, other_dataset: "MDataset", splits: Optional[List[str]] = None
    ) -> None:
        """
        Merge multiple splits from another dataset into this dataset.

        :param other_dataset: The source dataset to copy the splits from
        :param splits: List of split names to merge. If None, all splits from the other dataset will be merged.
        """
        if splits is None:
            # Merge all splits from the other dataset
            splits = other_dataset.available_splits

        for split in splits:
            try:
                self.merge_split(other_dataset, split)
            except KeyError as e:
                print(f"Warning: {e}")

    def __reduce__(self):
        """Custom serialization for Ray/pickle compatibility"""
        # If we have metadata to reconstruct from HuggingFace, use that
        if (self._dataset_metadata.get("source") == DatasetSource.HUGGING_FACE and
                not self._dataset_metadata.get("converted_to_serializable", False)):
            return (
                self.reconstruct_from_metadata,
                (self._dataset_metadata, self._partitions)
            )
        else:
            # Normal serialization for already-safe datasets
            return (
                self.__class__,
                (self._splits, self._dataset_metadata),
                {"_partitions": self._partitions}
            )

    @classmethod
    def reconstruct_from_metadata(cls: type[T], metadata: Dict[str, Any], partitions: Dict[str, Dict[int, List[int]]]) -> T:
        """Reconstruct dataset from metadata on remote nodes"""
        logger = logging.getLogger("murmura.dataset")

        try:
            if metadata["source"] == DatasetSource.HUGGING_FACE:
                logger.info(f"Reconstructing HuggingFace dataset '{metadata['dataset_name']}' on remote node")

                # Load dataset on the remote node
                dataset_name = metadata["dataset_name"]
                split = metadata["split"]
                kwargs = metadata.get("kwargs", {})

                if split is None:
                    full_dataset = load_dataset(dataset_name, **kwargs)
                    dataset = MDataset(full_dataset, metadata)
                else:
                    hf_dataset = load_dataset(dataset_name, split=split, **kwargs)
                    if isinstance(hf_dataset, list) and isinstance(split, list):
                        dataset_dict = DatasetDict()
                        for i, s in enumerate(split):
                            dataset_dict[s] = hf_dataset[i]
                        dataset = MDataset(dataset_dict, metadata)
                    else:
                        dataset = MDataset(DatasetDict({split or "train": hf_dataset}), metadata)

                # Restore partitions
                dataset._partitions = partitions

                # Convert to serializable format to avoid future issues
                dataset = dataset.prepare_for_multinode_distribution()
                dataset._dataset_metadata["converted_to_serializable"] = True

                logger.info("Dataset reconstruction completed on remote node")
                return dataset
            else:
                raise ValueError(f"Cannot reconstruct dataset with source: {metadata['source']}")

        except Exception as e:
            logger.error(f"Failed to reconstruct dataset on remote node: {e}")
            # Fallback: create empty dataset
            empty_splits = DatasetDict()
            return MDataset(empty_splits, metadata)

    def __setstate__(self, state):
        """Custom deserialization state restoration"""
        if isinstance(state, dict):
            self._partitions = state.get("_partitions", {})

    def __repr__(self) -> str:
        base = f"MDataset(splits={self.available_splits}"
        if self._partitions:
            partitioned_splits = len(self._partitions)
            base += f", partitions={partitioned_splits} split{'s' if partitioned_splits > 1 else ''}"
        if self._dataset_metadata.get("multinode_compatible"):
            base += ", multinode_compatible=True"
        return f"{base})"
