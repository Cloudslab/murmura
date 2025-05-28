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
    def __init__(
        self, splits: DatasetDict, dataset_metadata: Optional[Dict[str, Any]] = None
    ):
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
        if (
            self.is_serializable_for_multinode()
            and self._dataset_metadata.get("source") != DatasetSource.HUGGING_FACE
        ):
            return self  # Already safe to distribute

        self._logger.info("Converting dataset to multi-node compatible format...")

        # Convert each split to in-memory format
        serializable_splits = DatasetDict()

        for split_name, split_dataset in self._splits.items():
            self._logger.debug(
                f"Converting split '{split_name}' to serializable format..."
            )

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
    def load_dataset_with_multinode_support(cls, source, dataset_name, split):
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
        cls,
        source: DatasetSource,
        split: Optional[Union[str, List[str]]] = None,
        force_serializable: bool = True,
        **kwargs: Any,
    ):
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
        dataset._dataset_metadata.update(
            {
                "source": source,
                "split": split,
                "kwargs": kwargs,
                "multinode_compatible": True,
            }
        )

        # Convert to serializable format if needed
        if force_serializable and source == DatasetSource.HUGGING_FACE:
            dataset = dataset.prepare_for_multinode_distribution()

        return dataset

    @classmethod
    def load(
        cls,
        source: DatasetSource,
        split: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ):
        """
        Load data from specified source with consistent interface

        Args:
            source: Data source type (from Source enum)
            split: Split(s) to load (format depends on source)
            **kwargs: Source-specific parameters
        """
        loader: Dict[DatasetSource, Callable[..., Any]] = {
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
        cls, path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ):
        df = pd.read_csv(path, **kwargs)
        return cls._create_from_data(df, split)

    @classmethod
    def _load_json(
        cls, path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ):
        df = pd.read_json(path, **kwargs)
        return cls._create_from_data(df, split)

    @classmethod
    def _load_parquet(
        cls, path: Union[str, Path], split: Optional[str] = None, **kwargs: Any
    ):
        df = pd.read_parquet(path, **kwargs)
        return cls._create_from_data(df, split)

    @classmethod
    def _load_pandas(cls, data: pd.DataFrame, split: Optional[str] = None, **_: Any):
        return cls._create_from_data(data, split)

    @classmethod
    def _load_dict(cls, data: Dict, split: Optional[str] = None, **_: Any):
        return cls._create_from_data(data, split)

    @classmethod
    def _load_list(cls, data: List, split: Optional[str] = None, **_: Any):
        return cls._create_from_data(data, split)

    @classmethod
    def _load_hf(
        cls,
        dataset_name: str,
        split: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ):
        try:
            if split is None:
                raise ValueError("split is not specified")

            # Remove dataset_name from kwargs if it exists to avoid duplicate parameter error
            load_kwargs = {k: v for k, v in kwargs.items() if k != "dataset_name"}

            dataset = load_dataset(dataset_name, split=split, **load_kwargs)
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
                "kwargs": load_kwargs,  # Use filtered kwargs
            }

            return result
        except ValueError as e:
            if "split" in str(e):
                # Remove dataset_name from kwargs if it exists
                load_kwargs = {k: v for k, v in kwargs.items() if k != "dataset_name"}
                full_dataset = load_dataset(dataset_name, **load_kwargs)
                result = cls(full_dataset)
                result._dataset_metadata = {
                    "source": DatasetSource.HUGGING_FACE,
                    "dataset_name": dataset_name,
                    "split": None,
                    "kwargs": load_kwargs,  # Use filtered kwargs
                }
                return result
            raise

    @classmethod
    def _create_from_data(
        cls, data: Union[pd.DataFrame, Dict, List], split: Optional[str] = None
    ):
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
            self._dataset_metadata.copy(),
        )

    def get_partitions(self, split_name: str) -> Dict[int, List[int]]:
        if split_name not in self._splits:
            raise KeyError(f"Split {split_name} does not exist")
        return self._partitions.get(split_name, {})

    def add_partitions(self, split_name: str, partitions: Dict[int, List[int]]) -> None:
        """Store partitions for a specific split. Enforce that all partition values are lists of indices."""
        for k, v in partitions.items():
            if not isinstance(v, list):
                raise TypeError(f"Partition for key {k} must be a list of indices, got {type(v)}")
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
        """
        Enhanced custom serialization for Ray/pickle compatibility with better error handling.
        """
        # For HuggingFace datasets, prefer metadata reconstruction to avoid serializing large files
        if self._dataset_metadata.get(
            "source"
        ) == DatasetSource.HUGGING_FACE and not self._dataset_metadata.get(
            "converted_to_serializable", False
        ):
            # Ensure metadata is complete before attempting reconstruction
            required_metadata = ["source", "dataset_name"]
            if all(key in self._dataset_metadata for key in required_metadata):
                return (
                    self.reconstruct_from_metadata,
                    (self._dataset_metadata, self._partitions),
                )

        # Fallback to normal serialization for other cases
        return (
            self.__class__,
            (self._splits, self._dataset_metadata),
            {"_partitions": self._partitions},
        )

    @classmethod
    def reconstruct_from_metadata(
        cls, metadata: Dict[str, Any], partitions: Dict[str, Dict[int, List[int]]]
    ):
        """
        Reconstruct dataset from metadata on remote nodes with enhanced error handling.
        """
        logger = logging.getLogger("murmura.dataset")

        try:
            # Validate metadata structure first
            required_keys = ["source", "dataset_name"]
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                raise ValueError(f"Metadata missing required keys: {missing_keys}")

            dataset_source = metadata["source"]

            if dataset_source == DatasetSource.HUGGING_FACE:
                logger.info(
                    f"Reconstructing HuggingFace dataset '{metadata['dataset_name']}' on remote node"
                )

                # Extract parameters with defaults
                dataset_name = metadata["dataset_name"]
                split = metadata.get("split")
                kwargs = metadata.get("kwargs", {})

                # Clean kwargs to avoid duplicate parameters
                clean_kwargs = {k: v for k, v in kwargs.items() if k != "dataset_name"}

                logger.debug(
                    f"Dataset parameters: name={dataset_name}, split={split}, kwargs={clean_kwargs}"
                )

                try:
                    if split is None:
                        # Load all available splits
                        logger.debug("Loading all available splits")
                        full_dataset = load_dataset(dataset_name, **clean_kwargs)
                        dataset = cls(full_dataset, metadata)
                    else:
                        # Load specific split(s)
                        logger.debug(f"Loading specific split(s): {split}")
                        hf_dataset = load_dataset(
                            dataset_name, split=split, **clean_kwargs
                        )

                        if isinstance(hf_dataset, list) and isinstance(split, list):
                            # Multiple splits loaded as list
                            dataset_dict = DatasetDict()
                            for i, s in enumerate(split):
                                dataset_dict[s] = hf_dataset[i]
                            dataset = cls(dataset_dict, metadata)
                        else:
                            # Single split or single dataset
                            split_name = split if isinstance(split, str) else "train"
                            dataset = cls(
                                DatasetDict({split_name: hf_dataset}), metadata
                            )

                    # Restore partitions if provided
                    if partitions:
                        dataset._partitions = partitions.copy()
                        logger.debug(
                            f"Restored partitions for {len(partitions)} splits"
                        )

                    logger.info(
                        "Dataset reconstruction completed successfully on remote node"
                    )
                    return dataset

                except Exception as e:
                    logger.error(f"HuggingFace dataset loading failed: {e}")
                    raise RuntimeError(
                        f"Failed to load HuggingFace dataset '{dataset_name}': {e}"
                    )

            else:
                raise ValueError(
                    f"Unsupported dataset source for reconstruction: {dataset_source}"
                )

        except Exception as e:
            logger.error(f"Dataset reconstruction failed on remote node: {e}")
            logger.error(f"Metadata: {metadata}")

            # Create a minimal fallback dataset to prevent complete failure
            try:
                logger.warning("Creating fallback empty dataset")
                empty_data = {"dummy_feature": [0], "dummy_label": [0]}
                empty_dataset = Dataset.from_dict(empty_data)
                empty_splits = DatasetDict({"train": empty_dataset})
                fallback_dataset = cls(empty_splits, metadata)
                fallback_dataset._partitions = partitions.copy() if partitions else {}
                return fallback_dataset

            except Exception as fallback_error:
                logger.error(f"Even fallback dataset creation failed: {fallback_error}")
                raise RuntimeError(f"Complete dataset reconstruction failure: {e}")

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
