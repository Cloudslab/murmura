import tempfile
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from murmura.data_processing.dataset import DatasetSource, MDataset
from murmura.data_processing.partitioner import DirichletPartitioner, IIDPartitioner


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"id": range(10), "value": [f"val_{i}" for i in range(10)]})


@pytest.fixture
def sample_dict():
    return {"id": list(range(10)), "value": [f"val_{i}" for i in range(10)]}


@pytest.fixture
def sample_list():
    return [{"id": i, "value": f"val_{i}"} for i in range(10)]


@pytest.fixture
def temp_csv_file(sample_dataframe):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def temp_json_file(sample_dict):
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        # Convert dictionary to DataFrame first, then to JSON
        df = pd.DataFrame(sample_dict)
        df.to_json(f.name, orient="records")
        yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def temp_parquet_file(sample_dataframe):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        sample_dataframe.to_parquet(f.name)
        yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def sample_mdataset():
    data = {"text": ["sample"] * 100, "label": [i % 5 for i in range(100)]}
    return MDataset(DatasetDict({"train": Dataset.from_dict(data)}))


def test_init():
    splits = DatasetDict(
        {
            "train": Dataset.from_dict({"a": [1, 2, 3]}),
            "test": Dataset.from_dict({"a": [4, 5, 6]}),
        }
    )
    dataset = MDataset(splits)
    assert dataset.available_splits == ["train", "test"]
    assert isinstance(dataset.get_split("train"), Dataset)


def test_load_csv(temp_csv_file):
    dataset = MDataset.load(DatasetSource.CSV, path=temp_csv_file)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_json(temp_json_file):
    dataset = MDataset.load(DatasetSource.JSON, path=temp_json_file, orient="records")
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_parquet(temp_parquet_file):
    dataset = MDataset.load(DatasetSource.PARQUET, path=temp_parquet_file)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_pandas(sample_dataframe):
    dataset = MDataset.load(DatasetSource.PANDAS, data=sample_dataframe)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_dict(sample_dict):
    dataset = MDataset.load(DatasetSource.DICT, data=sample_dict)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_list(sample_list):
    dataset = MDataset.load(DatasetSource.LIST, data=sample_list)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10
    assert repr(dataset) == "MDataset(splits=['train'])"


def test_custom_split_name():
    df = pd.DataFrame({"a": [1, 2, 3]})
    dataset = MDataset.load(DatasetSource.PANDAS, data=df, split="validation")
    assert "validation" in dataset.available_splits
    assert len(dataset.get_split("validation")) == 3


def test_train_test_split():
    df = pd.DataFrame({"a": range(100)})
    dataset = MDataset.load(DatasetSource.PANDAS, data=df)

    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 100

    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)

    assert "train" in split_dataset.available_splits
    assert "test" in split_dataset.available_splits
    assert len(split_dataset.get_split("train")) == 70
    assert len(split_dataset.get_split("test")) == 30


def test_train_test_split_custom_names():
    df = pd.DataFrame({"a": range(100)})
    dataset = MDataset.load(DatasetSource.PANDAS, data=df)
    split_dataset = dataset.train_test_split(
        test_size=0.3, seed=42, new_split_names=("validation", "eval")
    )

    assert "validation" in split_dataset.available_splits
    assert "eval" in split_dataset.available_splits
    assert len(split_dataset.get_split("validation")) == 70
    assert len(split_dataset.get_split("eval")) == 30


def test_invalid_split_access():
    dataset = MDataset.load(DatasetSource.DICT, data={"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        dataset.get_split("invalid_split")


def test_partition_storage_initialization():
    """Test that partitions are initialized as empty dict"""
    splits = DatasetDict({"train": Dataset.from_dict({"a": [1, 2, 3]})})
    dataset = MDataset(splits)
    assert dataset.partitions == {}
    assert dataset.list_partitioned_splits() == []


def test_add_and_retrieve_partitions():
    """Test basic partition storage and retrieval"""
    splits = DatasetDict({"train": Dataset.from_dict({"a": range(10)})})
    dataset = MDataset(splits)

    sample_partitions = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
    dataset.add_partitions("train", sample_partitions)

    assert "train" in dataset.partitions
    assert dataset.list_partitioned_splits() == ["train"]
    assert dataset.get_partitions("train") == sample_partitions


def test_clear_partitions():
    """Test partition clearing functionality"""
    splits = DatasetDict(
        {
            "train": Dataset.from_dict({"a": range(10)}),
            "test": Dataset.from_dict({"a": range(5)}),
        }
    )
    dataset = MDataset(splits)

    # Add partitions to multiple splits
    dataset.add_partitions("train", {0: [1, 2, 3]})
    dataset.add_partitions("test", {0: [1, 2]})

    # Clear specific split
    dataset.clear_partitions("train")
    assert "train" not in dataset.partitions
    assert "test" in dataset.partitions

    # Clear all partitions
    dataset.clear_partitions()
    assert dataset.partitions == {}


def test_partition_integration_with_partitioners(sample_mdataset):
    """Test end-to-end workflow with actual partitioners"""
    # Test Dirichlet Partitioner
    dirichlet_part = DirichletPartitioner(
        num_partitions=3, alpha=0.5, partition_by="label", min_partition_size=10
    )
    dirichlet_part.partition(sample_mdataset, "train")

    dirichlet_partitions = sample_mdataset.get_partitions("train")
    assert len(dirichlet_partitions) == 3
    assert sum(len(p) for p in dirichlet_partitions.values()) == 100

    # Test IID Partitioner
    iid_part = IIDPartitioner(num_partitions=4)
    sample_mdataset.clear_partitions()
    iid_part.partition(sample_mdataset, "train")

    iid_partitions = sample_mdataset.get_partitions("train")
    assert len(iid_partitions) == 4
    assert (
        max(
            len(p) - min(len(p) for p in iid_partitions.values())
            for p in iid_partitions.values()
        )
        <= 1
    )


def test_multiple_split_partitioning():
    """Test partitioning different splits independently"""
    splits = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"a": range(100), "label": [i % 5 for i in range(100)]}
            ),
            "test": Dataset.from_dict(
                {"a": range(20), "label": [i % 5 for i in range(20)]}
            ),
        }
    )
    dataset = MDataset(splits)

    # Partition different splits with different strategies
    dirichlet_part = DirichletPartitioner(3, 0.5, "label")
    iid_part = IIDPartitioner(2)

    dirichlet_part.partition(dataset, "train")
    iid_part.partition(dataset, "test")

    # Verify separate partition storage
    assert len(dataset.get_partitions("train")) == 3
    assert len(dataset.get_partitions("test")) == 2
    assert dataset.list_partitioned_splits() == ["train", "test"]


def test_get_nonexistent_partitions():
    """Test behavior when requesting partitions for unpartitioned split"""
    dataset = MDataset(DatasetDict({"train": Dataset.from_dict({"a": [1, 2, 3]})}))

    # Should return empty dict for unpartitioned split
    assert dataset.get_partitions("train") == {}

    # Should raise KeyError for unknown split
    with pytest.raises(KeyError):
        dataset.get_partitions("invalid_split")


def test_partition_metadata_persistence():
    """Test that partitions survive dataset operations"""
    splits = DatasetDict({"train": Dataset.from_dict({"a": range(10)})})
    dataset = MDataset(splits)
    dataset.add_partitions("train", {0: [1, 2, 3], 1: [4, 5, 6]})

    # Access underlying dataset
    hf_dataset = dataset.get_split("train")
    assert len(hf_dataset) == 10

    # Verify partitions remain after dataset operations
    assert dataset.get_partitions("train") == {0: [1, 2, 3], 1: [4, 5, 6]}


def test_partition_repr():
    """Test string representation includes partition info"""
    splits = DatasetDict({"train": Dataset.from_dict({"a": range(10)})})
    dataset = MDataset(splits)

    # Test initial state
    assert repr(dataset) == "MDataset(splits=['train'])"

    # Test with partitions
    dataset.add_partitions("train", {0: [1, 2, 3]})
    assert "partitions=1 split" in repr(dataset)

    # Test with multiple partitioned splits
    dataset.add_partitions("test", {0: [4, 5, 6]})
    assert "partitions=2 splits" in repr(dataset)


# Note: The following test requires internet connection and the dataset to exist
@pytest.mark.integration
def test_load_hugging_face():
    dataset = MDataset.load(
        DatasetSource.HUGGING_FACE, dataset_name="mnist", split=["train", "test"]
    )

    assert "train" in dataset.available_splits
    assert "test" in dataset.available_splits

    dataset = MDataset.load(DatasetSource.HUGGING_FACE, dataset_name="mnist")

    assert "train" in dataset.available_splits
    assert "test" in dataset.available_splits

    dataset = MDataset.load(
        DatasetSource.HUGGING_FACE, dataset_name="mnist", split="test"
    )

    assert "test" in dataset.available_splits
    assert "train" not in dataset.available_splits
