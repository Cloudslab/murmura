import tempfile
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from murmura.data_processing.dataset import Source, MDataset


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
    dataset = MDataset.load(Source.CSV, path=temp_csv_file)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_json(temp_json_file):
    dataset = MDataset.load(Source.JSON, path=temp_json_file, orient="records")
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_parquet(temp_parquet_file):
    dataset = MDataset.load(Source.PARQUET, path=temp_parquet_file)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_pandas(sample_dataframe):
    dataset = MDataset.load(Source.PANDAS, data=sample_dataframe)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_dict(sample_dict):
    dataset = MDataset.load(Source.DICT, data=sample_dict)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_load_list(sample_list):
    dataset = MDataset.load(Source.LIST, data=sample_list)
    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 10


def test_custom_split_name():
    df = pd.DataFrame({"a": [1, 2, 3]})
    dataset = MDataset.load(Source.PANDAS, data=df, split="validation")
    assert "validation" in dataset.available_splits
    assert len(dataset.get_split("validation")) == 3


def test_train_test_split():
    df = pd.DataFrame({"a": range(100)})
    dataset = MDataset.load(Source.PANDAS, data=df)

    assert "train" in dataset.available_splits
    assert len(dataset.get_split("train")) == 100

    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)

    assert "train" in split_dataset.available_splits
    assert "test" in split_dataset.available_splits
    assert len(split_dataset.get_split("train")) == 70
    assert len(split_dataset.get_split("test")) == 30


def test_train_test_split_custom_names():
    df = pd.DataFrame({"a": range(100)})
    dataset = MDataset.load(Source.PANDAS, data=df)
    split_dataset = dataset.train_test_split(
        test_size=0.3, seed=42, new_split_names=("validation", "eval")
    )

    assert "validation" in split_dataset.available_splits
    assert "eval" in split_dataset.available_splits
    assert len(split_dataset.get_split("validation")) == 70
    assert len(split_dataset.get_split("eval")) == 30


def test_invalid_split_access():
    dataset = MDataset.load(Source.DICT, data={"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        dataset.get_split("invalid_split")


# Note: The following test requires internet connection and the dataset to exist
@pytest.mark.integration
def test_load_hugging_face():
    dataset = MDataset.load(
        Source.HUGGING_FACE, dataset_name="mnist", split=["train", "test"]
    )

    assert "train" in dataset.available_splits
    assert "test" in dataset.available_splits

    dataset = MDataset.load(Source.HUGGING_FACE, dataset_name="mnist")

    assert "train" in dataset.available_splits
