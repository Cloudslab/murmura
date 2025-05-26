#!/usr/bin/env python3
"""
Test script to validate the dataset distribution fixes
"""

import logging

import torch
import torch.nn as nn
from datasets import DatasetDict, Dataset

# Set up basic logging
logging.basicConfig(level=logging.INFO)


def test_mdataset_reconstruction():
    """Test MDataset reconstruction functionality"""
    print("Testing MDataset reconstruction...")

    # Import the fixed MDataset
    from murmura.data_processing.dataset import MDataset, DatasetSource

    # Create test metadata
    metadata = {
        "source": DatasetSource.HUGGING_FACE,
        "dataset_name": "mnist",
        "split": "train",
        "kwargs": {},
    }

    partitions = {"train": {0: [0, 1, 2], 1: [3, 4, 5]}}

    # Test reconstruction
    try:
        reconstructed = MDataset.reconstruct_from_metadata(metadata, partitions)
        print("✓ MDataset reconstruction test passed")
        return True
    except Exception as e:
        print(f"✗ MDataset reconstruction test failed: {e}")
        return False


def test_simple_dataset_creation():
    """Test simple dataset creation and serialization"""
    print("Testing simple dataset creation...")

    from murmura.data_processing.dataset import MDataset

    try:
        # Create a simple test dataset
        test_data = {"image": [[1, 2, 3]] * 10, "label": [0, 1] * 5}

        dataset = Dataset.from_dict(test_data)
        dataset_dict = DatasetDict({"train": dataset})

        # Create MDataset
        mdataset = MDataset(dataset_dict)

        # Test serialization compatibility
        is_serializable = mdataset.is_serializable_for_multinode()
        print(f"✓ Simple dataset creation test passed. Serializable: {is_serializable}")
        return True
    except Exception as e:
        print(f"✗ Simple dataset creation test failed: {e}")
        return False


def test_model_wrapper():
    """Test model wrapper functionality"""
    print("Testing model wrapper...")

    from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper

    class SimpleModel(PyTorchModel):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)

        def forward(self, x):
            return self.linear(x)

    try:
        model = SimpleModel()
        wrapper = TorchModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            input_shape=(10,),
            device="cpu",
        )

        # Test parameter getting/setting
        params = wrapper.get_parameters()
        wrapper.set_parameters(params)

        print("✓ Model wrapper test passed")
        return True
    except Exception as e:
        print(f"✗ Model wrapper test failed: {e}")
        return False


def test_ray_setup():
    """Test Ray cluster setup"""
    print("Testing Ray setup...")

    import ray

    try:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(local_mode=True)

        # Test basic Ray functionality
        @ray.remote
        def test_function():
            return "Ray is working"

        result = ray.get(test_function.remote())

        if result == "Ray is working":
            print("✓ Ray setup test passed")
            return True
        else:
            print("✗ Ray setup test failed: unexpected result")
            return False

    except Exception as e:
        print(f"✗ Ray setup test failed: {e}")
        return False
    finally:
        # Clean up
        if ray.is_initialized():
            ray.shutdown()


def run_mini_integration_test():
    """Run a minimal integration test"""
    print("Running mini integration test...")

    import ray
    from murmura.data_processing.dataset import MDataset
    from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
    from murmura.aggregation.aggregation_config import (
        AggregationConfig,
        AggregationStrategyType,
    )
    from murmura.network_management.topology import TopologyConfig, TopologyType
    from murmura.node.resource_config import RayClusterConfig, ResourceConfig
    from murmura.orchestration.orchestration_config import OrchestrationConfig
    from datasets import Dataset, DatasetDict

    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(local_mode=True, num_cpus=2)

        # Create simple test data
        test_data = {
            "image": [[1.0] * 10] * 100,  # 100 samples, 10 features each
            "label": [0, 1] * 50,  # Binary classification
        }

        dataset = Dataset.from_dict(test_data)
        dataset_dict = DatasetDict({"train": dataset})
        mdataset = MDataset(dataset_dict)

        # Create simple model
        class TestModel(PyTorchModel):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x):
                return self.linear(x)

        model = TestModel()
        model_wrapper = TorchModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            input_shape=(10,),
            device="cpu",
        )

        # Create minimal config
        config = OrchestrationConfig(
            num_actors=2,
            topology=TopologyConfig(topology_type=TopologyType.STAR),
            aggregation=AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG),
            dataset_name="test",
            ray_cluster=RayClusterConfig(address=None),
            resources=ResourceConfig(cpus_per_actor=0.5),
        )

        print("✓ Mini integration test components created successfully")
        return True

    except Exception as e:
        print(f"✗ Mini integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up
        if ray.is_initialized():
            ray.shutdown()
