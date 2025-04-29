import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from murmura.orchestration.cluster_manager import ClusterManager
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.aggregation.aggregation_config import AggregationConfig, AggregationStrategyType
from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.strategies.gossip_avg import GossipAvg
from murmura.network_management.topology_compatibility import TopologyCompatibilityManager


class ActorMock:
    """Mock actor implementation that doesn't need Ray to function"""
    def __init__(self, client_id):
        self.client_id = client_id
        self.has_model = False
        self.has_dataset = False
        self.params = None
        self.dataset = None
        self.feature_columns = None
        self.label_column = None
        self.neighbours = []
        self.partition = None
        self.metadata = {}

    def get_data_info(self):
        """Mock get_data_info method"""
        return {
            "client_id": self.client_id,
            "has_model": self.has_model,
            "has_dataset": self.has_dataset
        }

    def set_model(self, model):
        """Mock set_model method"""
        self.has_model = True
        self.model = model
        return True

    def set_dataset(self, dataset, feature_columns=None, label_column=None):
        """Mock set_dataset method"""
        self.has_dataset = True
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.label_column = label_column
        return True

    def set_model_parameters(self, params):
        """Mock set_model_parameters method"""
        self.params = params
        return True

    def get_model_parameters(self):
        """Mock get_model_parameters method"""
        return {"layer1": np.array([1.0, 2.0])}

    def train_model(self, **kwargs):
        """Mock train_model method"""
        return {"loss": 0.5, "accuracy": 0.7}

    def evaluate_model(self, **kwargs):
        """Mock evaluate_model method"""
        return {"loss": 0.3, "accuracy": 0.9}

    def receive_data(self, partition, metadata=None):
        """Mock receive_data method"""
        self.partition = partition
        self.metadata = metadata or {}
        return f"Received {len(partition)} samples"

    def set_neighbours(self, neighbours):
        """Mock set_neighbours method"""
        self.neighbours = neighbours
        return True


@pytest.fixture
def mock_ray():
    """Setup comprehensive Ray mocking"""
    with patch('ray.init') as mock_init, \
            patch('ray.is_initialized', return_value=True) as mock_is_init, \
            patch('ray.remote') as mock_remote, \
            patch('ray.get') as mock_get, \
            patch('ray.kill') as mock_kill:

        # Configure mock_remote to return a factory that creates ActorMock instances
        def mock_remote_factory(*args, **kwargs):
            return lambda client_id: ActorMock(client_id)

        mock_remote.return_value = mock_remote_factory

        # Configure mock_get to return the input directly (since we're using actual objects)
        mock_get.side_effect = lambda x: x

        # Return all mocks for flexible usage in tests
        yield {
            "init": mock_init,
            "is_initialized": mock_is_init,
            "remote": mock_remote,
            "get": mock_get,
            "kill": mock_kill
        }


@pytest.fixture
def cluster_manager(mock_ray):
    """Create a properly mocked cluster manager"""
    return ClusterManager(config={"ray_address": None})


def test_create_actors(cluster_manager):
    """Test actor creation with proper IDs and count"""
    num_actors = 3
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    actors = cluster_manager.create_actors(num_actors, topology)

    assert len(actors) == num_actors
    for i, actor in enumerate(actors):
        assert actor.client_id == f"client_{i}"


def test_distribute_data_equal_partitions(cluster_manager):
    """Test distribution with equal actors and partitions"""
    num_actors = 3
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(num_actors, topology)
    partitions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    metadata = {"split": "train"}

    acks = cluster_manager.distribute_data(partitions, metadata)

    # Expect an acknowledgment per actor
    assert len(acks) == num_actors
    for i, actor in enumerate(cluster_manager.actors):
        # Each partition has 3 items
        assert len(actor.partition) == 3
        # Metadata should include the provided metadata with partition_idx overridden per actor
        assert actor.metadata["split"] == "train"
        assert actor.metadata["partition_idx"] == i


def test_distribute_data_more_actors_than_partitions(cluster_manager):
    """Test round-robin distribution when there are more actors than partitions"""
    # Create 5 actors, but provide only 2 partitions. Distribution will wrap around.
    num_actors = 5
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(num_actors, topology)
    partitions = [[1, 2], [3, 4]]

    cluster_manager.distribute_data(partitions)

    # Expected: actor 0 gets partition 0, actor 1 gets partition 1, actor 2 gets partition 0, etc.
    expected_partition_indices = [0, 1, 0, 1, 0]
    expected_partition_data = [
        [1, 2], [3, 4], [1, 2], [3, 4], [1, 2]
    ]

    for i, actor in enumerate(cluster_manager.actors):
        assert actor.metadata["partition_idx"] == expected_partition_indices[i]
        assert actor.partition == expected_partition_data[i]


def test_update_aggregation_strategy_compatibility_check(cluster_manager):
    """Test updating aggregation strategy with compatibility check"""
    # Create topology and initial strategy
    topology_config = TopologyConfig(topology_type=TopologyType.COMPLETE)
    actors = cluster_manager.create_actors(3, topology_config)

    # Set initial strategy
    initial_strategy = FedAvg()
    cluster_manager.aggregation_strategy = initial_strategy

    # Update to a compatible strategy (GossipAvg is compatible with COMPLETE)
    new_strategy = GossipAvg()
    cluster_manager.update_aggregation_strategy(new_strategy)

    # Verify it was updated
    assert cluster_manager.aggregation_strategy == new_strategy

    # Try updating to strategy that's incompatible with mocked behavior
    with patch.object(
            TopologyCompatibilityManager, 'is_compatible', return_value=False
    ), patch.object(
        TopologyCompatibilityManager, 'get_compatible_topologies', return_value=["star"]
    ):
        with pytest.raises(ValueError, match="is not compatible with the current topology"):
            cluster_manager.update_aggregation_strategy(initial_strategy)

    # Now try with topology check disabled
    cluster_manager.update_aggregation_strategy(initial_strategy, topology_check=False)
    assert cluster_manager.aggregation_strategy == initial_strategy


def test_distribute_model_with_parameters(cluster_manager):
    """Test distributing a model with its parameters to actors"""
    # Create actors
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(3, topology)

    # Create a mock model
    model = MagicMock()
    params = {"layer1": np.array([1.0, 2.0]), "layer2": np.array([3.0, 4.0])}
    model.get_parameters.return_value = params

    # Distribute the model
    cluster_manager.distribute_model(model)

    # Verify all actors received the model and parameters
    for actor in cluster_manager.actors:
        assert actor.has_model
        assert actor.model == model


def test_update_models_with_parameters(cluster_manager):
    """Test updating model parameters on actors"""
    # Create actors
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(3, topology)

    # Set up a model on each actor
    for actor in cluster_manager.actors:
        actor.has_model = True

    # Define new parameters
    new_params = {"layer1": np.array([5.0, 6.0])}

    # Update all models
    cluster_manager.update_models(new_params)

    # Verify parameters were updated on all actors
    for actor in cluster_manager.actors:
        assert actor.params == new_params


def test_distribute_dataset_with_options(cluster_manager):
    """Test distributing a dataset with feature columns and label column"""
    # Create actors
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(3, topology)

    # Create a mock dataset
    dataset = MagicMock()
    feature_columns = ["feature1", "feature2"]
    label_column = "label"

    # Distribute dataset
    cluster_manager.distribute_dataset(dataset, feature_columns, label_column)

    # Verify all actors received the dataset and options
    for actor in cluster_manager.actors:
        assert actor.has_dataset
        assert actor.dataset == dataset
        assert actor.feature_columns == feature_columns
        assert actor.label_column == label_column


def test_train_models_with_parameters(cluster_manager):
    """Test training models with specific parameters"""
    # Create actors
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(3, topology)

    # Set a model on each actor
    for actor in cluster_manager.actors:
        actor.has_model = True

    # Train models with specific parameters
    train_kwargs = {"epochs": 5, "batch_size": 32, "verbose": True}
    metrics = cluster_manager.train_models(**train_kwargs)

    # Verify we got metrics from each actor
    assert len(metrics) == 3
    for metric in metrics:
        assert "loss" in metric
        assert "accuracy" in metric


def test_evaluate_models_with_parameters(cluster_manager):
    """Test evaluating models with specific parameters"""
    # Create actors
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(3, topology)

    # Set a model on each actor
    for actor in cluster_manager.actors:
        actor.has_model = True

    # Evaluate models with specific parameters
    eval_kwargs = {"batch_size": 64}
    metrics = cluster_manager.evaluate_models(**eval_kwargs)

    # Verify we got metrics from each actor
    assert len(metrics) == 3
    for metric in metrics:
        assert "loss" in metric
        assert "accuracy" in metric


def test_get_topology_information(cluster_manager):
    """Test getting topology information after initialization"""
    # Create actors with star topology
    topology_config = TopologyConfig(topology_type=TopologyType.STAR, hub_index=1)
    cluster_manager.create_actors(3, topology_config)

    # Get topology information
    info = cluster_manager.get_topology_information()

    # Verify the information is correct
    assert info["initialized"] is True
    assert info["type"] == "star"
    assert info["num_actors"] == 3
    assert info["hub_index"] == 1
    assert "adjacency_list" in info


def test_get_compatible_strategies(cluster_manager):
    """Test getting compatible strategies for a topology"""
    # Create topology
    topology_config = TopologyConfig(topology_type=TopologyType.STAR)
    cluster_manager.create_actors(3, topology_config)

    # Get compatible strategies
    strategies = cluster_manager.get_compatible_strategies()

    # Verify we got a list of strategies
    assert isinstance(strategies, list)
    assert len(strategies) > 0

    # For star topology, fedavg should be included
    assert "fedavg" in strategies


def test_get_compatible_strategies_without_topology(cluster_manager):
    """Test getting compatible strategies without a topology"""
    # No topology set yet
    strategies = cluster_manager.get_compatible_strategies()

    # Should return empty list
    assert strategies == []


def test_initialize_coordinator(cluster_manager):
    """Test initialization of the topology coordinator"""
    # Create topology
    topology_config = TopologyConfig(topology_type=TopologyType.STAR)
    cluster_manager.create_actors(3, topology_config)

    # Set aggregation strategy
    aggregation_config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
    cluster_manager.set_aggregation_strategy(aggregation_config)

    # Verify coordinator was initialized
    assert cluster_manager.topology_coordinator is not None


def test_initialize_coordinator_with_existing_topology(cluster_manager):
    """Test initializing coordinator when topology already exists"""
    # Create topology
    topology_config = TopologyConfig(topology_type=TopologyType.STAR)
    cluster_manager.create_actors(3, topology_config)

    # Set first strategy
    aggregation_config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
    cluster_manager.set_aggregation_strategy(aggregation_config)

    # Get initial coordinator
    initial_coordinator = cluster_manager.topology_coordinator
    assert initial_coordinator is not None

    # Set new strategy
    new_config = AggregationConfig(strategy_type=AggregationStrategyType.TRIMMED_MEAN)
    cluster_manager.set_aggregation_strategy(new_config)

    # Verify coordinator was reinitialized (should be a different object)
    assert cluster_manager.topology_coordinator is not None
    assert cluster_manager.topology_coordinator is not initial_coordinator


def test_aggregation_strategy_with_topology_check(cluster_manager):
    """Test setting aggregation strategy with topology compatibility check"""
    # Create a topology
    topology_config = TopologyConfig(topology_type=TopologyType.RING)
    cluster_manager.create_actors(3, topology_config)

    # Try setting a strategy with a compatible topology
    with patch.object(TopologyCompatibilityManager, 'is_compatible', return_value=True):
        aggregation_config = AggregationConfig(strategy_type=AggregationStrategyType.GOSSIP_AVG)
        cluster_manager.set_aggregation_strategy(aggregation_config)

    # Verify strategy was set
    assert cluster_manager.aggregation_strategy is not None

    # Try setting a strategy with an incompatible topology
    with patch.object(
            TopologyCompatibilityManager, 'is_compatible', return_value=False
    ), patch.object(
        TopologyCompatibilityManager, 'get_compatible_topologies', return_value=[]
    ), patch.object(
        TopologyCompatibilityManager, 'get_compatible_strategies', return_value=[]
    ):
        incompatible_config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
        with pytest.raises(ValueError, match="not compatible with topology"):
            cluster_manager.set_aggregation_strategy(incompatible_config)


def test_aggregate_model_parameters(cluster_manager):
    """Test aggregating model parameters using the coordinator"""
    # Create topology and actors
    topology_config = TopologyConfig(topology_type=TopologyType.STAR)
    cluster_manager.create_actors(3, topology_config)

    # Set strategy
    aggregation_config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
    cluster_manager.set_aggregation_strategy(aggregation_config)

    # Mock the coordinator's coordinate_aggregation method
    mock_result = {"layer1": np.array([1.5, 2.5])}
    cluster_manager.topology_coordinator.coordinate_aggregation = MagicMock(return_value=mock_result)

    # Aggregate parameters with weights
    weights = [0.5, 0.3, 0.2]
    result = cluster_manager.aggregate_model_parameters(weights=weights)

    # Verify coordinator's method was called with weights
    cluster_manager.topology_coordinator.coordinate_aggregation.assert_called_once_with(weights=weights)

    # Verify the result
    assert result == mock_result


def test_distribute_data_with_custom_metadata(cluster_manager):
    """Test distributing data partitions with custom metadata"""
    # Create actors
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(3, topology)

    # Create data partitions and metadata
    partitions = [[1, 2], [3, 4], [5, 6]]
    metadata = {
        "dataset_name": "mnist",
        "split": "train",
        "custom_field": "test_value"
    }

    # Distribute data
    cluster_manager.distribute_data(partitions, metadata)

    # Verify metadata was correctly merged with partition index
    for i, actor in enumerate(cluster_manager.actors):
        assert actor.metadata["dataset_name"] == "mnist"
        assert actor.metadata["split"] == "train"
        assert actor.metadata["custom_field"] == "test_value"
        assert actor.metadata["partition_idx"] == i


def test_distribute_data_unequal_partitions(cluster_manager):
    """Test distributing data when there are fewer partitions than actors"""
    # Create more actors than partitions
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(5, topology)

    # Create fewer partitions than actors
    partitions = [[1, 2], [3, 4]]

    # Distribute data
    cluster_manager.distribute_data(partitions)

    # Verify wrapping around of partitions (round-robin distribution)
    expected_partition_indices = [0, 1, 0, 1, 0]
    expected_partition_data = [
        [1, 2], [3, 4], [1, 2], [3, 4], [1, 2]
    ]

    for i, actor in enumerate(cluster_manager.actors):
        assert actor.metadata["partition_idx"] == expected_partition_indices[i]
        assert actor.partition == expected_partition_data[i]


def test_cluster_manager_initialization_with_ray_address(mock_ray):
    """Test initialization with Ray address"""
    # Create cluster manager with custom Ray address
    config = {"ray_address": "ray://10.0.0.1:10001"}
    ClusterManager(config)

    # Verify ray.init was called with the address
    mock_ray["init"].assert_called_once_with(address="ray://10.0.0.1:10001")


def test_cluster_manager_initialization_without_ray_address(mock_ray):
    """Test initialization without Ray address"""
    # Create cluster manager without Ray address
    ClusterManager(config={})

    # Verify ray.init was called with None address
    mock_ray["init"].assert_called_with(address=None)


def test_apply_topology(cluster_manager):
    """Test applying topology to actors"""
    # Create a topology with specific adjacency
    adjacency = {
        0: [1, 2],
        1: [0],
        2: [0]
    }
    topology_config = TopologyConfig(
        topology_type=TopologyType.CUSTOM,
        adjacency_list=adjacency
    )

    # Create actors
    actors = cluster_manager.create_actors(3, topology_config)

    # Verify neighbors are set based on adjacency
    # Node 0 should have nodes 1 and 2 as neighbors
    assert len(actors[0].neighbours) == 2
    assert actors[1] in actors[0].neighbours
    assert actors[2] in actors[0].neighbours

    # Node 1 should have only node 0 as neighbor
    assert len(actors[1].neighbours) == 1
    assert actors[0] in actors[1].neighbours

    # Node 2 should have only node 0 as neighbor
    assert len(actors[2].neighbours) == 1
    assert actors[0] in actors[2].neighbours


def test_shutdown(mock_ray):
    """Test cluster shutdown calls ray.shutdown"""
    ClusterManager.shutdown()
    mock_ray["is_initialized"].assert_called()
