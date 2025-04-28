import time

from murmura.visualization.training_event import (
    TrainingEvent,
    LocalTrainingEvent,
    ParameterTransferEvent,
    AggregationEvent,
    ModelUpdateEvent,
    EvaluationEvent,
    InitialStateEvent,
)


def test_base_training_event():
    """Test the base TrainingEvent class"""
    # Record start time
    before = time.time()

    # Create event
    event = TrainingEvent(round_num=1, step_name="test_step")

    # Record end time
    after = time.time()

    # Check properties
    assert event.round_num == 1
    assert event.step_name == "test_step"
    assert before <= event.timestamp <= after


def test_local_training_event():
    """Test the LocalTrainingEvent class"""
    active_nodes = [0, 1, 2]
    metrics = {"loss": 0.5, "accuracy": 0.9}

    event = LocalTrainingEvent(round_num=2, active_nodes=active_nodes, metrics=metrics)

    assert event.round_num == 2
    assert event.step_name == "local_training"
    assert event.active_nodes == active_nodes
    assert event.metrics == metrics


def test_parameter_transfer_event():
    """Test the ParameterTransferEvent class"""
    source_nodes = [0]
    target_nodes = [1, 2, 3]
    param_summary = {0: {"norm": 1.0, "mean": 0.5, "std": 0.1}}

    event = ParameterTransferEvent(
        round_num=3,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        param_summary=param_summary
    )

    assert event.round_num == 3
    assert event.step_name == "parameter_transfer"
    assert event.source_nodes == source_nodes
    assert event.target_nodes == target_nodes
    assert event.param_summary == param_summary


def test_aggregation_event():
    """Test the AggregationEvent class"""
    participating_nodes = [0, 1, 2, 3]
    aggregator_node = 0
    strategy_name = "FedAvg"

    event = AggregationEvent(
        round_num=4,
        participating_nodes=participating_nodes,
        aggregator_node=aggregator_node,
        strategy_name=strategy_name
    )

    assert event.round_num == 4
    assert event.step_name == "aggregation"
    assert event.participating_nodes == participating_nodes
    assert event.aggregator_node == aggregator_node
    assert event.strategy_name == strategy_name


def test_model_update_event():
    """Test the ModelUpdateEvent class"""
    updated_nodes = [0, 1, 2, 3]
    param_convergence = 0.01

    event = ModelUpdateEvent(
        round_num=5,
        updated_nodes=updated_nodes,
        param_convergence=param_convergence
    )

    assert event.round_num == 5
    assert event.step_name == "model_update"
    assert event.updated_nodes == updated_nodes
    assert event.param_convergence == param_convergence


def test_evaluation_event():
    """Test the EvaluationEvent class"""
    metrics = {"loss": 0.3, "accuracy": 0.95}

    event = EvaluationEvent(round_num=6, metrics=metrics)

    assert event.round_num == 6
    assert event.step_name == "evaluation"
    assert event.metrics == metrics


def test_initial_state_event():
    """Test the InitialStateEvent class"""
    topology_type = "star"
    num_nodes = 5

    event = InitialStateEvent(topology_type=topology_type, num_nodes=num_nodes)

    assert event.round_num == 0
    assert event.step_name == "initial_state"
    assert event.topology_type == topology_type
    assert event.num_nodes == num_nodes
