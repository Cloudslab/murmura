"""Shared factory builders for dataset adapters, models, aggregators, and attacks.

Used by both the CLI (simulation mode) and distributed node processes so that
neither duplicates the config-to-object wiring logic.
"""

import importlib
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn

from murmura.config.schema import Config


def build_dataset_adapter(config: Config) -> Any:
    """Instantiate the dataset adapter specified in config."""
    name = config.data.adapter

    if name.startswith("leaf."):
        dataset_type = name.split(".")[1]
        from murmura.examples.leaf import load_leaf_adapter
        return load_leaf_adapter(
            dataset_type,
            num_nodes=config.topology.num_nodes,
            seed=config.experiment.seed,
            **config.data.params,
        )

    if name.startswith("wearables."):
        dataset_type = name.split(".")[1]
        from murmura.examples.wearables import load_wearable_adapter
        return load_wearable_adapter(
            dataset_type=dataset_type,
            num_nodes=config.topology.num_nodes,
            seed=config.experiment.seed,
            **config.data.params,
        )

    module_path, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**config.data.params)


def build_model_factory(config: Config) -> Callable[[], nn.Module]:
    """Return a zero-argument callable that creates a fresh model instance."""
    path = config.model.factory

    if path.startswith("examples.leaf."):
        from murmura.examples.leaf import get_leaf_model_factory
        return get_leaf_model_factory(path.split(".")[-1], **config.model.params)

    if path.startswith("examples.wearables."):
        from murmura.examples.wearables import get_wearable_model_factory
        return get_wearable_model_factory(path.split(".")[-1], **config.model.params)

    module_path, factory_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    factory_fn = getattr(module, factory_name)
    params = dict(config.model.params)
    return lambda: factory_fn(**params)


def build_aggregator_factory(
    config: Config,
    model_factory: Callable[[], nn.Module],
    device: torch.device,
) -> Callable[[int], Any]:
    """Return a factory(node_id) -> Aggregator callable."""
    from murmura.aggregation import (
        FedAvgAggregator,
        KrumAggregator,
        BALANCEAggregator,
        SketchguardAggregator,
        UBARAggregator,
        EvidentialTrustAggregator,
    )
    from murmura.aggregation.base import calculate_model_dimension

    agg_type = config.aggregation.algorithm.lower()
    params = dict(config.aggregation.params)

    if agg_type == "sketchguard":
        params["model_dim"] = calculate_model_dimension(model_factory())
        params["total_rounds"] = config.experiment.rounds

    if agg_type in ("balance", "ubar", "evidential_trust"):
        params["total_rounds"] = config.experiment.rounds

    _map = {
        "fedavg": FedAvgAggregator,
        "krum": KrumAggregator,
        "balance": BALANCEAggregator,
        "sketchguard": SketchguardAggregator,
        "ubar": UBARAggregator,
        "evidential_trust": EvidentialTrustAggregator,
    }

    if agg_type not in _map:
        raise ValueError(f"Unknown aggregation algorithm: {agg_type}")

    cls = _map[agg_type]
    return lambda node_id: cls(**params)


def build_criterion(config: Config) -> Tuple[Optional[nn.Module], bool]:
    """Return (criterion, is_evidential) for the given config."""
    evidential = config.model.factory.startswith("examples.wearables.")
    if not evidential:
        return None, False

    from murmura.examples.wearables import get_evidential_loss
    num_classes = config.model.params.get("num_classes", 6)
    annealing_rounds = config.experiment.rounds // 2
    criterion = get_evidential_loss(
        num_classes=num_classes,
        annealing_epochs=annealing_rounds,
        lambda_weight=0.1,
    )
    return criterion, True


def build_attack(config: Config) -> Optional[Any]:
    """Instantiate the attack object from config, or None if disabled."""
    if not config.attack.enabled or not config.attack.type:
        return None

    from murmura.attacks.gaussian import GaussianAttack
    from murmura.attacks.directed import DirectedDeviationAttack
    from murmura.attacks.topology_liar import TopologyLiarAttack

    n   = config.topology.num_nodes
    pct = config.attack.percentage
    seed = config.experiment.seed

    if config.attack.type == "gaussian":
        return GaussianAttack(
            num_nodes=n,
            attack_percentage=pct,
            noise_std=config.attack.params.get("noise_std", 10.0),
            seed=seed,
        )
    if config.attack.type == "directed_deviation":
        return DirectedDeviationAttack(
            num_nodes=n,
            attack_percentage=pct,
            lambda_param=config.attack.params.get("lambda_param", -5.0),
            seed=seed,
        )
    if config.attack.type == "topology_liar":
        # Optionally wrap a model-state attack underneath the topology liar.
        inner_type = config.attack.params.get("model_attack_type")
        model_attack = None
        if inner_type == "gaussian":
            model_attack = GaussianAttack(
                num_nodes=n,
                attack_percentage=pct,
                noise_std=config.attack.params.get("noise_std", 10.0),
                seed=seed,
            )
        elif inner_type == "directed_deviation":
            model_attack = DirectedDeviationAttack(
                num_nodes=n,
                attack_percentage=pct,
                lambda_param=config.attack.params.get("lambda_param", -5.0),
                seed=seed,
            )
        return TopologyLiarAttack(
            num_nodes=n,
            attack_percentage=pct,
            seed=seed,
            model_attack=model_attack,
        )
    return None


def build_mobility_model(config: Config):
    """Instantiate MobilityModel from config.mobility, or None if absent."""
    if config.mobility is None:
        return None
    from murmura.topology.dynamic import MobilityModel
    m = config.mobility
    return MobilityModel(
        num_nodes=config.topology.num_nodes,
        area_size=m.area_size,
        comm_range=m.comm_range,
        max_speed=m.max_speed,
        seed=m.seed,
        ensure_connected=m.ensure_connected,
    )
