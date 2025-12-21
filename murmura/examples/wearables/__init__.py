"""Wearable dataset adapters for Murmura.

Provides dataset loaders and evidential deep learning models for:
- UCI HAR: Human Activity Recognition from smartphones
- PAMAP2: Physical Activity Monitoring with body-worn sensors
- PPG-DaLiA: PPG-based activity recognition from wrist-worn sensors

All models use Dirichlet-based evidential outputs for uncertainty
quantification, enabling trust-aware aggregation in decentralized FL.
"""

from murmura.examples.wearables.adapter import (
    load_wearable_adapter,
    get_wearable_dataset_info,
)
from murmura.examples.wearables.models import (
    create_har_model,
    create_pamap2_model,
    create_ppg_dalia_model,
    get_wearable_model_factory,
    get_evidential_loss,
    EvidentialLoss,
    EvidentialHead,
    compute_uncertainty,
)

__all__ = [
    # Adapter
    "load_wearable_adapter",
    "get_wearable_dataset_info",
    # Model factories
    "create_har_model",
    "create_pamap2_model",
    "create_ppg_dalia_model",
    "get_wearable_model_factory",
    # Evidential components
    "get_evidential_loss",
    "EvidentialLoss",
    "EvidentialHead",
    "compute_uncertainty",
]
