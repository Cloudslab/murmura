from typing import List, Dict, Any, Optional

import numpy as np

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategy_interface import AggregationStrategy


class FedAvg(AggregationStrategy):
    """
    Implements the Federated Averaging (FedAvg) algorithm for model aggregation.

    FedAvg computes the weighted average of model parameters from multiple clients.
    """

    coordination_mode = CoordinationMode.CENTRALIZED

    def __init__(self, central_privacy_config=None):
        """
        Initialize the FedAvg strategy.
        """
        super().__init__()
        self.central_privacy_mechanism = None
        self.central_privacy_epsilon = None
        self.central_privacy_delta = None
        if central_privacy_config:
            mechanism = central_privacy_config.get("mechanism", "gaussian")
            epsilon = central_privacy_config.get("epsilon", 1.0)
            delta = central_privacy_config.get("delta", 1e-5)
            if mechanism == "laplace":
                from murmura.privacy.central.laplace import LaplaceMechanismCDP
                self.central_privacy_mechanism = LaplaceMechanismCDP()
            elif mechanism == "gaussian":
                from murmura.privacy.central.gaussian import GaussianMechanismCDP
                self.central_privacy_mechanism = GaussianMechanismCDP()
            self.central_privacy_epsilon = epsilon
            self.central_privacy_delta = delta

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters using FedAvg.

        :param parameters_list: List of model parameters from different clients.
        :param weights: Optional list of weights for each client's parameters.

        :return: Aggregated model parameters.
        """
        if not parameters_list:
            raise ValueError("Empty parameters list")

        if weights is None:
            weights = [1.0 / len(parameters_list)] * len(parameters_list)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        aggregated_params = {}

        for key in parameters_list[0].keys():
            try:
                stacked_params = np.stack(
                    [params[key] for params in parameters_list], axis=0
                )
                weighted_params = np.zeros_like(stacked_params[0])

                for i, weight in enumerate(weights):
                    weighted_params += weight * stacked_params[i]

                aggregated_params[key] = weighted_params

            except Exception as e:
                raise ValueError(f"Error aggregating parameter {key}: {e}")

        # Add central DP noise if enabled
        if self.central_privacy_mechanism:
            for k, v in aggregated_params.items():
                aggregated_params[k] = self.central_privacy_mechanism.add_noise(
                    v, self.central_privacy_epsilon, self.central_privacy_delta
                )

        return aggregated_params
