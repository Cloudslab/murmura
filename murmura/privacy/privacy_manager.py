from typing import Dict, Optional, Any, List

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.network_management.topology import TopologyType
from murmura.privacy.gaussian_mechanism import GaussianMechanism
from murmura.privacy.privacy_config import (
    PrivacyConfig,
    PrivacyMechanismType,
    PrivacyMode,
)
from murmura.privacy.rdp_accountant import RDPAccountant


class PrivacyManager:
    """
    Manages differential privacy operations for training processes.

    This class coordinates the application of privacy mechanism based on the specified configuration
    and ensures proper accounting of the privacy budget.
    """

    def __init__(self, privacy_config: PrivacyConfig):
        """
        Initialize the PrivacyManager with a given privacy configuration.

        Args:
            privacy_config (PrivacyConfig): Configuration object containing privacy settings.
        """
        self.config = privacy_config
        self.accountant = RDPAccountant()
        self.privacy_mechanism = self._create_mechanism()
        self.clipping_norms: Dict[str, float] = {}
        self._initialize_norms()
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}

    def _create_mechanism(self) -> Optional[Any]:
        """
        Create the privacy mechanism based on configuration.

        Returns:
            Optional[Any]: An instance of the privacy mechanism class.
        """
        if (
            not self.config.enabled
            or self.config.mechanism_type == PrivacyMechanismType.NONE
        ):
            return None

        if self.config.mechanism_type == PrivacyMechanismType.GAUSSIAN:
            return GaussianMechanism(
                noise_multiplier=self.config.noise_multiplier or 1.0,
                max_grad_norm=self.config.max_grad_norm,
                per_layer_clipping=self.config.per_layer_clipping,
                accountant=self.accountant,
            )

        raise ValueError(
            f"Unsupported privacy mechanism type: {self.config.mechanism_type}"
        )

    def _initialize_norms(self) -> None:
        """
        Initialize clipping norms for the model parameters.

        This method sets the clipping norms based on the configuration and the model's parameters.
        """
        if self.config.clipping_norm is not None:
            self.clipping_norms = {"__default__": self.config.clipping_norm}
        else:
            # Initialize with default norms if not provided
            self.clipping_norms = {"__default__": self.config.max_grad_norm}

    def is_compatible_with_topology(self, topology_type: TopologyType) -> bool:
        """
        Check if the privacy mechanism is compatible with the given topology type.

        Args:
            topology_type (TopologyType): The topology type to check compatibility with.

        Returns:
            bool: True if compatible, False otherwise.
        """
        if not self.config.enabled:
            return True

        if self.config.privacy_mode == PrivacyMode.CENTRAL:
            return topology_type in [TopologyType.STAR, TopologyType.COMPLETE]

        if self.config.privacy_mode == PrivacyMode.LOCAL:
            return True

        return False

    def privatize_parameters(
        self, parameters: Dict[str, Any], is_client: bool = True
    ) -> Dict[str, Any]:
        """
        Apply the privacy mechanism to the model parameters.

        Args:
            parameters (Dict[str, Any]): Model parameters to be privatized.
            is_client (bool): Flag indicating if the operation is on client-side.

        Returns:
            Dict[str, Any]: Privatized model parameters.
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return parameters

        if self.config.privacy_mode == PrivacyMode.LOCAL and is_client:
            clipped_params = self.privacy_mechanism.clip_parameters(
                parameters, self.clipping_norms
            )
            noised_params = self.privacy_mechanism.add_noise(
                clipped_params, self.clipping_norms
            )
            return noised_params
        elif self.config.privacy_mode == PrivacyMode.CENTRAL and not is_client:
            noised_params = self.privacy_mechanism.add_noise(
                parameters, self.clipping_norms
            )
            return noised_params

        return parameters

    def update_clipping_norms(self, parameters_list: List[Dict[str, Any]]) -> None:
        """
        Update clipping norms adaptively based on the distribution of parameter values.

        Args:
            parameters_list (List[Dict[str, Any]]): List of parameter dictionaries from multiple clients
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return

        new_norms = self.privacy_mechanism.compute_adaptive_clipping_norms(
            parameters_list=parameters_list,
            quantile=self.config.adaptive_clipping_quantile,
            per_layer=self.config.per_layer_clipping,
        )

        if new_norms:
            self.clipping_norms = new_norms

    def update_privacy_budget(
        self, sample_count: int, batch_size: int
    ) -> Dict[str, float]:
        """
        Update the privacy budget based on the current round and sample count.

        Args:
            sample_count (int): Number of samples processed in the current round.
            batch_size (int): Size of each batch.

        Returns:
            Dict[str, float]: Updated privacy budget (epsilon and delta).
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return {"epsilon": 0.0, "delta": 0.0}

        self.current_round += 1

        self.privacy_spent = self.privacy_mechanism.get_privacy_spent(
            num_iterations=self.current_round,
            noise_multiplier=self.config.noise_multiplier or 1.0,
            batch_size=batch_size,
            total_samples=sample_count,
        )

        if self.config.target_epsilon > 0:
            noise_multiplier = self.accountant.compute_noise_multiplier(
                target_epsilon=self.config.target_epsilon,
                target_delta=self.config.target_delta,
                sampling_rate=batch_size / sample_count,
                iterations=self.current_round + 1,
            )

            if self.privacy_mechanism and isinstance(
                self.privacy_mechanism, GaussianMechanism
            ):
                self.privacy_mechanism.noise_multiplier = noise_multiplier

            self.config.noise_multiplier = noise_multiplier

        return self.privacy_spent

    def get_current_privacy_spent(self) -> Dict[str, float]:
        """
        Get the current privacy budget spent.

        Returns:
            Dict[str, float]: Current privacy budget (epsilon and delta).
        """
        return self.privacy_spent

    def reset(self) -> None:
        """
        Reset the privacy manager to its initial state.
        """
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}
        self.clipping_norms = {}
        self._initialize_norms()

    def is_compatible_with_coordination_mode(
        self, coordination_mode: CoordinationMode
    ) -> bool:
        """
        Check if the privacy mechanism is compatible with the given coordination mode.

        Args:
            coordination_mode (CoordinationMode): The coordination mode to check compatibility with.

        Returns:
            bool: True if compatible, False otherwise.
        """
        if not self.config.enabled:
            return True

        if self.config.privacy_mode == PrivacyMode.CENTRAL:
            return coordination_mode == CoordinationMode.CENTRALIZED

        if self.config.privacy_mode == PrivacyMode.LOCAL:
            return True

        return False
