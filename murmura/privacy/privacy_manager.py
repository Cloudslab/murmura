from typing import Dict, Any, List, Optional

import numpy as np

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.network_management.topology import TopologyType
from murmura.privacy.gaussian_mechanism import GaussianMechanism
from murmura.privacy.privacy_config import (
    PrivacyConfig,
    PrivacyMode,
    PrivacyMechanismType,
)
from murmura.privacy.rdp_accountant import RDPAccountant


class PrivacyManager:
    """
    Manages differential privacy operations for training processes.

    This class coordinates the application of privacy mechanisms based on the
    specified configuration and ensures proper accounting of the privacy budget.
    """

    def __init__(self, privacy_config: PrivacyConfig):
        """
        Initialize the privacy manager.

        Args:
            privacy_config: Privacy configuration
        """
        self.config = privacy_config
        self.accountant = RDPAccountant()
        self.privacy_mechanism = self._create_mechanism()
        self.clipping_norms: Dict[str, float] = {}
        self._initialize_clipping_norms()
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}
        self.total_samples = 0
        self.batch_size = 0
        self.initial_setup_done = False

    def _create_mechanism(self) -> Optional[Any]:
        """
        Create the privacy mechanism based on configuration.

        Returns:
            Initialized privacy mechanism
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

        raise ValueError(f"Unsupported privacy mechanism: {self.config.mechanism_type}")

    def _initialize_clipping_norms(self) -> None:
        """Initialize clipping norms."""
        if self.config.clipping_norm is not None:
            # Use provided explicit clipping norm
            self.clipping_norms = {"__default__": self.config.clipping_norm}
        else:
            # Start with a default clipping norm
            default_norm = self.config.max_grad_norm
            self.clipping_norms = {"__default__": default_norm}

            # We'll update these adaptively during training

    def is_compatible_with_topology(self, topology_type: TopologyType) -> bool:
        """
        Check if the privacy configuration is compatible with the topology.

        Args:
            topology_type: Topology type to check compatibility with

        Returns:
            True if compatible, False otherwise
        """
        if not self.config.enabled:
            return True

        if self.config.privacy_mode == PrivacyMode.CENTRAL:
            # Central DP is only compatible with star/centralized topologies
            return topology_type in [TopologyType.STAR, TopologyType.COMPLETE]

        if self.config.privacy_mode == PrivacyMode.LOCAL:
            # Local DP is compatible with all topologies
            return True

        # Default: not compatible
        return False

    def is_compatible_with_coordination_mode(
        self, coordination_mode: CoordinationMode
    ) -> bool:
        """
        Check if the privacy configuration is compatible with the coordination mode.

        Args:
            coordination_mode: Coordination mode to check compatibility with

        Returns:
            True if compatible, False otherwise
        """
        if not self.config.enabled:
            return True

        if self.config.privacy_mode == PrivacyMode.CENTRAL:
            # Central DP requires centralized coordination
            return coordination_mode == CoordinationMode.CENTRALIZED

        if self.config.privacy_mode == PrivacyMode.LOCAL:
            # Local DP is compatible with all coordination modes
            return True

        # Default: not compatible
        return False

    def privatize_parameters(
            self,
            parameters: Dict[str, Any],
            is_client: bool = True
    ) -> Dict[str, Any]:
        """
        Apply privacy mechanism to parameters based on configuration.

        Args:
            parameters: Parameters to privatize
            is_client: Whether the caller is a client (vs server)

        Returns:
            Privatized parameters
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return parameters

        # Make a copy of parameters to avoid modifying the original
        param_copy = {}
        for key, param in parameters.items():
            # Handle different parameter types safely
            if hasattr(param, 'copy'):
                param_copy[key] = param.copy()
            else:
                # For non-numpy arrays, convert to numpy for consistent handling
                param_copy[key] = np.array(param)

        # Apply the appropriate privacy based on mode and caller
        if self.config.privacy_mode == PrivacyMode.LOCAL and is_client:
            # Local DP: Clip and add noise before sending
            print(f"Local DP: Clipping with norm {list(self.clipping_norms.values())[0]:.4f} and adding noise")
            clipped_params = self.privacy_mechanism.clip_parameters(
                param_copy, self.clipping_norms
            )

            # Reduce noise for LOCAL DP as it accumulates across nodes
            # For local DP, each client adds noise independently, so we can reduce it
            effective_noise = self.config.noise_multiplier
            if self.config.noise_multiplier > 0.5:
                # Scale down noise by num_actors since it accumulates
                num_actors = len(self.config.params.get("actors", [])) or 10
                effective_noise = self.config.noise_multiplier / np.sqrt(max(1, num_actors))
                print(f"  Adjusted noise for local DP: {effective_noise:.4f} (original: {self.config.noise_multiplier:.4f})")

            noised_params = {}
            for key, param in clipped_params.items():
                # Get appropriate clipping norm for this parameter
                clip_norm = self.clipping_norms.get(key, self.clipping_norms.get("__default__", 1.0))

                # Calculate noise scale based on L2 sensitivity and noise multiplier
                noise_scale = clip_norm * effective_noise / np.sqrt(2)

                # Generate and add noise
                noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)
                noised_params[key] = param + noise

            return noised_params

        elif self.config.privacy_mode == PrivacyMode.CENTRAL and not is_client:
            # Central DP: Add noise after aggregation
            print(f"Central DP: Clipping with norm {list(self.clipping_norms.values())[0]:.4f} and adding noise")
            clipped_params = self.privacy_mechanism.clip_parameters(
                param_copy, self.clipping_norms
            )

            # For central DP, we apply the full noise scale
            noised_params = {}
            for key, param in clipped_params.items():
                # Get appropriate clipping norm for this parameter
                clip_norm = self.clipping_norms.get(key, self.clipping_norms.get("__default__", 1.0))

                # Calculate noise scale based on L2 sensitivity and noise multiplier
                noise_scale = clip_norm * self.config.noise_multiplier / np.sqrt(2)

                # Generate and add noise
                noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)
                noised_params[key] = param + noise

            return noised_params

        # If conditions don't match, return original parameters
        return parameters

    def update_clipping_norms(self, parameters_list: List[Dict[str, Any]]) -> None:
        """
        Update clipping norms adaptively based on the distribution of parameter values.

        Args:
            parameters_list: List of parameter dictionaries from multiple clients
        """
        if (
            not self.config.enabled
            or self.privacy_mechanism is None
            or not parameters_list
        ):
            return

        # Only compute adaptive clipping if enabled (when clipping_norm is None)
        if self.config.clipping_norm is not None:
            return

        # Compute adaptive clipping norms
        new_norms = self.privacy_mechanism.compute_adaptive_clipping_norms(
            parameters_list=parameters_list,
            quantile=self.config.adaptive_clipping_quantile,
            per_layer=self.config.per_layer_clipping,
        )

        # Update the clipping norms
        if new_norms:
            # Get max norm for diagnostic purposes
            max_norm = max(new_norms.values()) if new_norms else 0
            print(f"Updated clipping norms - max: {max_norm:.4f}")

            # When using per-layer clipping, update each layer's norm
            if self.config.per_layer_clipping:
                self.clipping_norms.update(new_norms)
            else:
                # Set a single norm for global clipping
                global_norm = (
                    next(iter(new_norms.values()))
                    if new_norms
                    else self.config.max_grad_norm
                )
                self.clipping_norms = {"__default__": global_norm}

    def setup_privacy_accounting(
            self,
            sample_count: int,
            batch_size: int
    ) -> None:
        """
        Set up initial privacy accounting parameters.

        Args:
            sample_count: Total number of samples in the dataset
            batch_size: Batch size used in training
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return

        # Ensure we don't have zero values
        self.total_samples = max(1, sample_count)
        self.batch_size = max(1, batch_size)

        print(f"Setting up privacy accounting with {self.total_samples} samples, batch size {self.batch_size}")

        # Calculate initial sampling rate
        sampling_rate = self.batch_size / self.total_samples
        print(f"Sampling rate: {sampling_rate}")

        # Use user-provided noise multiplier if specified
        if self.config.noise_multiplier is not None:
            noise_multiplier = self.config.noise_multiplier
            print(f"Using specified noise multiplier: {noise_multiplier}")
        # Otherwise calculate a reasonable starting value
        else:
            # Calculate a safe starting noise multiplier based on target privacy
            # This is an approximation - will be refined during training
            noise_multiplier = 1.0  # Default starting point

            # Higher target epsilon = less noise needed
            if self.config.target_epsilon > 5.0:
                noise_multiplier = 0.8
            elif self.config.target_epsilon > 1.0:
                noise_multiplier = 1.2
            else:
                noise_multiplier = 1.5  # More noise for stricter privacy

            print(f"Calculated initial noise multiplier: {noise_multiplier}")

        # Update the noise multiplier
        if self.privacy_mechanism and isinstance(self.privacy_mechanism, GaussianMechanism):
            self.privacy_mechanism.noise_multiplier = noise_multiplier

        # Update the config
        self.config.noise_multiplier = noise_multiplier

        self.initial_setup_done = True

    def update_privacy_budget(self) -> Dict[str, float]:
        """
        Update the privacy budget tracking.

        Returns:
            Dictionary containing updated privacy budget
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return {"epsilon": 0.0, "delta": 0.0}

        # Increment round counter
        self.current_round += 1

        # Ensure initial setup is done
        if not self.initial_setup_done or self.total_samples == 0:
            print("Warning: Privacy accounting not properly initialized.")
            return self.privacy_spent

        # Calculate privacy spent based on current state
        # Calculate effective iterations for a federated setting
        # In federated learning, each round involves multiple local updates
        local_epochs = max(1, self.config.params.get("local_epochs", 1))
        batches_per_epoch = max(1, self.total_samples // self.batch_size)
        effective_iterations = self.current_round * local_epochs  # Simplified model

        print(f"Computing privacy for round {self.current_round}, " +
              f"effective iterations: {effective_iterations}, " +
              f"noise multiplier: {self.config.noise_multiplier}")

        # Compute privacy spent
        self.privacy_spent = self.privacy_mechanism.get_privacy_spent(
            num_iterations=effective_iterations,
            noise_multiplier=self.config.noise_multiplier,
            batch_size=self.batch_size,
            total_samples=self.total_samples
        )

        print(f"Updated privacy budget: ε = {self.privacy_spent.get('epsilon', 0.0):.6f}")

        return self.privacy_spent

    def get_current_privacy_spent(self) -> Dict[str, float]:
        """
        Get the current privacy budget spent.

        Returns:
            Dictionary containing current epsilon and delta values
        """
        return self.privacy_spent

    def reset(self) -> None:
        """Reset the privacy manager state."""
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}
        self._initialize_clipping_norms()
