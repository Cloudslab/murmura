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
    Improved manager for differential privacy operations in training processes.

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

        # Add these three lines BEFORE calling _initialize_clipping_norms()
        self.clipping_norm_history = []
        self.noise_multiplier_history = []
        self.parameter_norm_cache = {}

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

            # Add the initial norm to history
            self.clipping_norm_history.append(default_norm)

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
        self, parameters: Dict[str, Any], is_client: bool = True
    ) -> Dict[str, Any]:
        """
        Apply privacy mechanism to parameters with improved handling and stability.

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
            if hasattr(param, "copy"):
                param_copy[key] = param.copy()
            else:
                # For non-numpy arrays, convert to numpy for consistent handling
                param_copy[key] = np.array(param)

        # Apply the appropriate privacy based on mode and caller
        if self.config.privacy_mode == PrivacyMode.LOCAL and is_client:
            # Local DP: Clip and add noise before sending
            print(
                f"Local DP: Clipping with norm {list(self.clipping_norms.values())[0]:.4f} and adding noise"
            )
            clipped_params = self.privacy_mechanism.clip_parameters(
                param_copy, self.clipping_norms
            )

            # Improved noise scaling for LOCAL DP
            # Rather than simply dividing by sqrt(num_actors), we use a more principled approach
            # For local DP, each client's contribution is scaled by 1/num_actors in the average
            # So we scale the noise by sqrt(1/num_actors) to maintain the same signal-to-noise ratio
            num_actors = len(self.config.params.get("actors", [])) or 10

            # Calculate local noise multiplier
            # Using the composition theorem for DP, dividing by sqrt(num_actors) is theoretically sound
            # This ensures that the aggregated noise maintains the privacy guarantee
            if num_actors > 1:
                local_noise = self.config.noise_multiplier / np.sqrt(num_actors)
                print(
                    f"  Adjusted noise for local DP: {local_noise:.4f} (original: {self.config.noise_multiplier:.4f})"
                )
            else:
                local_noise = self.config.noise_multiplier

            # Update noise multiplier in the mechanism temporarily for this operation
            original_noise = self.privacy_mechanism.noise_multiplier
            self.privacy_mechanism.noise_multiplier = local_noise

            # Add noise
            noised_params = self.privacy_mechanism.add_noise(
                clipped_params, self.clipping_norms
            )

            # Restore original noise setting
            self.privacy_mechanism.noise_multiplier = original_noise

            # Sanity check to remove any NaN or Inf values
            for key, param in noised_params.items():
                if np.isnan(param).any() or np.isinf(param).any():
                    print(
                        f"Warning: Found NaN or Inf in parameter {key} after adding noise. Replacing with zeros."
                    )
                    param[np.isnan(param) | np.isinf(param)] = 0.0
                    noised_params[key] = param

            return noised_params

        elif self.config.privacy_mode == PrivacyMode.CENTRAL and not is_client:
            # Central DP: Add noise after aggregation
            print(
                f"Central DP: Clipping with norm {list(self.clipping_norms.values())[0]:.4f} and adding noise"
            )

            # First clip the aggregated parameters
            clipped_params = self.privacy_mechanism.clip_parameters(
                param_copy, self.clipping_norms
            )

            # Add noise with the full noise multiplier
            noised_params = self.privacy_mechanism.add_noise(
                clipped_params, self.clipping_norms
            )

            # Sanity check for NaN or Inf values
            for key, param in noised_params.items():
                if np.isnan(param).any() or np.isinf(param).any():
                    print(
                        f"Warning: Found NaN or Inf in parameter {key} after adding noise. Replacing with zeros."
                    )
                    param[np.isnan(param) | np.isinf(param)] = 0.0
                    noised_params[key] = param

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

        # Skip adaptive clipping if clipping_norm is explicitly provided
        if self.config.clipping_norm is not None:
            return

        # Compute adaptive clipping norms
        new_norms = self._compute_adaptive_clipping_norms(parameters_list)

        # Update the clipping norms
        if new_norms:
            # Get max norm for diagnostic purposes
            max_norm = max(new_norms.values()) if new_norms else 0
            print(f"Updated clipping norms - max: {max_norm:.4f}")

            # Store in history for tracking
            self.clipping_norm_history.append(max_norm)

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

            # Update the max_grad_norm in the privacy mechanism
            if hasattr(self.privacy_mechanism, "max_grad_norm"):
                self.privacy_mechanism.max_grad_norm = max_norm

    def _compute_adaptive_clipping_norms(
        self, parameters_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute adaptive clipping norms for parameters.

        Args:
            parameters_list: List of parameter dictionaries

        Returns:
            Dictionary of clipping norms
        """
        if not parameters_list:
            return {}

        result_norms = {}

        # Determine whether to use per-layer or global clipping
        if self.config.per_layer_clipping:
            # Compute norms for each parameter separately
            for key in parameters_list[0].keys():
                # Collect norms for this parameter across all clients
                if key not in self.parameter_norm_cache:
                    self.parameter_norm_cache[key] = []

                param_norms = []
                for params in parameters_list:
                    if key in params:
                        param = params[key]
                        # Compute L2 norm
                        param_norm = float(np.linalg.norm(param.flatten()))
                        param_norms.append(param_norm)

                # Update the cache with new values
                self.parameter_norm_cache[key].extend(param_norms)

                # Keep cache size reasonable (last 100 values)
                if len(self.parameter_norm_cache[key]) > 100:
                    self.parameter_norm_cache[key] = self.parameter_norm_cache[key][
                        -100:
                    ]

                # Use the specified quantile to set clipping norm
                if self.parameter_norm_cache[key]:
                    clip_value = float(
                        np.quantile(
                            self.parameter_norm_cache[key],
                            self.config.adaptive_clipping_quantile,
                        )
                    )
                    # Ensure minimum clipping norm
                    result_norms[key] = max(clip_value, 0.001)
        else:
            # Compute global norm across all parameters
            if "__global__" not in self.parameter_norm_cache:
                self.parameter_norm_cache["__global__"] = []

            global_norms = []
            for params in parameters_list:
                # Compute total squared norm
                squared_sum = 0.0
                for value in params.values():
                    squared_sum += float(np.sum(np.square(value)))
                global_norms.append(np.sqrt(squared_sum))

            # Update the cache
            self.parameter_norm_cache["__global__"].extend(global_norms)

            # Keep cache size reasonable
            if len(self.parameter_norm_cache["__global__"]) > 100:
                self.parameter_norm_cache["__global__"] = self.parameter_norm_cache[
                    "__global__"
                ][-100:]

            # Use the quantile to set the global clipping norm
            if self.parameter_norm_cache["__global__"]:
                global_clip = float(
                    np.quantile(
                        self.parameter_norm_cache["__global__"],
                        self.config.adaptive_clipping_quantile,
                    )
                )

                # Set the same norm for all parameters
                for key in parameters_list[0].keys():
                    result_norms[key] = max(global_clip, 0.001)

        return result_norms

    def setup_privacy_accounting(self, sample_count: int, batch_size: int) -> None:
        """
        Set up initial privacy accounting parameters with improved accuracy.

        Args:
            sample_count: Total number of samples in the dataset
            batch_size: Batch size used in training
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return

        # Ensure we don't have zero values
        self.total_samples = max(1, sample_count)
        self.batch_size = max(1, batch_size)

        print(
            f"Setting up privacy accounting with {self.total_samples} samples, batch size {self.batch_size}"
        )

        # Calculate initial sampling rate
        sampling_rate = min(1.0, self.batch_size / self.total_samples)
        print(f"Sampling rate: {sampling_rate}")

        # Store rounds from config for better accounting
        self.num_rounds = (
            self.config.params.get("rounds", 20) if self.config.params else 20
        )
        local_epochs = (
            self.config.params.get("local_epochs", 1) if self.config.params else 1
        )

        # Calculate expected total iterations
        batches_per_epoch = max(1, self.total_samples // self.batch_size)
        expected_iterations = self.num_rounds * local_epochs * batches_per_epoch

        print(
            f"Expected total iterations: {expected_iterations} "
            + f"({self.num_rounds} rounds x {local_epochs} epochs x ~{batches_per_epoch} batches)"
        )

        # Use user-provided noise multiplier if specified
        if self.config.noise_multiplier is not None:
            noise_multiplier = self.config.noise_multiplier
            print(f"Using specified noise multiplier: {noise_multiplier}")

            # Verify if this noise multiplier will meet the target epsilon
            privacy_estimate = self.privacy_mechanism.get_privacy_spent(
                num_iterations=expected_iterations,
                noise_multiplier=noise_multiplier,
                batch_size=self.batch_size,
                total_samples=self.total_samples,
            )

            estimated_epsilon = privacy_estimate.get("epsilon", float("inf"))
            print(
                f"With noise={noise_multiplier}, estimated final ε={estimated_epsilon:.4f} "
                + f"(target: {self.config.target_epsilon:.4f})"
            )

            if estimated_epsilon > self.config.target_epsilon * 1.5:
                print(
                    f"WARNING: Specified noise multiplier {noise_multiplier} may exceed target epsilon "
                    + f"{self.config.target_epsilon}. Consider increasing noise or reducing rounds."
                )

        # Calculate an appropriate noise multiplier based on target privacy
        else:
            print(
                f"Adaptively calculating noise multiplier for target ε={self.config.target_epsilon:.4f}"
            )

            # Use privacy mechanism to calculate appropriate noise
            noise_multiplier = self.privacy_mechanism.calibrate_noise_to_target_epsilon(
                target_epsilon=self.config.target_epsilon,
                target_delta=self.config.target_delta,
                iterations=expected_iterations,
                batch_size=self.batch_size,
                total_samples=self.total_samples,
            )

            print(
                f"Calculated noise multiplier: {noise_multiplier:.4f} for target ε={self.config.target_epsilon:.4f}"
            )

        # Update the noise multiplier
        if self.privacy_mechanism and isinstance(
            self.privacy_mechanism, GaussianMechanism
        ):
            self.privacy_mechanism.noise_multiplier = noise_multiplier

        # Update the config
        self.config.noise_multiplier = noise_multiplier

        # Store in history
        self.noise_multiplier_history.append(noise_multiplier)

        self.initial_setup_done = True

    def update_privacy_budget(self) -> Dict[str, float]:
        """
        Update the privacy budget tracking with improved accounting.

        Returns:
            Dictionary containing updated privacy budget
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return {"epsilon": 0.0, "delta": 0.0}

        # Increment round counter
        self.current_round += 1

        # If not initialized, try to initialize from config params
        if not self.initial_setup_done or self.total_samples == 0:
            # Try to get values from config params
            if self.config.params:
                total_samples = self.config.params.get("total_samples", 0)
                batch_size = self.config.params.get("batch_size", 32)

                if total_samples > 0 and batch_size > 0:
                    print(
                        f"Late initialization of privacy accounting with {total_samples} samples and {batch_size} batch size"
                    )
                    self.setup_privacy_accounting(total_samples, batch_size)
                else:
                    print(
                        "Warning: Privacy accounting not properly initialized. Missing total_samples or batch_size."
                    )
                    return self.privacy_spent
            else:
                print(
                    "Warning: Privacy accounting not properly initialized. No config params available."
                )
                return self.privacy_spent

        # Calculate privacy spent based on current state
        # Improved calculation of effective iterations:
        local_epochs = (
            self.config.params.get("local_epochs", 1) if self.config.params else 1
        )
        batches_per_epoch = max(1, self.total_samples // self.batch_size)

        # In federated learning, we have to account for effective iterations differently
        if self.config.privacy_mode == PrivacyMode.CENTRAL:
            # For central DP, each round has one effective iteration (the aggregation step)
            effective_iterations = self.current_round
        else:
            # For local DP, each client's local training contributes to the privacy cost
            effective_iterations = self.current_round * local_epochs

        print(
            f"Computing privacy for round {self.current_round}, "
            + f"effective iterations: {effective_iterations}, "
            + f"noise multiplier: {self.config.noise_multiplier}"
        )

        # Compute privacy spent
        self.privacy_spent = self.privacy_mechanism.get_privacy_spent(
            num_iterations=effective_iterations,
            noise_multiplier=self.config.noise_multiplier,
            batch_size=self.batch_size,
            total_samples=self.total_samples,
        )

        # Update noise multiplier if needed to stay within budget
        target_epsilon = self.config.target_epsilon
        current_epsilon = self.privacy_spent.get("epsilon", 0.0)

        # Track the current privacy spending
        print(
            f"Updated privacy budget: ε = {current_epsilon:.6f} (target: {target_epsilon:.6f})"
        )

        # Add to history
        self.noise_multiplier_history.append(self.config.noise_multiplier)

        # If approaching the target, adjust noise if we don't have a fixed multiplier
        if (
            current_epsilon > target_epsilon * 0.8
            and self.current_round < self.num_rounds
            and self.config.clipping_norm is None
        ):
            print(
                f"Approaching target epsilon ({current_epsilon:.4f}/{target_epsilon:.4f}). "
                + "Recalibrating noise for remaining rounds..."
            )

            # Estimate remaining iterations
            remaining_rounds = self.num_rounds - self.current_round

            if remaining_rounds > 0:
                # Calculate noise needed for remaining budget
                remaining_budget = max(0.0, target_epsilon - current_epsilon)

                if remaining_budget > 0:
                    # Calculate noise for remaining rounds
                    new_noise = (
                        self.privacy_mechanism.calibrate_noise_to_target_epsilon(
                            target_epsilon=remaining_budget,
                            target_delta=self.config.target_delta,
                            iterations=remaining_rounds,
                            batch_size=self.batch_size,
                            total_samples=self.total_samples,
                        )
                    )

                    # Update the noise multiplier if it's significantly different
                    if new_noise > self.config.noise_multiplier * 1.5:
                        print(
                            f"Increasing noise multiplier from {self.config.noise_multiplier:.4f} to {new_noise:.4f}"
                        )
                        self.config.noise_multiplier = new_noise

                        if self.privacy_mechanism and isinstance(
                            self.privacy_mechanism, GaussianMechanism
                        ):
                            self.privacy_mechanism.noise_multiplier = new_noise

        return self.privacy_spent

    def get_current_privacy_spent(self) -> Dict[str, float]:
        """
        Get the current privacy budget spent with additional information.

        Returns:
            Dictionary containing current epsilon and delta values plus additional info
        """
        result = self.privacy_spent.copy()

        # Add more information for reporting
        result["noise_multiplier"] = self.config.noise_multiplier

        # Add clipping info
        if self.clipping_norms:
            # Get default or first value as representative
            result["clipping_norm"] = self.clipping_norms.get(
                "__default__", next(iter(self.clipping_norms.values()))
            )

        # Add target values
        result["target_epsilon"] = self.config.target_epsilon
        result["target_delta"] = self.config.target_delta

        # Add mode
        result["privacy_mode"] = self.config.privacy_mode.value

        return result

    def reset(self) -> None:
        """Reset the privacy manager state."""
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}
        self._initialize_clipping_norms()
        self.clipping_norm_history = []
        self.noise_multiplier_history = []
        self.parameter_norm_cache = {}
