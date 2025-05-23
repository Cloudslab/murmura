from typing import Dict, Any, List, Optional

import numpy as np

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.network_management.topology import TopologyType
from murmura.privacy.privacy_config import (
    PrivacyConfig,
    PrivacyMode,
    PrivacyMechanismType,
)


class PrivacyManager:
    """
    Fixed manager for differential privacy operations with smart initialization.
    """

    def __init__(self, privacy_config: PrivacyConfig):
        """Initialize the privacy manager."""
        self.config = privacy_config
        self.privacy_mechanism = self._create_mechanism()
        self.clipping_norms: Dict[str, float] = {}

        # History tracking
        self.clipping_norm_history = []
        self.noise_multiplier_history = []
        self.parameter_norm_cache = {}
        self.update_norm_history = []

        # Initialize with temporary defaults - will be updated when we see model
        self._initialize_clipping_norms()

        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}
        self.total_samples = 0
        self.batch_size = 0
        self.num_actors = 1
        self.initial_setup_done = False
        self.model_initialized = False
        self.learning_rate = None

    def _create_mechanism(self) -> Optional[Any]:
        """Create the privacy mechanism based on configuration."""
        if (
                not self.config.enabled
                or self.config.mechanism_type == PrivacyMechanismType.NONE
        ):
            return None

        if self.config.mechanism_type == PrivacyMechanismType.GAUSSIAN:
            from murmura.privacy.gaussian_mechanism import GaussianMechanism

            return GaussianMechanism(
                noise_multiplier=self.config.noise_multiplier or 1.0,
                max_grad_norm=self.config.max_grad_norm,
                per_layer_clipping=self.config.per_layer_clipping,
            )

        raise ValueError(f"Unsupported privacy mechanism: {self.config.mechanism_type}")

    def _initialize_clipping_norms(self) -> None:
        """Initialize clipping norms with temporary defaults."""
        if self.config.clipping_norm is not None:
            # Use provided explicit clipping norm
            self.clipping_norms = {"__default__": self.config.clipping_norm}
        else:
            # Temporary default - will be updated when we see the model
            default_norm = 0.1
            self.clipping_norms = {"__default__": default_norm}
            self.clipping_norm_history.append(default_norm)

    def initialize_from_model(self, model_params: Dict[str, Any],
                              learning_rate: float) -> None:
        """
        Initialize clipping norms based on actual model parameters.
        This should be called after the model is distributed to actors.

        Args:
            model_params: Initial model parameters
            learning_rate: Learning rate being used
        """
        if self.model_initialized or self.config.clipping_norm is not None:
            return  # Already initialized or using fixed clipping

        self.learning_rate = learning_rate

        # Calculate smart initial clipping norm
        initial_clip = self._estimate_initial_clipping_norm(
            model_params, learning_rate
        )

        if self.config.per_layer_clipping:
            # Initialize per-layer clipping based on layer sizes
            new_norms = {}
            for key, param in model_params.items():
                # Estimate layer-specific clipping
                layer_size = param.size
                layer_norm = float(np.linalg.norm(param))

                # Smaller layers might need different clipping
                size_factor = np.sqrt(layer_size / 1000)  # Normalize to 1000 params
                layer_clip = initial_clip * size_factor

                # Ensure reasonable bounds
                layer_clip = np.clip(layer_clip, 0.01, 2.0)
                new_norms[key] = layer_clip

            self.clipping_norms = new_norms
            print(f"Initialized per-layer clipping norms based on model structure")
        else:
            # Global clipping
            self.clipping_norms = {"__default__": initial_clip}
            print(f"Initialized global clipping norm: {initial_clip:.6f}")

        # Update mechanism if needed
        if hasattr(self.privacy_mechanism, "max_grad_norm"):
            self.privacy_mechanism.max_grad_norm = initial_clip

        self.clipping_norm_history.append(initial_clip)
        self.model_initialized = True

    def _estimate_initial_clipping_norm(self, model_params: Dict[str, Any],
                                        learning_rate: float) -> float:
        """
        Estimate a reasonable initial clipping norm based on model size and learning rate.

        Args:
            model_params: Initial model parameters
            learning_rate: Learning rate being used

        Returns:
            Estimated initial clipping norm for updates
        """
        # Calculate model size and average parameter magnitude
        total_params = 0
        total_norm = 0.0

        for param in model_params.values():
            total_params += param.size
            total_norm += float(np.sum(np.square(param)))

        avg_param_magnitude = np.sqrt(total_norm / total_params)

        # Estimate expected update magnitude
        # Updates are typically learning_rate * gradient
        # Gradients are often 1-10x parameter magnitude for neural nets
        gradient_scale = 2.0  # Conservative estimate
        expected_update_magnitude = learning_rate * avg_param_magnitude * gradient_scale

        # For safety, multiply by a factor to avoid over-clipping initially
        safety_factor = 5.0 if self.config.privacy_mode == PrivacyMode.CENTRAL else 3.0

        initial_clip = expected_update_magnitude * safety_factor

        # Ensure reasonable bounds for updates
        initial_clip = np.clip(initial_clip, 0.01, 1.0)

        print(f"Model statistics for clipping norm estimation:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Avg parameter magnitude: {avg_param_magnitude:.6f}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Expected update magnitude: {expected_update_magnitude:.6f}")
        print(f"  Initial clipping norm: {initial_clip:.6f}")

        return initial_clip

    def compute_adaptive_clipping_for_updates(
            self, updates_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute adaptive clipping norms specifically for updates.
        """
        if not updates_list:
            return self.clipping_norms

        # Compute norms of updates
        all_norms = []
        per_layer_norms = {}

        for updates in updates_list:
            # Skip if updates are all zeros
            if all(np.allclose(update, 0) for update in updates.values()):
                continue

            # Compute global norm of this client's update
            global_norm_sq = 0.0
            for key, update in updates.items():
                if self.config.per_layer_clipping:
                    if key not in per_layer_norms:
                        per_layer_norms[key] = []
                    layer_norm = float(np.linalg.norm(update.flatten()))
                    if layer_norm > 0:
                        per_layer_norms[key].append(layer_norm)

                global_norm_sq += float(np.sum(np.square(update)))

            global_norm = np.sqrt(global_norm_sq)
            if global_norm > 0:
                all_norms.append(global_norm)

        # If we have no valid norms, keep current clipping
        if not all_norms and not any(per_layer_norms.values()):
            return self.clipping_norms

        # Use a percentile-based approach
        target_quantile = self.config.adaptive_clipping_quantile

        if self.config.per_layer_clipping:
            new_norms = {}
            for key in updates_list[0].keys():
                if key in per_layer_norms and per_layer_norms[key]:
                    observed_norm = float(np.quantile(per_layer_norms[key], target_quantile))

                    if key in self.clipping_norms:
                        # Smooth update with momentum
                        momentum = 0.9
                        old_value = self.clipping_norms[key]
                        new_value = momentum * old_value + (1 - momentum) * observed_norm
                    else:
                        new_value = observed_norm * 0.5

                    # Bounds based on expected update size
                    if self.learning_rate:
                        # Scale bounds with learning rate
                        max_expected = self.learning_rate * 100  # Generous upper bound
                        min_expected = self.learning_rate * 0.1   # Lower bound
                    else:
                        max_expected = 1.0
                        min_expected = 0.001

                    new_norms[key] = np.clip(new_value, min_expected, max_expected)
                else:
                    if key in self.clipping_norms:
                        new_norms[key] = self.clipping_norms[key]
                    else:
                        new_norms[key] = 0.1

            return new_norms
        else:
            # Global clipping
            if all_norms:
                observed_norm = float(np.quantile(all_norms, target_quantile))

                if "__default__" in self.clipping_norms:
                    momentum = 0.9
                    old_value = self.clipping_norms["__default__"]
                    new_value = momentum * old_value + (1 - momentum) * observed_norm
                else:
                    new_value = observed_norm * 0.5

                # Learning rate aware bounds
                if self.learning_rate:
                    max_expected = self.learning_rate * 100
                    min_expected = self.learning_rate * 0.1
                else:
                    max_expected = 1.0
                    min_expected = 0.001

                new_value = np.clip(new_value, min_expected, max_expected)

                print(f"Adaptive clipping: {new_value:.6f} (from {self.clipping_norms.get('__default__', 0.1):.6f})")

                return {"__default__": new_value}

        return self.clipping_norms

    def privatize_parameters(
            self, parameters: Dict[str, Any], is_client: bool = True
    ) -> Dict[str, Any]:
        """
        Apply privacy mechanism to parameters.
        """
        if not self.config.enabled or self.privacy_mechanism is None:
            return parameters

        # Make a copy to avoid modifying the original
        param_copy = {}
        for key, param in parameters.items():
            if hasattr(param, "copy"):
                param_copy[key] = param.copy()
            else:
                param_copy[key] = np.array(param)

        # Apply privacy based on mode
        if self.config.privacy_mode == PrivacyMode.LOCAL and is_client:
            # Local DP: Each client clips and adds noise
            clipped_params = self.privacy_mechanism.clip_parameters(
                param_copy, self.clipping_norms
            )

            # For local DP, use full noise
            noised_params = self.privacy_mechanism.add_noise(
                clipped_params, self.clipping_norms
            )

            return noised_params

        elif self.config.privacy_mode == PrivacyMode.CENTRAL and not is_client:
            # Central DP: Server clips aggregated updates and adds noise

            # First clip the updates
            clipped_params = self.privacy_mechanism.clip_parameters(
                param_copy, self.clipping_norms
            )

            # Add noise
            noised_params = self.privacy_mechanism.add_noise(
                clipped_params, self.clipping_norms
            )

            return noised_params

        # If conditions don't match, return original
        return parameters

    def update_clipping_norms(self, parameters_list: List[Dict[str, Any]]) -> None:
        """Update clipping norms adaptively based on parameter distribution."""
        if (
                not self.config.enabled
                or self.privacy_mechanism is None
                or not parameters_list
        ):
            return

        # Skip if clipping norm is explicitly set
        if self.config.clipping_norm is not None:
            return

        # For Central DP, these should be updates, not full parameters
        # Compute adaptive clipping based on update magnitudes
        new_norms = self.compute_adaptive_clipping_for_updates(parameters_list)

        if new_norms:
            self.clipping_norms = new_norms

            # Track history
            if "__default__" in new_norms:
                max_norm = new_norms["__default__"]
            else:
                max_norm = max(new_norms.values())

            self.clipping_norm_history.append(max_norm)

            # Update mechanism
            if hasattr(self.privacy_mechanism, "max_grad_norm"):
                self.privacy_mechanism.max_grad_norm = max_norm

            print(f"Updated clipping norm to: {max_norm:.6f}")

    def _compute_adaptive_clipping_norms(
            self, parameters_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute adaptive clipping norms for parameters."""
        if not parameters_list:
            return {}

        result_norms = {}

        if self.config.per_layer_clipping:
            # Per-layer clipping
            for key in parameters_list[0].keys():
                if key not in self.parameter_norm_cache:
                    self.parameter_norm_cache[key] = []

                # Compute norms for this parameter
                param_norms = []
                for params in parameters_list:
                    if key in params:
                        param_norm = float(np.linalg.norm(params[key].flatten()))
                        param_norms.append(param_norm)

                # Update cache
                self.parameter_norm_cache[key].extend(param_norms)

                # Keep cache size reasonable
                if len(self.parameter_norm_cache[key]) > 100:
                    self.parameter_norm_cache[key] = self.parameter_norm_cache[key][-100:]

                # Use quantile for clipping
                if self.parameter_norm_cache[key]:
                    clip_value = float(
                        np.quantile(
                            self.parameter_norm_cache[key],
                            self.config.adaptive_clipping_quantile,
                        )
                    )
                    result_norms[key] = max(clip_value, 0.01)
        else:
            # Global clipping
            if "__global__" not in self.parameter_norm_cache:
                self.parameter_norm_cache["__global__"] = []

            # Compute global norms
            global_norms = []
            for params in parameters_list:
                squared_sum = 0.0
                for value in params.values():
                    squared_sum += float(np.sum(np.square(value)))
                global_norms.append(np.sqrt(squared_sum))

            # Update cache
            self.parameter_norm_cache["__global__"].extend(global_norms)

            # Keep cache size reasonable
            if len(self.parameter_norm_cache["__global__"]) > 100:
                self.parameter_norm_cache["__global__"] = self.parameter_norm_cache["__global__"][-100:]

            # Use quantile for global clipping
            if self.parameter_norm_cache["__global__"]:
                global_clip = float(
                    np.quantile(
                        self.parameter_norm_cache["__global__"],
                        self.config.adaptive_clipping_quantile,
                    )
                )

                # Set same norm for all parameters
                for key in parameters_list[0].keys():
                    result_norms[key] = max(global_clip, 0.01)

        return result_norms

    def setup_privacy_accounting(self, sample_count: int, batch_size: int) -> None:
        """Set up initial privacy accounting parameters."""
        if not self.config.enabled or self.privacy_mechanism is None:
            return

        self.total_samples = max(1, sample_count)
        self.batch_size = max(1, batch_size)

        # Get number of actors if available
        if self.config.params and "num_actors" in self.config.params:
            self.num_actors = self.config.params["num_actors"]
        else:
            self.num_actors = 1

        print(f"Setting up privacy accounting:")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Number of actors: {self.num_actors}")

        # Get rounds and epochs
        self.num_rounds = self.config.params.get("rounds", 10) if self.config.params else 10
        local_epochs = self.config.params.get("local_epochs", 1) if self.config.params else 1

        # Calculate expected iterations based on privacy mode
        if self.config.privacy_mode == PrivacyMode.CENTRAL:
            # For central DP, privacy cost is per round (one aggregation)
            expected_iterations = self.num_rounds
        else:
            # For local DP, privacy cost is per client update
            # But since we're adding noise independently at each client,
            # the composition is different
            expected_iterations = self.num_rounds * local_epochs

        print(f"  Expected iterations: {expected_iterations}")
        print(f"  Privacy mode: {self.config.privacy_mode.value}")

        # Calibrate noise if needed
        if self.config.noise_multiplier is None and self.config.adaptive_noise:
            print(f"Calibrating noise for target ε={self.config.target_epsilon}")

            # For Local DP, we need to account for the fact that privacy
            # amplification happens due to data being distributed
            target_epsilon = self.config.target_epsilon

            noise_multiplier = self.privacy_mechanism.calibrate_noise_to_target_epsilon(
                target_epsilon=target_epsilon,
                target_delta=self.config.target_delta,
                iterations=expected_iterations,
                batch_size=self.batch_size,
                total_samples=self.total_samples,
            )

            self.config.noise_multiplier = noise_multiplier

            if self.privacy_mechanism:
                self.privacy_mechanism.noise_multiplier = noise_multiplier

        elif self.config.noise_multiplier is not None:
            print(f"Using specified noise multiplier: {self.config.noise_multiplier}")

            # Check if it meets target
            privacy_estimate = self.privacy_mechanism.get_privacy_spent(
                num_iterations=expected_iterations,
                noise_multiplier=self.config.noise_multiplier,
                batch_size=self.batch_size,
                total_samples=self.total_samples,
            )

            estimated_epsilon = privacy_estimate.get("epsilon", float("inf"))
            print(f"  Estimated final ε: {estimated_epsilon:.4f} (target: {self.config.target_epsilon})")

        self.noise_multiplier_history.append(self.config.noise_multiplier)
        self.initial_setup_done = True

    def update_privacy_budget(self) -> Dict[str, float]:
        """Update the privacy budget tracking."""
        if not self.config.enabled or self.privacy_mechanism is None:
            return {"epsilon": 0.0, "delta": 0.0}

        self.current_round += 1

        # Late initialization if needed
        if not self.initial_setup_done:
            if self.config.params:
                total_samples = self.config.params.get("total_samples", 0)
                batch_size = self.config.params.get("batch_size", 32)

                if total_samples > 0:
                    self.setup_privacy_accounting(total_samples, batch_size)

        # Calculate privacy spent
        if self.config.privacy_mode == PrivacyMode.CENTRAL:
            # Central DP: one privacy cost per round
            effective_iterations = self.current_round
        else:
            # Local DP: privacy cost accumulates with local training
            local_epochs = self.config.params.get("local_epochs", 1) if self.config.params else 1
            effective_iterations = self.current_round * local_epochs

        # Compute privacy spent
        self.privacy_spent = self.privacy_mechanism.get_privacy_spent(
            num_iterations=effective_iterations,
            noise_multiplier=self.config.noise_multiplier,
            batch_size=self.batch_size,
            total_samples=self.total_samples,
        )

        current_epsilon = self.privacy_spent.get("epsilon", 0.0)
        print(f"Round {self.current_round}: ε = {current_epsilon:.4f} (target: {self.config.target_epsilon})")

        # Check if we should stop early
        if self.config.early_stopping and current_epsilon > self.config.target_epsilon:
            print(f"WARNING: Privacy budget exceeded! ε = {current_epsilon:.4f} > {self.config.target_epsilon}")

        return self.privacy_spent

    def get_current_privacy_spent(self) -> Dict[str, float]:
        """Get the current privacy budget spent."""
        result = self.privacy_spent.copy()

        # Add additional information
        result["noise_multiplier"] = self.config.noise_multiplier

        if self.clipping_norms:
            result["clipping_norm"] = self.clipping_norms.get(
                "__default__", next(iter(self.clipping_norms.values()))
            )

        result["target_epsilon"] = self.config.target_epsilon
        result["target_delta"] = self.config.target_delta
        result["privacy_mode"] = self.config.privacy_mode.value

        return result

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

    def reset(self) -> None:
        """Reset the privacy manager state."""
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}
        self._initialize_clipping_norms()
        self.clipping_norm_history = []
        self.noise_multiplier_history = []
        self.parameter_norm_cache = {}
