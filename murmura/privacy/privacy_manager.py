from typing import Dict, Any, List, Optional

import numpy as np

from murmura.privacy.privacy_config import (
    PrivacyConfig,
    PrivacyMode,
    PrivacyMechanismType,
)


class PrivacyManager:
    """
    Fixed manager for differential privacy operations that properly handles updates.
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
        self.update_norm_history = []  # Track update norms over time

        self._initialize_clipping_norms()
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}
        self.total_samples = 0
        self.batch_size = 0
        self.num_actors = 1
        self.initial_setup_done = False

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
        """Initialize clipping norms."""
        if self.config.clipping_norm is not None:
            # Use provided explicit clipping norm
            self.clipping_norms = {"__default__": self.config.clipping_norm}
        else:
            # For adaptive clipping, start with a conservative default
            default_norm = 0.1  # Start small for updates
            self.clipping_norms = {"__default__": default_norm}
            self.clipping_norm_history.append(default_norm)

    def compute_adaptive_clipping_for_updates(
            self, updates_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute adaptive clipping norms specifically for updates.
        This is crucial for stable DP training.
        """
        if not updates_list:
            return self.clipping_norms

        # Compute norms of updates
        all_norms = []
        per_layer_norms = {}

        for updates in updates_list:
            # Compute global norm of this client's update
            global_norm_sq = 0.0
            for key, update in updates.items():
                if self.config.per_layer_clipping:
                    # Track per-layer norms
                    if key not in per_layer_norms:
                        per_layer_norms[key] = []
                    layer_norm = float(np.linalg.norm(update.flatten()))
                    per_layer_norms[key].append(layer_norm)

                global_norm_sq += float(np.sum(np.square(update)))

            global_norm = np.sqrt(global_norm_sq)
            all_norms.append(global_norm)

        # Use a percentile-based approach for robustness
        target_quantile = self.config.adaptive_clipping_quantile

        if self.config.per_layer_clipping:
            # Set per-layer clipping norms
            new_norms = {}
            for key, norms in per_layer_norms.items():
                if norms:
                    # Use percentile of observed norms
                    clip_value = float(np.quantile(norms, target_quantile))
                    # Add some headroom to avoid clipping too many updates
                    clip_value *= 1.5
                    # But ensure it's not too small
                    new_norms[key] = max(clip_value, 0.001)
            return new_norms
        else:
            # Global clipping
            if all_norms:
                clip_value = float(np.quantile(all_norms, target_quantile))
                # Add some headroom
                clip_value *= 1.5
                # But ensure it's not too small
                clip_value = max(clip_value, 0.001)
                return {"__default__": clip_value}

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

    def reset(self) -> None:
        """Reset the privacy manager state."""
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": self.config.target_delta}
        self._initialize_clipping_norms()
        self.clipping_norm_history = []
        self.noise_multiplier_history = []
        self.parameter_norm_cache = {}
