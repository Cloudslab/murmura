import logging
import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from murmura.model.pytorch_model import TorchModelWrapper
from murmura.privacy.dp_config import DPConfig

try:
    from opacus import PrivacyEngine  # type: ignore[import-untyped]
    from opacus.utils.batch_memory_manager import BatchMemoryManager  # type: ignore[import-untyped]
    from opacus.validators import ModuleValidator  # type: ignore[import-untyped]
    from opacus.accountants.utils import get_noise_multiplier  # type: ignore[import-untyped]

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = None


class DPTorchModelWrapper(TorchModelWrapper):
    """
    Differential Privacy-aware extension of TorchModelWrapper using Opacus.

    This wrapper provides SOTA differential privacy implementation that maintains
    good utility for both MNIST and skin lesion classification tasks.
    """

    def __init__(
        self,
        model,
        dp_config: DPConfig,
        loss_fn: Optional[nn.Module] = None,
        optimizer_class=None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        data_preprocessor=None,
    ):
        """
        Initialize DP-aware model wrapper.

        Args:
            model: PyTorch model to wrap
            dp_config: Differential privacy configuration
            loss_fn: Loss function
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer kwargs
            device: Device to use
            input_shape: Input shape
            data_preprocessor: Data preprocessor
        """
        # Initialize parent class first
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            input_shape=input_shape,
            data_preprocessor=data_preprocessor,
        )

        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is required for differential privacy. "
                "Install with: pip install opacus"
            )

        self.dp_config = dp_config
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.is_dp_enabled = False
        self.privacy_spent = {"epsilon": 0.0, "delta": 0.0}

        # Setup logging
        self.logger = logging.getLogger("murmura.dp_model_wrapper")

        # Initialize DP components (defer privacy engine setup until training)
        self.logger.info(
            "DP wrapper initialized - privacy engine will be set up during training"
        )

    def _setup_differential_privacy(self) -> None:
        """Setup differential privacy components"""
        try:
            # Validate model compatibility with Opacus
            self._validate_model_for_dp()

            # Create privacy engine
            self.privacy_engine = PrivacyEngine(secure_mode=self.dp_config.secure_mode)

            # Set up accounting method
            if self.dp_config.accounting_method.value == "rdp":
                # Use RDP accountant (default and recommended)
                if self.dp_config.alphas is not None:
                    self.privacy_engine.set_alphas(self.dp_config.alphas)

            self.logger.info(
                f"Initialized DP with target (ε={self.dp_config.target_epsilon}, "
                f"δ={self.dp_config.target_delta})"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup differential privacy: {e}")
            raise

    def _validate_model_for_dp(self) -> None:
        """Validate that the model is compatible with differential privacy"""
        try:
            # Check if model needs fixing for DP
            errors = ModuleValidator.validate(self.model, strict=False)

            if errors:
                self.logger.warning(f"Model validation found {len(errors)} issues")

                # Try to fix automatically
                try:
                    self.model = ModuleValidator.fix(self.model)
                    self.logger.info("Automatically fixed model for DP compatibility")

                    # Re-validate after fixing
                    remaining_errors = ModuleValidator.validate(
                        self.model, strict=False
                    )
                    if remaining_errors:
                        self.logger.warning(
                            f"Still have {len(remaining_errors)} validation issues after auto-fix"
                        )
                        for error in remaining_errors:
                            self.logger.debug(f"Validation issue: {error}")
                except Exception as fix_error:
                    self.logger.warning(f"Could not auto-fix model: {fix_error}")
            else:
                self.logger.info("Model is compatible with differential privacy")

        except Exception as e:
            self.logger.warning(f"Model validation failed: {e}")
            # Continue anyway - some models work even if validation fails

    def _compute_privacy_parameters(
        self, dataset_size: int, batch_size: int, epochs: int
    ) -> Tuple[float, float]:
        """
        Compute optimal privacy parameters for the given training setup.

        Args:
            dataset_size: Total number of training samples
            batch_size: Training batch size
            epochs: Number of training epochs

        Returns:
            Tuple of (noise_multiplier, sample_rate)
        """
        # Compute sample rate
        sample_rate = self.dp_config.sample_rate
        if sample_rate is None:
            sample_rate = batch_size / dataset_size

        # Apply subsampling amplification if enabled
        if self.dp_config.use_amplification_by_subsampling:
            amplified_sample_rate = self.dp_config.get_amplified_sample_rate()
            self.logger.info(
                f"Using amplified sample rate: {amplified_sample_rate:.6f} "
                f"(original: {sample_rate:.6f}, amplification factor: {amplified_sample_rate/sample_rate:.3f})"
            )
            sample_rate = amplified_sample_rate

        # Limit sample rate to reasonable bounds
        sample_rate = min(sample_rate, 1.0)
        sample_rate = max(sample_rate, 0.001)

        # Compute noise multiplier
        noise_multiplier = self.dp_config.noise_multiplier

        if noise_multiplier is None and self.dp_config.auto_tune_noise:
            try:
                # Use Opacus to compute optimal noise multiplier
                steps = epochs * (dataset_size // batch_size)

                noise_multiplier = get_noise_multiplier(
                    target_epsilon=self.dp_config.target_epsilon,
                    target_delta=self.dp_config.target_delta,
                    sample_rate=sample_rate,
                    steps=steps,
                )

                self.logger.info(
                    f"Auto-computed noise multiplier: {noise_multiplier:.3f} "
                    f"for {steps} steps"
                )

            except Exception as e:
                self.logger.warning(
                    f"Could not auto-compute noise multiplier: {e}. "
                    f"Using default value."
                )
                noise_multiplier = 1.0

        if noise_multiplier is None:
            noise_multiplier = 1.0

        return noise_multiplier, sample_rate

    def _make_private(
        self, dataloader: DataLoader, epochs: int, dataset_size: Optional[int] = None
    ) -> DataLoader:
        """
        Make the model and optimizer private using Opacus.

        Args:
            dataloader: Training dataloader
            epochs: Number of training epochs
            dataset_size: Size of dataset (estimated if not provided)

        Returns:
            Private dataloader
        """
        if self.is_dp_enabled:
            self.logger.debug("DP already enabled, skipping re-initialization")
            return dataloader

        # Setup privacy engine if not already done
        if self.privacy_engine is None:
            self._setup_differential_privacy()

        # Ensure optimizer is fresh and matches current model parameters
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )

        try:
            # Estimate dataset size if not provided
            if dataset_size is None:
                if hasattr(dataloader.dataset, '__len__'):
                    dataset_size = len(dataloader.dataset)
                else:
                    raise ValueError("Dataset size must be provided for datasets without __len__ method")

            # Get batch size
            batch_size = dataloader.batch_size or 1

            # Compute privacy parameters
            noise_multiplier, sample_rate = self._compute_privacy_parameters(
                dataset_size=dataset_size, batch_size=batch_size, epochs=epochs
            )

            self.logger.info(
                f"Making model private with noise_multiplier={noise_multiplier:.3f}, "
                f"sample_rate={sample_rate:.3f}, max_grad_norm={self.dp_config.max_grad_norm}"
            )

            # Attach privacy engine - try with epsilon target first, fallback to manual
            if self.privacy_engine is None:
                raise RuntimeError("Privacy engine not initialized")
                
            try:
                (
                    self.model,
                    self.optimizer,
                    private_dataloader,
                ) = self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=dataloader,
                    target_epsilon=self.dp_config.target_epsilon,
                    target_delta=self.dp_config.target_delta,
                    epochs=epochs,
                    max_grad_norm=self.dp_config.max_grad_norm,
                )
                self.logger.info("Successfully made model private with epsilon target")
            except Exception as epsilon_error:
                self.logger.warning(
                    f"make_private_with_epsilon failed: {epsilon_error}"
                )
                self.logger.info("Trying manual privacy engine attachment...")

                # Fallback to manual attachment
                if self.privacy_engine is None:
                    raise RuntimeError("Privacy engine not initialized")
                    
                (
                    self.model,
                    self.optimizer,
                    private_dataloader,
                ) = self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=dataloader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=self.dp_config.max_grad_norm,
                )
                self.logger.info(
                    "Successfully made model private with manual parameters"
                )

            self.is_dp_enabled = True

            # Log privacy parameters
            if self.privacy_engine is not None:
                actual_noise = self.privacy_engine.noise_multiplier
                self.logger.info(
                    f"DP enabled with actual noise multiplier: {actual_noise:.3f}"
                )

            return private_dataloader

        except Exception as e:
            self.logger.error(f"Failed to make model private: {e}")
            # Fall back to non-private training with warning
            warnings.warn(
                f"Differential privacy setup failed: {e}. "
                "Falling back to non-private training.",
                UserWarning,
            )
            return dataloader

    def train(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model with differential privacy.

        Args:
            data: Training data
            labels: Training labels
            **kwargs: Additional training parameters

        Returns:
            Training metrics
        """
        batch_size = kwargs.get("batch_size", 32)
        epochs = kwargs.get("epochs", 1)
        verbose = kwargs.get("verbose", False)
        log_interval = kwargs.get("log_interval", 1)

        # Prepare data
        self.detect_and_set_device()
        dataloader = self._prepare_data(data, labels, batch_size)

        # Enable differential privacy if configured
        if self.dp_config.enable_client_dp:
            dataloader = self._make_private(
                dataloader=dataloader, epochs=epochs, dataset_size=len(data)
            )

        # Training loop with DP considerations
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        try:
            for epoch in range(epochs):
                epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

                # Use BatchMemoryManager for large batches if DP is enabled
                if self.is_dp_enabled and batch_size > 64:
                    # Use memory manager to handle large batches efficiently
                    with BatchMemoryManager(
                        data_loader=dataloader,
                        max_physical_batch_size=32,  # Adjust based on GPU memory
                        optimizer=self.optimizer,
                    ) as memory_safe_dataloader:
                        epoch_stats = self._train_epoch(
                            memory_safe_dataloader, epoch, epochs, verbose, log_interval
                        )
                else:
                    epoch_stats = self._train_epoch(
                        dataloader, epoch, epochs, verbose, log_interval
                    )

                epoch_loss, epoch_correct, epoch_total = epoch_stats
                total_loss += epoch_loss
                correct += epoch_correct
                total += epoch_total

                # Check privacy budget after each epoch if DP is enabled
                if self.is_dp_enabled:
                    self._update_privacy_spent()

                    if self.dp_config.strict_privacy_check:
                        if self.dp_config.is_privacy_exhausted(
                            self.privacy_spent["epsilon"]
                        ):
                            self.logger.warning(
                                f"Privacy budget exhausted! "
                                f"Current ε={self.privacy_spent['epsilon']:.3f}, "
                                f"target ε={self.dp_config.target_epsilon}"
                            )
                            break

            # Final privacy accounting
            if self.is_dp_enabled:
                self._update_privacy_spent()
                privacy_msg = self.dp_config.get_privacy_spent_message(
                    self.privacy_spent["epsilon"], self.privacy_spent["delta"]
                )
                self.logger.info(privacy_msg)

            return {
                "loss": total_loss / total,
                "accuracy": correct / total,
                "privacy_spent": self.privacy_spent.copy(),
                "dp_enabled": self.is_dp_enabled,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int,
        verbose: bool,
        log_interval: int,
    ) -> Tuple[float, int, int]:
        """Train a single epoch"""
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = (
                batch_data.to(self.device),
                batch_labels.to(self.device),
            )

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.loss_fn(outputs, batch_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            epoch_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += batch_labels.size(0)
            epoch_correct += int(torch.eq(predicted, batch_labels).sum().item())

        if verbose and (epoch + 1) % log_interval == 0:
            dp_info = ""
            if self.is_dp_enabled:
                dp_info = f" [DP: ε={self.privacy_spent['epsilon']:.3f}]"

            print(
                f"Epoch [{epoch + 1}/{total_epochs}], "
                f"Loss: {epoch_loss / epoch_total:.4f}, "
                f"Accuracy: {100 * epoch_correct / epoch_total:.2f}%{dp_info}"
            )

        return epoch_loss, epoch_correct, epoch_total

    def _update_privacy_spent(self) -> None:
        """Update privacy accounting"""
        if self.is_dp_enabled and self.privacy_engine:
            try:
                epsilon = self.privacy_engine.get_epsilon(
                    delta=self.dp_config.target_delta
                )
                self.privacy_spent = {
                    "epsilon": epsilon,
                    "delta": self.dp_config.target_delta or 0.0,
                }
            except Exception as e:
                self.logger.warning(f"Could not update privacy accounting: {e}")

    def get_privacy_spent(self) -> Dict[str, float]:
        """Get current privacy expenditure"""
        if self.is_dp_enabled:
            self._update_privacy_spent()
        return self.privacy_spent.copy()

    def is_privacy_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        if not self.is_dp_enabled:
            return False
        return self.dp_config.is_privacy_exhausted(self.privacy_spent["epsilon"])

    def get_dp_parameters(self) -> Dict[str, Any]:
        """Get DP configuration and current state"""
        return {
            "dp_enabled": self.is_dp_enabled,
            "target_epsilon": self.dp_config.target_epsilon,
            "target_delta": self.dp_config.target_delta,
            "max_grad_norm": self.dp_config.max_grad_norm,
            "noise_multiplier": getattr(self.privacy_engine, "noise_multiplier", None),
            "current_privacy_spent": self.privacy_spent.copy(),
            "privacy_exhausted": self.is_privacy_budget_exhausted(),
        }

    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters as CPU tensors, handling Opacus module wrapping."""
        # Move model to CPU temporarily for serialization safety
        self.model.to("cpu")

        state_dict = self.model.state_dict()

        # Handle Opacus module wrapping - remove _module. prefix if present
        cleaned_state_dict = {}
        for name, param in state_dict.items():
            clean_name = (
                name.replace("_module.", "") if name.startswith("_module.") else name
            )
            cleaned_state_dict[clean_name] = param.cpu().numpy()

        # Move back to the original device if set
        if hasattr(self, "device") and self.device != "cpu":
            self.model.to(self.device)
        return cleaned_state_dict

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set parameters safely, handling Opacus module wrapping."""
        # Check if we need to add _module. prefix (model is wrapped by Opacus)
        if self.is_dp_enabled and hasattr(self.model, "_module"):
            # Add _module. prefix to parameters to match wrapped model structure
            wrapped_params = {}
            for name, param in parameters.items():
                if not name.startswith("_module."):
                    wrapped_params[f"_module.{name}"] = torch.tensor(param)
                else:
                    wrapped_params[name] = torch.tensor(param)
            state_dict = wrapped_params
        else:
            # Regular parameter setting
            state_dict = {
                name: torch.tensor(param) for name, param in parameters.items()
            }

        # Always load to CPU first
        self.model.load_state_dict(state_dict)

        # Then move to the target device if needed
        if hasattr(self, "device") and self.device != "cpu":
            self.model.to(self.device)
