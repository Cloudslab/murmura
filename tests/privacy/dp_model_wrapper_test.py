"""
Tests for differential privacy model wrapper.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader, TensorDataset

from murmura.privacy.dp_model_wrapper import DPTorchModelWrapper, OPACUS_AVAILABLE
from murmura.privacy.dp_config import DPConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)


class TestDPTorchModelWrapper:
    """Test DPTorchModelWrapper functionality."""

    @pytest.fixture
    def model(self):
        """Create simple test model."""
        return SimpleModel()

    @pytest.fixture
    def dp_config(self):
        """Create test DP configuration."""
        return DPConfig(
            target_epsilon=1.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            enable_client_dp=True
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        data = np.random.randn(100, 10).astype(np.float32)
        labels = np.random.randint(0, 2, 100)
        return data, labels

    @pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
    def test_init_with_opacus(self, model, dp_config):
        """Test initialization when Opacus is available."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        assert wrapper.dp_config == dp_config
        assert wrapper.privacy_engine is None
        assert wrapper.is_dp_enabled is False
        assert wrapper.privacy_spent == {"epsilon": 0.0, "delta": 0.0}

    def test_init_without_opacus(self, model, dp_config):
        """Test initialization when Opacus is not available."""
        with patch('murmura.privacy.dp_model_wrapper.OPACUS_AVAILABLE', False):
            with pytest.raises(ImportError, match="Opacus is required"):
                DPTorchModelWrapper(
                    model=model,
                    dp_config=dp_config,
                    optimizer_class=torch.optim.SGD,
                    optimizer_kwargs={"lr": 0.01}
                )

    @pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
    def test_setup_differential_privacy(self, model, dp_config):
        """Test differential privacy setup."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        with patch('opacus.PrivacyEngine') as mock_pe, \
             patch.object(wrapper, '_validate_model_for_dp') as mock_validate:
            
            wrapper._setup_differential_privacy()
            
            mock_validate.assert_called_once()
            mock_pe.assert_called_once_with(secure_mode=dp_config.secure_mode)

    @pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
    def test_validate_model_for_dp_success(self, model, dp_config):
        """Test model validation for DP when model is valid."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        with patch('opacus.validators.ModuleValidator') as mock_validator:
            mock_validator.validate.return_value = []  # No errors
            
            wrapper._validate_model_for_dp()
            
            mock_validator.validate.assert_called_once_with(model, strict=False)

    @pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
    def test_validate_model_for_dp_with_fixes(self, model, dp_config):
        """Test model validation when model needs fixes."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        fixed_model = SimpleModel()
        
        with patch('opacus.validators.ModuleValidator') as mock_validator:
            mock_validator.validate.side_effect = [
                ["error1", "error2"],  # Initial validation
                []  # After fixing
            ]
            mock_validator.fix.return_value = fixed_model
            
            wrapper._validate_model_for_dp()
            
            assert wrapper.model is fixed_model
            mock_validator.fix.assert_called_once_with(model)

    def test_compute_privacy_parameters_default(self, model, dp_config):
        """Test privacy parameter computation with default settings."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        noise_mult, sample_rate = wrapper._compute_privacy_parameters(
            dataset_size=1000, batch_size=32, epochs=10
        )
        
        assert isinstance(noise_mult, float)
        assert isinstance(sample_rate, float)
        assert 0.0 < sample_rate <= 1.0

    def test_compute_privacy_parameters_with_amplification(self, model):
        """Test privacy parameter computation with amplification."""
        dp_config = DPConfig(
            target_epsilon=1.0,
            target_delta=1e-5,
            use_amplification_by_subsampling=True,
            amplification_factor=2.0
        )
        
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        noise_mult, sample_rate = wrapper._compute_privacy_parameters(
            dataset_size=1000, batch_size=32, epochs=10
        )
        
        # expected_base_rate = 32 / 1000  # Currently unused
        # The amplified rate should be calculated through get_amplified_sample_rate()
        expected_amplified_rate = dp_config.get_amplified_sample_rate()
        
        assert abs(sample_rate - expected_amplified_rate) < 1e-6

    @pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
    def test_compute_privacy_parameters_auto_tune(self, model):
        """Test privacy parameter computation with auto-tuning."""
        dp_config = DPConfig(
            target_epsilon=1.0,
            target_delta=1e-5,
            auto_tune_noise=True,
            noise_multiplier=None
        )
        
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        with patch('murmura.privacy.dp_model_wrapper.get_noise_multiplier') as mock_get_noise:
            mock_get_noise.return_value = 1.5
            
            noise_mult, sample_rate = wrapper._compute_privacy_parameters(
                dataset_size=1000, batch_size=32, epochs=10
            )
            
            assert noise_mult == 1.5
            mock_get_noise.assert_called_once()

    @pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
    def test_make_private_success(self, model, dp_config, sample_data):
        """Test successful privacy setup."""
        data, labels = sample_data
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        # Create dataloader
        dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
        dataloader = DataLoader(dataset, batch_size=32)
        
        mock_private_dataloader = Mock()
        
        with patch.object(wrapper, '_setup_differential_privacy'), \
             patch('murmura.privacy.dp_model_wrapper.PrivacyEngine') as mock_pe_class:
            
            mock_pe = Mock()
            mock_pe_class.return_value = mock_pe
            mock_pe.make_private_with_epsilon.return_value = (
                model, wrapper.optimizer, mock_private_dataloader
            )
            wrapper.privacy_engine = mock_pe
            
            wrapper._make_private(dataloader, epochs=5, dataset_size=100)
            
            assert wrapper.is_dp_enabled is True
            assert result is mock_private_dataloader

    @pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
    def test_make_private_fallback(self, model, dp_config, sample_data):
        """Test privacy setup with fallback to manual method."""
        data, labels = sample_data
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
        dataloader = DataLoader(dataset, batch_size=32)
        
        mock_private_dataloader = Mock()
        
        with patch.object(wrapper, '_setup_differential_privacy'), \
             patch('murmura.privacy.dp_model_wrapper.PrivacyEngine') as mock_pe_class:
            
            mock_pe = Mock()
            mock_pe_class.return_value = mock_pe
            # First method fails
            mock_pe.make_private_with_epsilon.side_effect = Exception("Failed")
            # Second method succeeds
            mock_pe.make_private.return_value = (
                model, wrapper.optimizer, mock_private_dataloader
            )
            wrapper.privacy_engine = mock_pe
            
            wrapper._make_private(dataloader, epochs=5, dataset_size=100)
            
            assert wrapper.is_dp_enabled is True
            mock_pe.make_private.assert_called_once()

    def test_make_private_already_enabled(self, model, dp_config, sample_data):
        """Test make_private when DP is already enabled."""
        data, labels = sample_data
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
        dataloader = DataLoader(dataset, batch_size=32)
        
        wrapper.is_dp_enabled = True
        
        result = wrapper._make_private(dataloader, epochs=5, dataset_size=100)
        
        assert result is dataloader  # Returns original dataloader

    @pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
    def test_train_with_dp_enabled(self, model, dp_config, sample_data):
        """Test training with DP enabled."""
        data, labels = sample_data
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        with patch.object(wrapper, '_make_private') as mock_make_private, \
             patch.object(wrapper, '_train_epoch') as mock_train_epoch, \
             patch.object(wrapper, '_update_privacy_spent'):
            
            mock_make_private.return_value = Mock()
            mock_train_epoch.return_value = (10.0, 80, 100)  # loss, correct, total
            wrapper.is_dp_enabled = True
            
            result = wrapper.train(data, labels, batch_size=32, epochs=2)
            
            assert "loss" in result
            assert "accuracy" in result
            assert "privacy_spent" in result
            assert "dp_enabled" in result
            assert result["dp_enabled"] is True

    def test_train_without_dp(self, model, sample_data):
        """Test training without DP enabled."""
        data, labels = sample_data
        dp_config = DPConfig(enable_client_dp=False, enable_central_dp=True)
        
        with patch('murmura.privacy.dp_model_wrapper.OPACUS_AVAILABLE', True):
            wrapper = DPTorchModelWrapper(
                model=model,
                dp_config=dp_config,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.SGD,
                optimizer_kwargs={"lr": 0.01}
            )
            
            with patch.object(wrapper, '_train_epoch') as mock_train_epoch:
                mock_train_epoch.return_value = (10.0, 80, 100)
                
                result = wrapper.train(data, labels, batch_size=32, epochs=2)
                
                assert result["dp_enabled"] is False

    def test_train_epoch(self, model, dp_config, sample_data):
        """Test single epoch training."""
        data, labels = sample_data
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01},
            device="cpu"
        )
        
        # Create small dataset for testing
        small_data = data[:20]
        small_labels = labels[:20]
        dataset = TensorDataset(
            torch.tensor(small_data), 
            torch.tensor(small_labels, dtype=torch.long)
        )
        dataloader = DataLoader(dataset, batch_size=10)
        
        loss, correct, total = wrapper._train_epoch(
            dataloader, epoch=0, total_epochs=1, verbose=False, log_interval=1
        )
        
        assert isinstance(loss, float)
        assert isinstance(correct, int)
        assert isinstance(total, int)
        assert total == 20

    def test_update_privacy_spent(self, model, dp_config):
        """Test privacy accounting update."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        mock_pe = Mock()
        mock_pe.get_epsilon.return_value = 0.5
        wrapper.privacy_engine = mock_pe
        wrapper.is_dp_enabled = True
        
        wrapper._update_privacy_spent()
        
        assert wrapper.privacy_spent["epsilon"] == 0.5
        assert wrapper.privacy_spent["delta"] == dp_config.target_delta

    def test_get_privacy_spent(self, model, dp_config):
        """Test getting privacy spent."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        wrapper.privacy_spent = {"epsilon": 0.5, "delta": 1e-5}
        
        result = wrapper.get_privacy_spent()
        
        assert result == {"epsilon": 0.5, "delta": 1e-5}
        assert result is not wrapper.privacy_spent  # Should be a copy

    def test_is_privacy_budget_exhausted(self, model, dp_config):
        """Test privacy budget exhaustion check."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        # Not DP enabled
        assert wrapper.is_privacy_budget_exhausted() is False
        
        # DP enabled but budget not exhausted
        wrapper.is_dp_enabled = True
        wrapper.privacy_spent = {"epsilon": 0.5, "delta": 1e-5}
        assert wrapper.is_privacy_budget_exhausted() is False
        
        # Budget exhausted
        wrapper.privacy_spent = {"epsilon": 2.0, "delta": 1e-5}  # > target_epsilon
        assert wrapper.is_privacy_budget_exhausted() is True

    def test_get_dp_parameters(self, model, dp_config):
        """Test getting DP parameters."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        wrapper.is_dp_enabled = True
        wrapper.privacy_spent = {"epsilon": 0.5, "delta": 1e-5}
        
        result = wrapper.get_dp_parameters()
        
        expected_keys = [
            "dp_enabled", "target_epsilon", "target_delta", "max_grad_norm",
            "noise_multiplier", "current_privacy_spent", "privacy_exhausted"
        ]
        
        for key in expected_keys:
            assert key in result
        
        assert result["dp_enabled"] is True
        assert result["target_epsilon"] == dp_config.target_epsilon

    def test_get_parameters_with_opacus_wrapping(self, model, dp_config):
        """Test parameter extraction with Opacus module wrapping."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        # Mock Opacus wrapping by adding _module. prefix
        original_state_dict = model.state_dict()
        wrapped_state_dict = {f"_module.{k}": v for k, v in original_state_dict.items()}
        
        with patch.object(wrapper.model, 'state_dict', return_value=wrapped_state_dict):
            params = wrapper.get_parameters()
        
        # Should remove _module. prefix
        for key in params:
            assert not key.startswith("_module.")
        
        # Should have same keys as original model
        assert set(params.keys()) == set(original_state_dict.keys())

    def test_set_parameters_with_opacus_wrapping(self, model, dp_config):
        """Test parameter setting with Opacus module wrapping."""
        with patch('murmura.privacy.dp_model_wrapper.OPACUS_AVAILABLE', True):
            wrapper = DPTorchModelWrapper(
                model=model,
                dp_config=dp_config,
                optimizer_class=torch.optim.SGD,
                optimizer_kwargs={"lr": 0.01}
            )
            
            # Simulate DP enabled with wrapped model without recursion
            wrapper.is_dp_enabled = True
            
            # Mock the _module attribute without creating recursion
            mock_module = Mock()
            wrapper.model._module = mock_module
            
            # Test parameters modification
            test_params = {"linear.weight": np.array([[1.0, 2.0]]), "linear.bias": np.array([0.5])}
            
            with patch.object(wrapper.model, 'load_state_dict') as mock_load:
                wrapper.set_parameters(test_params)
                
                # Should add _module. prefix for wrapped model
                call_args = mock_load.call_args[0][0]
                for key in call_args:
                    assert key.startswith("_module.")

    def test_set_parameters_without_wrapping(self, model, dp_config):
        """Test parameter setting without Opacus wrapping."""
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        original_params = wrapper.get_parameters()
        modified_params = {k: v + 0.1 for k, v in original_params.items()}
        
        with patch.object(wrapper.model, 'load_state_dict') as mock_load:
            wrapper.set_parameters(modified_params)
            
            # Should not add _module. prefix for non-wrapped model
            call_args = mock_load.call_args[0][0]
            for key in call_args:
                assert not key.startswith("_module.")

    def test_privacy_budget_check_during_training(self, model, sample_data):
        """Test privacy budget checking during training."""
        data, labels = sample_data
        dp_config = DPConfig(
            target_epsilon=1.0,
            target_delta=1e-5,
            strict_privacy_check=True,
            enable_client_dp=True
        )
        
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        with patch.object(wrapper, '_make_private') as mock_make_private, \
             patch.object(wrapper, '_train_epoch') as mock_train_epoch, \
             patch.object(wrapper, '_update_privacy_spent'):
            
            mock_make_private.return_value = Mock()
            mock_train_epoch.return_value = (10.0, 80, 100)
            wrapper.is_dp_enabled = True
            
            # Simulate privacy budget exhaustion on second epoch
            def update_privacy_side_effect():
                if mock_update.call_count == 1:
                    wrapper.privacy_spent = {"epsilon": 0.5, "delta": 1e-5}
                else:
                    wrapper.privacy_spent = {"epsilon": 1.5, "delta": 1e-5}  # Exceeds target
            
            mock_update.side_effect = update_privacy_side_effect
            
            wrapper.train(data, labels, batch_size=32, epochs=5)
            
            # Should stop early due to privacy budget exhaustion
            assert mock_train_epoch.call_count < 5

    def test_memory_manager_for_large_batches(self, model, dp_config, sample_data):
        """Test that BatchMemoryManager is used for large batches when DP is enabled."""
        data, labels = sample_data
        wrapper = DPTorchModelWrapper(
            model=model,
            dp_config=dp_config,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01}
        )
        
        with patch.object(wrapper, '_make_private') as mock_make_private, \
             patch.object(wrapper, '_train_epoch') as mock_train_epoch, \
             patch('murmura.privacy.dp_model_wrapper.BatchMemoryManager') as mock_bmm:
            
            mock_make_private.return_value = Mock()
            mock_train_epoch.return_value = (10.0, 80, 100)
            wrapper.is_dp_enabled = True
            
            # Use large batch size to trigger memory manager
            wrapper.train(data, labels, batch_size=128, epochs=1)
            
            # BatchMemoryManager should be used
            mock_bmm.assert_called_once()