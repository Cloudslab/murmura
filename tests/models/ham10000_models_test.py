import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from murmura.models.ham10000_models import (
    HAM10000Model, 
    HAM10000ModelComplex, 
    SimpleHAM10000Model,
    ResidualBlock
)


class TestHAM10000Model:
    """Test cases for HAM10000Model."""

    def test_init_standard_architecture(self):
        """Test initialization with standard BatchNorm architecture."""
        model = HAM10000Model(input_size=224, use_dp_compatible_norm=False)
        
        assert model.input_size == 224
        assert isinstance(model.features, nn.Sequential)
        assert isinstance(model.classifier, nn.Sequential)
        
        # Check that BatchNorm layers are used
        has_batchnorm = any(isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)) 
                           for layer in model.modules())
        assert has_batchnorm

    def test_init_dp_compatible_architecture(self):
        """Test initialization with DP-compatible normalization."""
        model = HAM10000Model(input_size=224, use_dp_compatible_norm=True)
        
        assert model.input_size == 224
        
        # Check that GroupNorm and LayerNorm are used instead of BatchNorm
        has_groupnorm = any(isinstance(layer, nn.GroupNorm) for layer in model.modules())
        has_layernorm = any(isinstance(layer, nn.LayerNorm) for layer in model.modules())
        has_batchnorm = any(isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)) 
                           for layer in model.modules())
        
        assert has_groupnorm
        assert has_layernorm
        assert not has_batchnorm

    def test_init_custom_input_size(self):
        """Test initialization with custom input size."""
        input_size = 128
        model = HAM10000Model(input_size=input_size)
        
        assert model.input_size == input_size
        
        # Feature map size should be input_size // 8 due to 3 MaxPool layers
        expected_feature_size = input_size // 8
        
        # Check classifier input size is calculated correctly
        first_linear = None
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break
        
        expected_input_features = 128 * expected_feature_size * expected_feature_size
        assert first_linear.in_features == expected_input_features

    def test_forward_with_batch_dimension(self):
        """Test forward pass with proper batch dimension."""
        model = HAM10000Model(input_size=64)  # Smaller size for faster testing
        
        # Input with batch dimension: [batch_size, channels, height, width]
        batch_input = torch.randn(2, 3, 64, 64)
        output = model(batch_input)
        
        # Output should be [batch_size, num_classes]
        assert output.shape == (2, 7)  # 7 classes for HAM10000

    def test_forward_without_batch_dimension(self):
        """Test forward pass adds batch dimension when missing."""
        model = HAM10000Model(input_size=64)
        
        # Input without batch dimension: [channels, height, width]
        single_input = torch.randn(3, 64, 64)
        output = model(single_input)
        
        # Output should be [1, num_classes] after adding batch dimension
        assert output.shape == (1, 7)

    def test_forward_output_range(self):
        """Test that forward pass produces reasonable output values."""
        model = HAM10000Model(input_size=64)
        model.eval()
        
        input_tensor = torch.randn(1, 3, 64, 64)
        output = model(input_tensor)
        
        # Output should be finite
        assert torch.isfinite(output).all()
        
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0), atol=1e-6)

    def test_model_parameters_trainable(self):
        """Test that model parameters are trainable by default."""
        model = HAM10000Model()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        assert trainable_params > 0
        assert trainable_params == total_params


class TestHAM10000ModelComplex:
    """Test cases for HAM10000ModelComplex."""

    def test_init_standard_architecture(self):
        """Test initialization with standard architecture."""
        model = HAM10000ModelComplex(input_size=224, use_dp_compatible_norm=False)
        
        assert model.input_size == 224
        assert not model.use_dp_compatible_norm
        assert isinstance(model.layer1, nn.Sequential)
        assert isinstance(model.layer2, nn.Sequential)
        assert isinstance(model.layer3, nn.Sequential)

    def test_init_dp_compatible_architecture(self):
        """Test initialization with DP-compatible architecture."""
        model = HAM10000ModelComplex(input_size=224, use_dp_compatible_norm=True)
        
        assert model.use_dp_compatible_norm
        
        # Check normalization layer types
        has_groupnorm = any(isinstance(layer, nn.GroupNorm) for layer in model.modules())
        has_layernorm = any(isinstance(layer, nn.LayerNorm) for layer in model.modules())
        has_batchnorm = any(isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)) 
                           for layer in model.modules())
        
        assert has_groupnorm
        assert has_layernorm
        assert not has_batchnorm

    def test_make_layer_structure(self):
        """Test that _make_layer creates correct structure."""
        model = HAM10000ModelComplex(input_size=224, use_dp_compatible_norm=False)
        
        # Test layer creation
        layer = model._make_layer(64, 128, 2, stride=2)
        assert isinstance(layer, nn.Sequential)
        assert len(layer) == 2  # Should have 2 blocks
        
        # First block should be ResidualBlock
        assert isinstance(layer[0], ResidualBlock)
        assert isinstance(layer[1], ResidualBlock)

    def test_make_layer_with_downsample(self):
        """Test _make_layer with downsampling."""
        model = HAM10000ModelComplex(input_size=224, use_dp_compatible_norm=False)
        
        # When stride != 1 or channels change, should create downsample
        layer = model._make_layer(64, 128, 1, stride=2)
        residual_block = layer[0]
        
        assert residual_block.downsample is not None
        assert isinstance(residual_block.downsample, nn.Sequential)

    def test_forward_with_batch_dimension(self):
        """Test forward pass with batch dimension."""
        model = HAM10000ModelComplex(input_size=64)
        
        batch_input = torch.randn(2, 3, 64, 64)
        output = model(batch_input)
        
        assert output.shape == (2, 7)

    def test_forward_without_batch_dimension(self):
        """Test forward pass adds batch dimension when missing."""
        model = HAM10000ModelComplex(input_size=64)
        
        single_input = torch.randn(3, 64, 64)
        output = model(single_input)
        
        assert output.shape == (1, 7)

    def test_adaptive_pooling_works(self):
        """Test that adaptive average pooling produces correct output size."""
        model = HAM10000ModelComplex(input_size=64)
        
        # Test with different input sizes
        for size in [32, 64, 128]:
            input_tensor = torch.randn(1, 3, size, size)
            
            # Extract features before classifier
            x = model.conv1(input_tensor)
            x = model.norm1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.avgpool(x)
            
            # Should always be (1, 1) after adaptive pooling
            assert x.shape[-2:] == (1, 1)


class TestResidualBlock:
    """Test cases for ResidualBlock."""

    def test_init_standard_norm(self):
        """Test initialization with standard normalization."""
        block = ResidualBlock(64, 64, use_dp_compatible_norm=False)
        
        assert isinstance(block.norm1, nn.BatchNorm2d)
        assert isinstance(block.norm2, nn.BatchNorm2d)
        assert block.downsample is None

    def test_init_dp_compatible_norm(self):
        """Test initialization with DP-compatible normalization."""
        block = ResidualBlock(64, 64, use_dp_compatible_norm=True)
        
        assert isinstance(block.norm1, nn.GroupNorm)
        assert isinstance(block.norm2, nn.GroupNorm)

    def test_init_with_downsample(self):
        """Test initialization with downsample layer."""
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )
        block = ResidualBlock(64, 128, stride=2, downsample=downsample)
        
        assert block.downsample is not None
        assert block.downsample == downsample

    def test_forward_no_downsample(self):
        """Test forward pass without downsampling."""
        block = ResidualBlock(64, 64)
        
        input_tensor = torch.randn(1, 64, 32, 32)
        output = block(input_tensor)
        
        # Output should have same shape as input
        assert output.shape == input_tensor.shape

    def test_forward_with_downsample(self):
        """Test forward pass with downsampling."""
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )
        block = ResidualBlock(64, 128, stride=2, downsample=downsample)
        
        input_tensor = torch.randn(1, 64, 32, 32)
        output = block(input_tensor)
        
        # Output should have different channels and spatial size
        assert output.shape == (1, 128, 16, 16)

    def test_forward_residual_connection(self):
        """Test that residual connection works correctly."""
        block = ResidualBlock(64, 64)
        
        # Create input
        input_tensor = torch.randn(1, 64, 32, 32)
        
        # Forward pass
        output = block(input_tensor)
        
        # The output should be different from input due to residual connection
        # and the conv operations, but should have the same shape
        assert output.shape == input_tensor.shape
        assert not torch.equal(output, input_tensor)


class TestSimpleHAM10000Model:
    """Test cases for SimpleHAM10000Model."""

    def test_init(self):
        """Test initialization."""
        model = SimpleHAM10000Model(input_size=224)
        
        assert model.input_size == 224
        assert isinstance(model.features, nn.Sequential)
        assert isinstance(model.classifier, nn.Sequential)

    def test_no_normalization_layers(self):
        """Test that model has no normalization layers."""
        model = SimpleHAM10000Model()
        
        # Should not have any normalization layers
        has_norm = any(isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d, 
                                         nn.GroupNorm, nn.LayerNorm)) 
                      for layer in model.modules())
        assert not has_norm

    def test_forward_with_batch_dimension(self):
        """Test forward pass with batch dimension."""
        model = SimpleHAM10000Model(input_size=64)
        
        batch_input = torch.randn(2, 3, 64, 64)
        output = model(batch_input)
        
        assert output.shape == (2, 7)

    def test_forward_without_batch_dimension(self):
        """Test forward pass adds batch dimension when missing."""
        model = SimpleHAM10000Model(input_size=64)
        
        single_input = torch.randn(3, 64, 64)
        output = model(single_input)
        
        assert output.shape == (1, 7)

    def test_feature_map_size_calculation(self):
        """Test that feature map size is calculated correctly."""
        input_size = 96
        model = SimpleHAM10000Model(input_size=input_size)
        
        # Feature map size should be input_size // 8 due to 3 MaxPool layers
        expected_feature_size = input_size // 8
        
        # Check first linear layer input size
        first_linear = None
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break
        
        expected_input_features = 128 * expected_feature_size * expected_feature_size
        assert first_linear.in_features == expected_input_features

    def test_model_simplicity(self):
        """Test that model is indeed simple (fewer parameters than complex models)."""
        simple_model = SimpleHAM10000Model(input_size=224)
        complex_model = HAM10000ModelComplex(input_size=224)
        
        simple_params = sum(p.numel() for p in simple_model.parameters())
        complex_params = sum(p.numel() for p in complex_model.parameters())
        
        # Simple model should have fewer parameters
        assert simple_params < complex_params

    def test_dropout_layers_present(self):
        """Test that dropout layers are present in classifier."""
        model = SimpleHAM10000Model()
        
        has_dropout = any(isinstance(layer, nn.Dropout) for layer in model.modules())
        assert has_dropout


class TestModelCompatibility:
    """Test compatibility between different HAM10000 models."""

    def test_all_models_same_output_shape(self):
        """Test that all models produce same output shape for same input."""
        input_size = 64
        batch_input = torch.randn(2, 3, input_size, input_size)
        
        models = [
            HAM10000Model(input_size=input_size),
            HAM10000ModelComplex(input_size=input_size),
            SimpleHAM10000Model(input_size=input_size)
        ]
        
        outputs = []
        for model in models:
            output = model(batch_input)
            outputs.append(output.shape)
        
        # All outputs should have same shape
        assert all(shape == outputs[0] for shape in outputs)
        assert outputs[0] == (2, 7)  # 7 classes

    def test_dp_compatible_vs_standard_architecture_compatibility(self):
        """Test that DP and non-DP versions produce same output shapes."""
        input_tensor = torch.randn(1, 3, 64, 64)
        
        standard_model = HAM10000Model(input_size=64, use_dp_compatible_norm=False)
        dp_model = HAM10000Model(input_size=64, use_dp_compatible_norm=True)
        
        standard_output = standard_model(input_tensor)
        dp_output = dp_model(input_tensor)
        
        assert standard_output.shape == dp_output.shape

    def test_models_inherit_from_pytorch_model(self):
        """Test that all models inherit from PyTorchModel."""
        from murmura.model.pytorch_model import PyTorchModel
        
        models = [
            HAM10000Model(),
            HAM10000ModelComplex(),
            SimpleHAM10000Model()
        ]
        
        for model in models:
            assert isinstance(model, PyTorchModel)

    @pytest.mark.parametrize("input_size", [32, 64, 128, 224])
    def test_models_work_with_different_input_sizes(self, input_size):
        """Test that models work with various input sizes."""
        batch_input = torch.randn(1, 3, input_size, input_size)
        
        models = [
            HAM10000Model(input_size=input_size),
            HAM10000ModelComplex(input_size=input_size),
            SimpleHAM10000Model(input_size=input_size)
        ]
        
        for model in models:
            output = model(batch_input)
            assert output.shape == (1, 7)  # Should always output 7 classes
            assert torch.isfinite(output).all()  # Output should be finite