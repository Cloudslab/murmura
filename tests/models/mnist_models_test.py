"""
Tests for MNIST model implementations.
"""
import torch
import torch.nn as nn

from murmura.models.mnist_models import MNISTModel, SimpleMNISTModel
from murmura.model.pytorch_model import PyTorchModel


class TestMNISTModel:
    """Test MNISTModel functionality."""

    def test_inheritance(self):
        """Test that MNISTModel inherits from PyTorchModel."""
        model = MNISTModel()
        assert isinstance(model, PyTorchModel)
        assert isinstance(model, nn.Module)

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = MNISTModel()
        
        # Should have features and classifier attributes
        assert hasattr(model, 'features')
        assert hasattr(model, 'classifier')
        assert isinstance(model.features, nn.Sequential)
        assert isinstance(model.classifier, nn.Sequential)

    def test_init_dp_compatible(self):
        """Test initialization with DP-compatible normalization."""
        model = MNISTModel(use_dp_compatible_norm=True)
        
        # Check that GroupNorm and LayerNorm are used instead of BatchNorm
        feature_modules = list(model.features.modules())
        classifier_modules = list(model.classifier.modules())
        
        # Should contain GroupNorm in features
        has_group_norm = any(isinstance(m, nn.GroupNorm) for m in feature_modules)
        assert has_group_norm
        
        # Should contain LayerNorm in classifier
        has_layer_norm = any(isinstance(m, nn.LayerNorm) for m in classifier_modules)
        assert has_layer_norm
        
        # Should not contain BatchNorm when DP-compatible
        has_batch_norm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) 
                           for m in feature_modules + classifier_modules)
        assert not has_batch_norm

    def test_init_standard(self):
        """Test initialization with standard normalization."""
        model = MNISTModel(use_dp_compatible_norm=False)
        
        feature_modules = list(model.features.modules())
        classifier_modules = list(model.classifier.modules())
        
        # Should contain BatchNorm
        has_batch_norm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) 
                           for m in feature_modules + classifier_modules)
        assert has_batch_norm

    def test_forward_4d_input(self):
        """Test forward pass with 4D input (batch, channel, height, width)."""
        model = MNISTModel()
        
        # Standard MNIST input shape
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)  # Batch size 2, 10 classes
        assert not torch.isnan(output).any()

    def test_forward_3d_input(self):
        """Test forward pass with 3D input (batch, height, width) - should add channel."""
        model = MNISTModel()
        
        # Input without channel dimension
        x = torch.randn(2, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)  # Should still work
        assert not torch.isnan(output).any()

    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = MNISTModel()
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that some parameters require gradients
        trainable_params = [p for p in params if p.requires_grad]
        assert len(trainable_params) > 0

    def test_model_eval_mode(self):
        """Test switching between train and eval modes."""
        model = MNISTModel()
        
        # Default should be training mode
        assert model.training
        
        # Switch to eval
        model.eval()
        assert not model.training
        
        # Switch back to train
        model.train()
        assert model.training

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        model = MNISTModel()
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(4, 1, 28, 28)
        targets = torch.randint(0, 10, (4,))
        
        output = model(x)
        loss = criterion(output, targets)
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients

    def test_dp_model_architecture(self):
        """Test DP-compatible model architecture details."""
        model = MNISTModel(use_dp_compatible_norm=True)
        
        # Check that features and classifier are Sequential modules
        assert isinstance(model.features, nn.Sequential)
        assert isinstance(model.classifier, nn.Sequential)
        
        # Check that first conv layer has correct input channels
        first_conv = model.features[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1  # Grayscale input
        assert first_conv.out_channels == 32

    def test_standard_model_architecture(self):
        """Test standard model architecture details."""
        model = MNISTModel(use_dp_compatible_norm=False)
        
        # Check that first conv layer is correct
        first_conv = model.features[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 32

    def test_output_distribution(self):
        """Test that model outputs reasonable distributions."""
        model = MNISTModel()
        model.eval()
        
        x = torch.randn(10, 1, 28, 28)
        
        with torch.no_grad():
            output = model(x)
        
        # Output should have shape (batch_size, num_classes)
        assert output.shape == (10, 10)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)
        
        # Probabilities should sum to 1 for each sample
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)


class TestSimpleMNISTModel:
    """Test SimpleMNISTModel functionality."""

    def test_inheritance(self):
        """Test that SimpleMNISTModel inherits from PyTorchModel."""
        model = SimpleMNISTModel()
        assert isinstance(model, PyTorchModel)
        assert isinstance(model, nn.Module)

    def test_init(self):
        """Test initialization."""
        model = SimpleMNISTModel()
        
        assert hasattr(model, 'features')
        assert hasattr(model, 'classifier')
        assert isinstance(model.features, nn.Sequential)
        assert isinstance(model.classifier, nn.Sequential)

    def test_no_normalization_layers(self):
        """Test that model has no normalization layers."""
        model = SimpleMNISTModel()
        
        all_modules = list(model.modules())
        
        # Should not contain any normalization layers
        norm_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm]
        has_norm = any(isinstance(m, tuple(norm_layers)) for m in all_modules)
        assert not has_norm

    def test_forward_4d_input(self):
        """Test forward pass with 4D input."""
        model = SimpleMNISTModel()
        
        x = torch.randn(3, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (3, 10)
        assert not torch.isnan(output).any()

    def test_forward_3d_input(self):
        """Test forward pass with 3D input."""
        model = SimpleMNISTModel()
        
        x = torch.randn(3, 28, 28)
        output = model(x)
        
        assert output.shape == (3, 10)
        assert not torch.isnan(output).any()

    def test_model_simplicity(self):
        """Test that model is simpler than MNISTModel."""
        simple_model = SimpleMNISTModel()
        regular_model = MNISTModel()
        
        # Count parameters
        simple_params = sum(p.numel() for p in simple_model.parameters())
        regular_params = sum(p.numel() for p in regular_model.parameters())
        
        # Simple model should have fewer or equal parameters due to no norm layers
        assert simple_params <= regular_params

    def test_architecture_components(self):
        """Test specific architecture components."""
        model = SimpleMNISTModel()
        
        # Features should have conv, relu, pool, conv, relu, pool
        assert len(model.features) == 6
        
        # Classifier should have linear, relu, linear
        assert len(model.classifier) == 3
        
        # Check first conv layer
        first_conv = model.features[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 32

    def test_gradient_flow(self):
        """Test gradient flow through simple model."""
        model = SimpleMNISTModel()
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(2, 1, 28, 28)
        targets = torch.randint(0, 10, (2,))
        
        output = model(x)
        loss = criterion(output, targets)
        loss.backward()
        
        # All parameters should have gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestModelComparison:
    """Test comparisons between different MNIST models."""

    def test_both_models_same_output_shape(self):
        """Test that both models produce same output shape."""
        mnist_model = MNISTModel()
        simple_model = SimpleMNISTModel()
        
        x = torch.randn(5, 1, 28, 28)
        
        output1 = mnist_model(x)
        output2 = simple_model(x)
        
        assert output1.shape == output2.shape == (5, 10)

    def test_both_models_handle_3d_input(self):
        """Test that both models handle 3D input correctly."""
        mnist_model = MNISTModel()
        simple_model = SimpleMNISTModel()
        
        x = torch.randn(3, 28, 28)
        
        output1 = mnist_model(x)
        output2 = simple_model(x)
        
        assert output1.shape == output2.shape == (3, 10)

    def test_dp_vs_standard_mnist_model(self):
        """Test DP-compatible vs standard MNIST model."""
        dp_model = MNISTModel(use_dp_compatible_norm=True)
        standard_model = MNISTModel(use_dp_compatible_norm=False)
        
        x = torch.randn(2, 1, 28, 28)
        
        output1 = dp_model(x)
        output2 = standard_model(x)
        
        # Same output shape
        assert output1.shape == output2.shape == (2, 10)
        
        # Different internal architectures
        dp_modules = set(type(m).__name__ for m in dp_model.modules())
        standard_modules = set(type(m).__name__ for m in standard_model.modules())
        
        assert 'GroupNorm' in dp_modules
        assert 'LayerNorm' in dp_modules
        assert 'BatchNorm2d' in standard_modules
        assert 'BatchNorm1d' in standard_modules