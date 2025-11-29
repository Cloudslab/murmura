#!/usr/bin/env python3
"""
Model variants for scaling experiments.
Provides different sized FEMNIST model architectures with varying parameter counts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FEMNISTTiny(nn.Module):
    """Tiny FEMNIST model with ~200K parameters (30x smaller than baseline)."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.num_classes = num_classes

        # Reduced channels: 8 and 16 instead of 32 and 64
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After pooling: 28->14->7, so 7x7x16 = 784
        # Smaller hidden layer: 256 instead of 2048
        self.fc1 = nn.Linear(7 * 7 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def parameter_count(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FEMNISTSmall(nn.Module):
    """Small FEMNIST model with ~800K parameters (8x smaller than baseline)."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.num_classes = num_classes

        # Reduced channels: 16 and 32 instead of 32 and 64
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After pooling: 28->14->7, so 7x7x32 = 1568
        # Smaller hidden layer: 512 instead of 2048
        self.fc1 = nn.Linear(7 * 7 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def parameter_count(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FEMNISTBaseline(nn.Module):
    """Baseline FEMNIST model (original LEAF architecture) with ~6.5M parameters."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.num_classes = num_classes

        # Original LEAF architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After pooling: 28->14->7, so 7x7x64
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def parameter_count(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FEMNISTLarge(nn.Module):
    """Large FEMNIST model with ~13M parameters (2x larger than baseline)."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.num_classes = num_classes

        # Increased channels: 64 and 128 instead of 32 and 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After pooling: 28->14->7, so 7x7x128 = 6272
        # Larger hidden layer: 4096 instead of 2048
        self.fc1 = nn.Linear(7 * 7 * 128, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def parameter_count(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FEMNISTXLarge(nn.Module):
    """Extra-large FEMNIST model with ~26M parameters (4x larger than baseline)."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.num_classes = num_classes

        # Three convolutional layers for more capacity
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After pooling: 28->14->7, so 7x7x256 = 12544
        # Two hidden layers for more capacity
        self.fc1 = nn.Linear(7 * 7 * 256, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

        # Dropout for regularization in larger model
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def parameter_count(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_variant(variant_name: str, num_classes: int = 62):
    """Factory function to get a specific model variant.

    Args:
        variant_name: One of 'tiny', 'small', 'baseline', 'large', 'xlarge'
        num_classes: Number of output classes

    Returns:
        Model instance
    """
    variants = {
        'tiny': FEMNISTTiny,
        'small': FEMNISTSmall,
        'baseline': FEMNISTBaseline,
        'large': FEMNISTLarge,
        'xlarge': FEMNISTXLarge
    }

    if variant_name not in variants:
        raise ValueError(f"Unknown variant: {variant_name}. Choose from {list(variants.keys())}")

    return variants[variant_name](num_classes)


def compare_model_sizes():
    """Print comparison of all model variant sizes."""
    num_classes = 62
    variants = ['tiny', 'small', 'baseline', 'large', 'xlarge']

    print("=== FEMNIST Model Variants Comparison ===")
    print(f"{'Variant':<12} {'Parameters':>12} {'Relative Size':>15} {'Memory (MB)':>12}")
    print("-" * 52)

    baseline_params = None
    for variant in variants:
        model = get_model_variant(variant, num_classes)
        params = model.parameter_count()

        if variant == 'baseline':
            baseline_params = params

        # Calculate memory in MB (assuming float32: 4 bytes per parameter)
        memory_mb = (params * 4) / (1024 * 1024)

        if baseline_params:
            relative = f"{params / baseline_params:.2f}x"
        else:
            relative = "1.00x (ref)"
            baseline_params = params

        print(f"{variant:<12} {params:>12,} {relative:>15} {memory_mb:>12.2f}")

    print("\n=== Model Complexity Analysis ===")
    for variant in variants:
        model = get_model_variant(variant, num_classes)
        conv_params = sum(p.numel() for name, p in model.named_parameters()
                         if 'conv' in name and p.requires_grad)
        fc_params = sum(p.numel() for name, p in model.named_parameters()
                       if 'fc' in name and p.requires_grad)

        print(f"{variant:<12}: Conv layers: {conv_params:>10,} | FC layers: {fc_params:>10,}")


if __name__ == "__main__":
    # Test instantiation and print sizes
    compare_model_sizes()

    # Test forward pass
    print("\n=== Testing forward pass ===")
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)  # FEMNIST input shape

    for variant in ['tiny', 'small', 'baseline', 'large', 'xlarge']:
        model = get_model_variant(variant)
        model.eval()
        with torch.no_grad():
            output = model(x)
        print(f"{variant:<12}: Input {x.shape} -> Output {output.shape}")