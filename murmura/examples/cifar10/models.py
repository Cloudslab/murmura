"""Models for CIFAR-10 classification in federated learning."""

from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10ResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 images, 10 classes).

    Modifications from standard ResNet-18:
    - First conv layer uses 3x3 kernel with stride 1 (instead of 7x7 stride 2)
    - Removes initial max pooling layer
    - Final FC layer outputs num_classes

    These changes are standard for CIFAR-10 to preserve spatial resolution
    given the smaller 32x32 input images.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()

        # Use torchvision's ResNet implementation as base
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.model = resnet18(weights=None)
        except ImportError:
            # Fallback for older torchvision versions
            from torchvision.models import resnet18
            self.model = resnet18(pretrained=pretrained)

        # Modify first conv for 32x32 input (smaller kernel, no stride)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove initial maxpool (not needed for small images)
        self.model.maxpool = nn.Identity()

        # Modify final FC for num_classes
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BasicBlock(nn.Module):
    """Basic residual block for custom ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFAR10SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 - faster training, smaller model.

    Architecture:
    - 3 convolutional blocks with BatchNorm and MaxPool
    - 2 fully connected layers
    - ~1.2M parameters (vs ~11M for ResNet-18)

    Suitable for quick experiments and resource-constrained settings.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels, 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 -> 64 channels, 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 -> 128 channels, 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_cifar10_model_factory(
    model_type: str = "resnet18",
    **kwargs
) -> Callable[[], nn.Module]:
    """Get model factory for CIFAR-10.

    Args:
        model_type: Model architecture to use
            - "resnet18": ResNet-18 adapted for CIFAR-10 (~11M params)
            - "simple_cnn": Lightweight CNN (~1.2M params)
        **kwargs: Model-specific parameters (e.g., num_classes, pretrained)

    Returns:
        Factory function that creates model instances
    """
    num_classes = kwargs.get("num_classes", 10)
    pretrained = kwargs.get("pretrained", False)

    if model_type.lower() == "resnet18":
        return lambda: CIFAR10ResNet18(
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif model_type.lower() in ("simple_cnn", "simple", "cnn"):
        return lambda: CIFAR10SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Use 'resnet18' or 'simple_cnn'."
        )


# Factory function aliases for config compatibility
def create_cifar10_resnet18(num_classes: int = 10, **kwargs) -> nn.Module:
    """Create CIFAR-10 ResNet-18 model."""
    return CIFAR10ResNet18(num_classes=num_classes, **kwargs)


def create_cifar10_simple_cnn(num_classes: int = 10, **kwargs) -> nn.Module:
    """Create CIFAR-10 Simple CNN model."""
    return CIFAR10SimpleCNN(num_classes=num_classes)
