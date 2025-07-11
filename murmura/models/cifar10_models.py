import torch
import torch.nn as nn
from murmura.model.pytorch_model import PyTorchModel


class CIFAR10Model(PyTorchModel):
    """
    CNN model for CIFAR-10 classification.
    Designed for 32x32 RGB images with 10 classes.
    Consistent architecture for both DP and non-DP training.
    """

    def __init__(self, use_dp_compatible_norm=True):
        super().__init__()

        if use_dp_compatible_norm:
            # Use GroupNorm and LayerNorm for DP compatibility
            self.features = nn.Sequential(
                # First block: 3 -> 32 channels
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 32),  # GroupNorm is DP-compatible
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Second block: 32 -> 64 channels
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 64),  # 8 groups for 64 channels
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Third block: 64 -> 128 channels
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 128),  # 8 groups for 128 channels
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.LayerNorm(256),  # LayerNorm is DP-compatible
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10),  # 10 classes for CIFAR-10
            )
        else:
            # Standard architecture with BatchNorm for non-DP training
            self.features = nn.Sequential(
                # First block: 3 -> 32 channels
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Second block: 32 -> 64 channels
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Third block: 64 -> 128 channels
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10),  # 10 classes for CIFAR-10
            )

    def forward(self, x):
        # Ensure input has the right shape
        if len(x.shape) == 3:
            # If missing batch dimension, add it
            x = x.unsqueeze(0)

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleCIFAR10Model(PyTorchModel):
    """
    Very simple CIFAR-10 model without normalization layers.
    Works with both DP and non-DP training.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # First block: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),  # 10 classes for CIFAR-10
        )

    def forward(self, x):
        # Ensure input has the right shape
        if len(x.shape) == 3:
            # If missing batch dimension, add it
            x = x.unsqueeze(0)

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNetCIFAR10Model(PyTorchModel):
    """
    ResNet-inspired model for CIFAR-10 classification.
    Deeper architecture with skip connections for better performance.
    Adapted for 32x32 images.
    """

    def __init__(self, use_dp_compatible_norm=True):
        super().__init__()
        self.use_dp_compatible_norm = use_dp_compatible_norm

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = (
            nn.GroupNorm(8, 64) if use_dp_compatible_norm else nn.BatchNorm2d(64)
        )
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if use_dp_compatible_norm:
            self.classifier = nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),  # 10 classes for CIFAR-10
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),  # 10 classes for CIFAR-10
            )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        # First block with potential downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels:
            if self.use_dp_compatible_norm:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.GroupNorm(8, out_channels),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels),
                )

        layers.append(
            ResidualBlock(
                in_channels,
                out_channels,
                stride,
                downsample,
                self.use_dp_compatible_norm,
            )
        )

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    use_dp_compatible_norm=self.use_dp_compatible_norm,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input has the right shape
        if len(x.shape) == 3:
            # If missing batch dimension, add it
            x = x.unsqueeze(0)

        # Initial layers
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global average pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class ResidualBlock(nn.Module):
    """Basic residual block for ResNetCIFAR10Model"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        use_dp_compatible_norm=False,
    ):
        super().__init__()

        if use_dp_compatible_norm:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.norm1 = nn.GroupNorm(8, out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.norm2 = nn.GroupNorm(8, out_channels)
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.norm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
