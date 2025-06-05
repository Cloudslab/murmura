import torch
import torch.nn as nn
from murmura.model.pytorch_model import PyTorchModel


class MNISTModel(PyTorchModel):
    """
    Simple CNN model for MNIST classification.
    Consistent architecture for both DP and non-DP training.
    """

    def __init__(self, use_dp_compatible_norm=False):
        super().__init__()
        
        if use_dp_compatible_norm:
            # Use GroupNorm and LayerNorm for DP compatibility
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 32),  # GroupNorm is DP-compatible
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 64),  # 8 groups for 64 channels
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.LayerNorm(128),  # LayerNorm is DP-compatible
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10)
            )
        else:
            # Standard architecture with BatchNorm for non-DP training
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10)
            )

    def forward(self, x):
        # Ensure input has the right shape (add channel dimension if needed)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleMNISTModel(PyTorchModel):
    """
    Very simple MNIST model without normalization layers.
    Works with both DP and non-DP training.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), 
            nn.ReLU(), 
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Ensure input has the right shape (add channel dimension if needed)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x