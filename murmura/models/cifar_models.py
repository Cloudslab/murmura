import torch
import torch.nn as nn
from murmura.model.pytorch_model import PyTorchModel


class CIFAR10Model(PyTorchModel):
    """
    CNN model for CIFAR-10 classification.
    Designed for 32x32 RGB images with 10 classes.
    """

    def __init__(self, use_dp_compatible_norm=False):
        super().__init__()

        if use_dp_compatible_norm:
            # Use GroupNorm and LayerNorm for DP compatibility
            self.features = nn.Sequential(
                # First block
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 64),  # GroupNorm is DP-compatible
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                nn.Dropout(0.25),
                
                # Second block
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, 128),  # 16 groups for 128 channels
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, 128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                nn.Dropout(0.25),
                
                # Third block
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, 256),  # 32 groups for 256 channels
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, 256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
                nn.Dropout(0.25),
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                nn.LayerNorm(512),  # LayerNorm is DP-compatible
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )
        else:
            # Standard architecture with BatchNorm for non-DP training
            self.features = nn.Sequential(
                # First block
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                nn.Dropout(0.25),
                
                # Second block
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                nn.Dropout(0.25),
                
                # Third block
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
                nn.Dropout(0.25),
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )

    def forward(self, x):
        # CIFAR-10 images are already 3-channel RGB (3, 32, 32)
        # No need to modify input shape like MNIST
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleCIFAR10Model(PyTorchModel):
    """
    Simpler CIFAR-10 model without normalization layers.
    Works with both DP and non-DP training.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNetCIFAR10(PyTorchModel):
    """
    ResNet-like model for CIFAR-10.
    More sophisticated architecture for better performance.
    """
    
    def __init__(self, use_dp_compatible_norm=False):
        super().__init__()
        
        if use_dp_compatible_norm:
            norm_layer = lambda channels: nn.GroupNorm(min(32, channels//4), channels)
        else:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, norm_layer, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, norm_layer, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, norm_layer, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)
        
    def _make_layer(self, in_planes, planes, blocks, norm_layer, stride=1):
        layers = []
        
        # First block (potentially with stride)
        layers.append(BasicBlock(in_planes, planes, norm_layer, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, norm_layer, stride=1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    
    def __init__(self, in_planes, planes, norm_layer, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out