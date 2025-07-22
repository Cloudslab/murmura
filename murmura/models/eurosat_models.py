"""
EuroSAT Satellite Image Classification Models

This module contains PyTorch model architectures for the EuroSAT dataset.
EuroSAT is a dataset of satellite images (64x64 RGB) with 10 land cover classes.

Classes:
- Annual Crop
- Forest  
- Herbaceous Vegetation
- Highway
- Industrial Buildings
- Pasture
- Permanent Crop
- Residential Buildings
- River
- SeaLake
"""

import torch
import torch.nn as nn
from murmura.model.pytorch_model import PyTorchModel


class EuroSATModel(PyTorchModel):
    """Simple EuroSAT model for satellite image classification."""
    
    def __init__(self, input_size: int = 64, use_dp_compatible_norm: bool = False):
        """
        Initialize EuroSAT model.
        
        Args:
            input_size: Input image size (default: 64 for EuroSAT)
            use_dp_compatible_norm: Whether to use GroupNorm instead of BatchNorm for DP compatibility
        """
        super().__init__()
        self.input_size = input_size
        self.use_dp_compatible_norm = use_dp_compatible_norm
        
        # EuroSAT has 10 classes
        self.num_classes = 10
        
        # Choose normalization layer based on DP compatibility
        if use_dp_compatible_norm:
            # GroupNorm with 8 groups for DP compatibility
            def norm_layer(channels):
                return nn.GroupNorm(min(8, channels), channels)
        else:
            # Standard BatchNorm
            def norm_layer(channels):
                return nn.BatchNorm2d(channels)
        
        # Feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        
        # Calculate flattened size based on input size
        flattened_size = self._calculate_flattened_size()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_flattened_size(self):
        """Calculate the flattened size after feature extraction."""
        # Create a dummy input to calculate the size
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
        dummy_output = self.features(dummy_input)
        return dummy_output.numel()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class EuroSATModelComplex(PyTorchModel):
    """Complex EuroSAT model with more layers and parameters."""
    
    def __init__(self, input_size: int = 64, use_dp_compatible_norm: bool = False):
        """
        Initialize complex EuroSAT model.
        
        Args:
            input_size: Input image size (default: 64 for EuroSAT)
            use_dp_compatible_norm: Whether to use GroupNorm instead of BatchNorm for DP compatibility
        """
        super().__init__()
        self.input_size = input_size
        self.use_dp_compatible_norm = use_dp_compatible_norm
        
        # EuroSAT has 10 classes
        self.num_classes = 10
        
        # Choose normalization layer based on DP compatibility
        if use_dp_compatible_norm:
            # GroupNorm with 8 groups for DP compatibility
            def norm_layer(channels):
                return nn.GroupNorm(min(8, channels), channels)
        else:
            # Standard BatchNorm
            def norm_layer(channels):
                return nn.BatchNorm2d(channels)
        
        # Feature extractor with residual-like blocks
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            
            # Fifth conv block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2
        )
        
        # Calculate flattened size based on input size
        flattened_size = self._calculate_flattened_size()
        
        # Classifier with more layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_flattened_size(self):
        """Calculate the flattened size after feature extraction."""
        # Create a dummy input to calculate the size
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
        dummy_output = self.features(dummy_input)
        return dummy_output.numel()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class EuroSATModelLite(PyTorchModel):
    """Lightweight EuroSAT model for resource-constrained environments."""
    
    def __init__(self, input_size: int = 64, use_dp_compatible_norm: bool = False):
        """
        Initialize lightweight EuroSAT model.
        
        Args:
            input_size: Input image size (default: 64 for EuroSAT)
            use_dp_compatible_norm: Whether to use GroupNorm instead of BatchNorm for DP compatibility
        """
        super().__init__()
        self.input_size = input_size
        self.use_dp_compatible_norm = use_dp_compatible_norm
        
        # EuroSAT has 10 classes
        self.num_classes = 10
        
        # Choose normalization layer based on DP compatibility
        if use_dp_compatible_norm:
            # GroupNorm with 4 groups for DP compatibility (fewer groups for lite model)
            def norm_layer(channels):
                return nn.GroupNorm(min(4, channels), channels)
        else:
            # Standard BatchNorm
            def norm_layer(channels):
                return nn.BatchNorm2d(channels)
        
        # Lightweight feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            norm_layer(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # Global average pooling to reduce parameters
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x