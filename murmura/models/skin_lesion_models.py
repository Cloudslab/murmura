import torch
import torch.nn as nn
import torch.nn.functional as func
from murmura.model.pytorch_model import PyTorchModel


class BasicBlock(PyTorchModel):
    """Basic block for WideResNet with configurable normalization."""

    def __init__(
        self, in_planes, out_planes, stride, drop_rate=0.0, use_dp_compatible_norm=False
    ):
        super(BasicBlock, self).__init__()

        if use_dp_compatible_norm:
            # Use GroupNorm for DP compatibility
            self.bn1 = nn.GroupNorm(min(8, in_planes), in_planes)
            self.bn2 = nn.GroupNorm(min(8, out_planes), out_planes)
        else:
            # Use BatchNorm for standard training
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(out_planes)

        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.drop_rate = drop_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        out = None
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = func.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(PyTorchModel):
    """Network block for WideResNet."""

    def __init__(
        self,
        nb_layers,
        in_planes,
        out_planes,
        block,
        stride,
        drop_rate=0.0,
        use_dp_compatible_norm=False,
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self.make_layer(
            block,
            in_planes,
            out_planes,
            nb_layers,
            stride,
            drop_rate,
            use_dp_compatible_norm,
        )

    @staticmethod
    def make_layer(
        block,
        in_planes,
        out_planes,
        nb_layers,
        stride,
        drop_rate,
        use_dp_compatible_norm,
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    drop_rate,
                    use_dp_compatible_norm,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(PyTorchModel):
    """
    WideResNet model for skin lesion classification.
    Supports both DP-compatible and standard normalization.
    """

    def __init__(
        self,
        depth=16,
        num_classes=7,
        widen_factor=8,
        drop_rate=0.3,
        use_dp_compatible_norm=False,
    ):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # Network blocks
        self.block1 = NetworkBlock(
            n, n_channels[0], n_channels[1], block, 1, drop_rate, use_dp_compatible_norm
        )
        self.block2 = NetworkBlock(
            n, n_channels[1], n_channels[2], block, 2, drop_rate, use_dp_compatible_norm
        )
        self.block3 = NetworkBlock(
            n, n_channels[2], n_channels[3], block, 2, drop_rate, use_dp_compatible_norm
        )

        # Final normalization and classifier
        if use_dp_compatible_norm:
            self.bn1 = nn.GroupNorm(min(8, n_channels[3]), n_channels[3])
        else:
            self.bn1 = nn.BatchNorm2d(n_channels[3])

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.nChannels = n_channels[3]

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    m.weight.data.fill_(1)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Ensure input has correct format
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Medical images should be RGB (3 channels)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = func.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


class SimpleWideResNet(PyTorchModel):
    """
    Simplified WideResNet without normalization layers.
    Works with both DP and non-DP training.
    """

    def __init__(self, num_classes=7, base_channels=16):
        super(SimpleWideResNet, self).__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Second block
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Third block
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 8 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Ensure input has correct format
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Medical images should be RGB (3 channels)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
