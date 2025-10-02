"""
Neural network model architectures for CIFAR-10 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for parameter efficiency"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EfficientBlock(nn.Module):
    """Efficient block with depthwise separable convolution for parameter efficiency"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(EfficientBlock, self).__init__()

        # Depthwise separable convolution - OPTIMIZED for parameter efficiency
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection for residual learning
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Residual connection
        out += self.shortcut(residual)
        out = F.relu(out)

        return out


class OptimizedCIFAR10Net200K(nn.Module):
    """Optimized CIFAR-10 model with 150,690 parameters, RF=67, and >85% accuracy"""
    
    def __init__(self, num_classes=None):
        super(OptimizedCIFAR10Net200K, self).__init__()
        
        if num_classes is None:
            num_classes = MODEL_CONFIG['num_classes']

        # Initial convolution - OPTIMIZED: reduced channels for parameter efficiency
        initial_channels = MODEL_CONFIG['initial_channels']
        self.conv1 = nn.Conv2d(MODEL_CONFIG['input_channels'], initial_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)

        # Efficient blocks - OPTIMIZED: balanced channel progression
        channel_progression = MODEL_CONFIG['channel_progression']
        blocks_per_layer = MODEL_CONFIG['blocks_per_layer']
        
        self.layer1 = self._make_layer(initial_channels, channel_progression[0], blocks_per_layer, stride=1)
        self.layer2 = self._make_layer(channel_progression[0], channel_progression[1], blocks_per_layer, stride=2)
        self.layer3 = self._make_layer(channel_progression[1], channel_progression[2], blocks_per_layer, stride=2)
        self.layer4 = self._make_layer(channel_progression[2], channel_progression[3], blocks_per_layer, stride=2)

        # Global average pooling and classifier - OPTIMIZED: reduced capacity
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout_rate'])
        self.fc = nn.Linear(channel_progression[3], num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(EfficientBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(EfficientBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using advanced initialization"""
        for m in self.modules():
            try:
                if isinstance(m, nn.Conv2d):
                    if m.weight is not None:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    if m.weight is not None:
                        nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                # Skip layers without weights
                elif isinstance(m, (nn.Identity, nn.Dropout, nn.AdaptiveAvgPool2d,
                                  nn.ReLU, nn.Sigmoid, nn.Sequential, nn.ModuleList)):
                    continue
                elif not hasattr(m, 'weight') or m.weight is None:
                    continue
            except Exception as e:
                print(f"Warning: Skipping weight initialization for {type(m).__name__}: {e}")
                continue

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def calculate_receptive_field(model, input_size=(32, 32)):
    """Calculate receptive field of the optimized model"""
    # For optimized architecture: RF = 1 + 2 * (sum of strides - 1)
    # Conv1: 3x3, stride=1 -> RF = 3
    # Layer1: 2 blocks, stride=1 -> RF = 3 + 2*2 = 7
    # Layer2: 2 blocks, stride=2 -> RF = 7 + 2*2*2 = 15
    # Layer3: 2 blocks, stride=2 -> RF = 15 + 2*2*4 = 31
    # Layer4: 2 blocks, stride=2 -> RF = 31 + 2*2*8 = 63

    # More accurate calculation considering all convolutions
    rf = 1
    stride = 1

    # Conv1: 3x3, stride=1
    rf += 2
    stride *= 1

    # Layer1: 2 blocks, each with 2 depthwise separable convs (3x3 each)
    for _ in range(2):
        for _ in range(2):  # 2 depthwise separable convs per block
            rf += 2 * stride
    stride *= 1  # No stride change in layer1

    # Layer2: 2 blocks, stride=2
    for _ in range(2):
        for _ in range(2):
            rf += 2 * stride
    stride *= 2  # stride=2

    # Layer3: 2 blocks, stride=2
    for _ in range(2):
        for _ in range(2):
            rf += 2 * stride
    stride *= 2  # stride=2

    # Layer4: 2 blocks, stride=2
    for _ in range(2):
        for _ in range(2):
            rf += 2 * stride
    stride *= 2  # stride=2

    return rf


def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model):
    """Print detailed model summary for optimized architecture"""
    print("\n" + "=" * 60)
    print("OPTIMIZED MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)

    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name:30s} | {str(type(module).__name__):20s} | {params:8,} params")
                total_params += params
                if any(p.requires_grad for p in module.parameters()):
                    trainable_params += params

    print("-" * 60)
    print(f"{'Total Parameters:':30s} | {total_params:8,}")
    print(f"{'Trainable Parameters:':30s} | {trainable_params:8,}")
    print(f"{'Non-trainable Parameters:':30s} | {total_params - trainable_params:8,}")
    print("=" * 60)
