import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    A custom layer that reduces parameter count 80% compared to some standard Conv2d.
    Important for keeping the final submission file size minimum: 
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 1. Depthwise Convolution: applies a single filter to each input channel
        # Setting groups=in_channels is what makes it depthwise
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 2. Pointwise Convolution: 1x1 conv to combine the depthwise outputs
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class TinyEncoder(nn.Module):
    """
    The core feature extractor. This learns the shapes and textures of images.
    Input: 3 channels (RGB), 96x96 pixels.
    """
    def __init__(self):
        super(TinyEncoder, self).__init__()
        
        # Initial standard convolution to extract base features
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # outputs 48x48
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Stack of Depthwise Separable Convolutions
        self.features = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),   # outputs 24x24
            DepthwiseSeparableConv(32, 64, stride=2),   # outputs 12x12
            DepthwiseSeparableConv(64, 128, stride=2),  # outputs 6x6
            DepthwiseSeparableConv(128, 128, stride=1)  # keeps 6x6
        )
        
        # Squashes the 6x6 spatial dimensions into a single 1D vector per image
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.init_conv(x)
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1) # Output shape: (batch_size, 128)


class SSLPretrainModel(nn.Module):
    """
    Used ONLY during Phase 1: Training on the 100,000 unlabeled images.
    Combines the encoder with a projection head for Contrastive Learning.
    """
    def __init__(self, encoder):
        super(SSLPretrainModel, self).__init__()
        self.encoder = encoder
        
        # Projects the 128-dim features into a lower dimension for the SSL loss function
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        features = self.encoder(x)
        embeddings = self.projector(features)
        # Normalize embeddings so they sit on a unit sphere (required for cosine similarity)
        return F.normalize(embeddings, dim=1)


class FinalClassifierModel(nn.Module):
    """
    Used ONLY during Phase 2: Fine-tuning on the 5,000 labeled images.
    This is the exact architecture you will submit for the competition.
    """
    def __init__(self, encoder, num_classes=10):
        super(FinalClassifierModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
