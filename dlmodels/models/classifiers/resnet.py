import torch
import torch.nn as nn

from ..backbones.resnet import ResNet as ResNetBackbone

class ResNet(nn.Module):
    def __init__(self, init_weights=True, **kwargs):
        super().__init__()
        self.backbone = ResNetBackbone(kwargs["block_nums"], kwargs["use_bottleneck"])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        out_channels = 2048 if kwargs["use_bottleneck"] else 512
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(7*7*out_channels, 1000),
            nn.ReLU(),
            nn.Linear(1000, kwargs['num_classes'])
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        out = self.classifier(features)
        return out

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 1)