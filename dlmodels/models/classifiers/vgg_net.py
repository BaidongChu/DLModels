import torch
import torch.nn as nn

from ..backbones.vgg import VGG


class VGGNet(nn.Module):
    def __init__(self, configs, init_weights=True):
        super().__init__()

        self.backbone = VGG(configs['model']['layer_num'], configs['model']['batchnorm'])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, configs['model']['num_classes'])
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