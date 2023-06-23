import sys
from collections import OrderedDict
import torch.nn as nn

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kerner_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kerner_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv_bn1 = ConvBn(in_channels, out_channels, 3, stride, 1)
        self.conv_bn2 = ConvBn(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels or stride != 1:
            self.down_sample = ConvBn(in_channels, out_channels, 1, stride, 0)
        else:
            self.down_sample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        identity = self.down_sample(x)
        return self.relu(out + identity)
    
class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv_bn1 = ConvBn(in_channels, out_channels, 1, 1, 0)
        self.conv_bn2 = ConvBn(out_channels, out_channels, 3, stride, 1)
        self.conv_bn3 = ConvBn(out_channels, out_channels*4, 1, 1, 0)
        self.down_sample = ConvBn(in_channels, out_channels*4, 1, stride, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv_bn1(x))
        out = self.relu(self.conv_bn2(out))
        out = self.conv_bn3(out)
        identity = self.down_sample(x)
        return self.relu(out + identity)
    
class ResNetLayer(nn.Module):
    def __init__(self, block_num, in_channels, out_channels, stride, use_bottle_neck=True):
        super().__init__()
        block_name = "BottleNeckBlock" if use_bottle_neck else "BasicBlock"
        block = getattr(sys.modules[__name__], block_name)
        blocks = [block(in_channels, out_channels, stride)]
        in_channels = out_channels*4 if use_bottle_neck else out_channels
        for _ in range(block_num - 1):
            blocks.append(block(in_channels, out_channels, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
    
class ResNet(nn.Module):
    def __init__(self, block_nums, use_bottleneck):
        super().__init__()
        expansion = 4 if use_bottleneck else 1
        self.layer1 = nn.Sequential(
            ConvBn(3, 64, 7, 2, 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            ResNetLayer(block_nums[0], 64, 64, 1, use_bottleneck)
        )
        self.layer3 = ResNetLayer(block_nums[1], 64*expansion, 128, 2, use_bottleneck)
        self.layer4 = ResNetLayer(block_nums[2], 128*expansion, 256, 2, use_bottleneck)
        self.layer5 = ResNetLayer(block_nums[3], 256*expansion, 512, 2, use_bottleneck)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

# if __name__ == "__main__":

#     model = ResNet([3, 4, 6, 3], True).cuda()
#     print(model)
#     import torch
#     input = torch.rand(1, 3, 224, 224).cuda()
#     output = model(input)
#     pass