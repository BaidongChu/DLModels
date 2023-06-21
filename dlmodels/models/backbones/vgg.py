import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, batchnorm=False):
        super().__init__()
        module_list = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]

        if batchnorm:
            module_list.append(nn.BatchNorm2d(out_channels))
        module_list.append(nn.ReLU(True))
        for _ in range(layer_num - 1):
            module_list.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            if batchnorm:
                module_list.append(nn.BatchNorm2d(out_channels))
            module_list.append(nn.ReLU(True))
        module_list.append(nn.MaxPool2d(2, 2))

        self.layers = nn.Sequential(*module_list)
    
    def forward(self, x):
        return self.layers(x)
        


class VGG(nn.Module):
    def __init__(self, layer_nums, batchnorm=False):
        super().__init__()

        self.block1 = BasicBlock(layer_nums[0], 3, 64, batchnorm)
        self.block2 = BasicBlock(layer_nums[1], 64, 128, batchnorm)
        self.block3 = BasicBlock(layer_nums[2], 128, 256, batchnorm)
        self.block4 = BasicBlock(layer_nums[3], 256, 512, batchnorm)
        self.block5 = BasicBlock(layer_nums[4], 512, 512, batchnorm)

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)

        return self.block5(output)
                              
        