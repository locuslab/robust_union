'''Halfspace Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from halfspace import Halfspace, FilterHalfspace

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, size=None, k=1, kernel_size=1, padding=0):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.hs1 = nn.Sequential(*[Halfspace(size, size, k=1) for _ in range(k)])
        #nhs = int(size*size*k)
        nhs = int(in_planes*k)
        #print(nhs, "# halfspaces")
        # self.hs2 = nn.Sequential(*[FilterHalfspace(in_planes) for _ in range(nhs)])
        self.hs1 = nn.Sequential(*[FilterHalfspace(in_planes, bias=False, kernel_size=kernel_size, padding=padding) if _ == 0 else FilterHalfspace(in_planes, bias=True, kernel_size=kernel_size, padding=padding) for _ in range(nhs)])
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.hs2 = nn.Sequential(*[Halfspace(size//stride, size//stride, k=1) for _ in range(k)])
        # nhs = int((size//stride)*(size//stride)*k)
        # self.hs2 = nn.Sequential(*[FilterHalfspace(planes) for _ in range(nhs)])
        nhs = int(planes*k)
        #print(nhs, "# halfspaces")
        self.hs2 = nn.Sequential(*[FilterHalfspace(planes, bias=False, kernel_size=kernel_size, padding=padding) if _ == 0 else FilterHalfspace(planes, bias=True, kernel_size=kernel_size, padding=padding) for _ in range(nhs)])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.hs1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.hs2(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class HalfspaceResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, k=1, kernel_size=1, padding=0):
        super(HalfspaceResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, size=32, k=k, kernel_size=kernel_size, padding=padding)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, size=32, k=k, kernel_size=kernel_size, padding=padding)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, size=16, k=k, kernel_size=kernel_size, padding=padding)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, size=8, k=k, kernel_size=kernel_size, padding=padding)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, size, k, kernel_size, padding):
        strides = [stride] + [1]*(num_blocks-1)
        sizes = [size] + [size//stride]*(num_blocks-1)
        layers = []
        for size,stride in zip(sizes,strides):
            layers.append(block(self.in_planes, planes, stride, size, k, kernel_size, padding))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def HalfspaceResNet18(**kwargs):
    return HalfspaceResNet(PreActBlock, [2,2,2,2], **kwargs)

# def PreActResNet34():
#     return PreActResNet(PreActBlock, [3,4,6,3])

# def PreActResNet50():
#     return PreActResNet(PreActBottleneck, [3,4,6,3])

# def PreActResNet101():
#     return PreActResNet(PreActBottleneck, [3,4,23,3])

# def PreActResNet152():
#     return PreActResNet(PreActBottleneck, [3,8,36,3])


# def test():
#     net = HalfspaceResNet18(kernel_size=1)
#     y = net((torch.randn(1,3,32,32)))
#     print(y, y.size())

# test()
