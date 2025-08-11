"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..tools.custom_modules import SequentialWithArgs, FakeReLU
from ..utils import conv_spectral_norm as spectral_norm


def get_spectral_norm(module: nn.Module) -> None | Any:
    if hasattr(module, "weight_orig"):
        w = module.weight_orig
        w_reshaped = w.view(w.shape[0], -1)
        return torch.linalg.svdvals(w_reshaped)[0]
    return None


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class BasicBlockSpectral(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockSpectral, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class BottleneckSpectral(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckSpectral, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = spectral_norm(nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False))
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNet, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale * widths[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert not no_relu, "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final


class ResNetSpectral(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNetSpectral, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = spectral_norm(nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = spectral_norm(nn.Linear(feat_scale * widths[3] * block.expansion, num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert not no_relu, "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], wm=5, **kwargs)


def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], wd=0.75, **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


# Spectral modes
def SpectralResNet18(**kwargs):
    return ResNetSpectral(BasicBlockSpectral, [2, 2, 2, 2], **kwargs)


def SpectralResNet18Wide(**kwargs):
    return ResNetSpectral(BasicBlockSpectral, [2, 2, 2, 2], wm=5, **kwargs)


def SpectralResNet18Thin(**kwargs):
    return ResNetSpectral(BasicBlockSpectral, [2, 2, 2, 2], wd=0.75, **kwargs)


def SpectralResNet34(**kwargs):
    return ResNetSpectral(BasicBlockSpectral, [3, 4, 6, 3], **kwargs)


def SpectralResNet50(**kwargs):
    return ResNetSpectral(BottleneckSpectral, [3, 4, 6, 3], **kwargs)


def SpectralResNet101(**kwargs):
    return ResNetSpectral(BottleneckSpectral, [3, 4, 23, 3], **kwargs)


def SpectralResNet152(**kwargs):
    return ResNetSpectral(BottleneckSpectral, [3, 8, 36, 3], **kwargs)


resnet50 = ResNet50
resnet18 = ResNet18
resnet34 = ResNet34
resnet101 = ResNet101
resnet152 = ResNet152
resnet18wide = ResNet18Wide

spectral_resnet50 = SpectralResNet50
spectral_resnet18 = SpectralResNet18
spectral_resnet34 = SpectralResNet34
spectral_resnet101 = SpectralResNet101
spectral_resnet152 = SpectralResNet152
spectral_resnet18wide = SpectralResNet18Wide


# resnet18thin = ResNet18Thin
def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
