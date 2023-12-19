import torch
import torch.nn as nn
from .model_config import ResNetConfig
from ...base_model import BaseModel


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
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

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
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

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


class ResNet(BaseModel):
    Config = ResNetConfig

    def __init__(self, config: ResNetConfig):
        super(ResNet, self).__init()
        self.config = config
        self.custom = config.custom
        self.in_planes = 64
        self.build_resnet()

    def build_resnet(self):
        if self.custom:
            blocks, num_layers = self.custom

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Use two 3x3 kernels for ResNet18 and ResNet34.
        if num_layers < 50:
            self.layer1 = self.make_layer(
                BasicBlock, 64, num_blocks=blocks[0], stride=1
            )
            self.layer2 = self.make_layer(
                BasicBlock, 128, num_blocks=blocks[1], stride=2
            )
            self.layer3 = self.make_layer(
                BasicBlock, 256, num_blocks=blocks[2], stride=2
            )
            self.layer4 = self.make_layer(
                BasicBlock, 512, num_blocks=blocks[3], stride=2
            )
        # Use 1x1, 3x3 1x1 kernels for ResNet50, ResNet101, and ResNet152.
        else:
            self.layer1 = self.make_layer(
                Bottleneck, 64, num_blocks=blocks[0], stride=1
            )
            self.layer2 = self.make_layer(
                Bottleneck, 128, num_blocks=blocks[1], stride=2
            )
            self.layer3 = self.make_layer(
                Bottleneck, 256, num_blocks=blocks[2], stride=2
            )
            self.layer4 = self.make_layer(
                Bottleneck, 512, num_blocks=blocks[3], stride=2
            )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, self.config.num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
