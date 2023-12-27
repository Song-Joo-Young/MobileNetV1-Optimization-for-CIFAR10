'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    # with residual connection
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # 입력과 출력의 차원이 다른 경우 Residual connection을 위한 1x1 컨볼루션
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.shortcut is not None:
            x = self.shortcut(x)
        out = out + x  # Residual connection 추가
        return out


class CustomMobileNet(nn.Module):
    cfg = [32, 32, 32, (64,2), 64, 64, 64, (128,2), 128, 128, 128, 128, 128, 128, 128, 128, (256,2), 256, 256, 256, (512,2), 512]

    def __init__(self, num_classes=10):
        super(CustomMobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_planes=16)
        self.linear = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 추가
        # 가중치 초기화 함수 호출
        self._initialize_weights()

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)

        # 드롭아웃 추가
        out = self.dropout(out)
        out = self.linear(out)
        return out

    # 가중치 초기화 추가
    # from: https://github.com/2KangHo/mobilenet_cifar10_pytorch/blob/master/mobilenet.py
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 초기화
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 평균이 0이고 표준편차가 sqrt(2/n)인 정규 분포로 초기화
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # 배치 정규화 레이어의 가중치를 1로 초기화
                m.weight.data.fill_(1)
                # 배치 정규화 레이어의 편향을 0으로 초기화
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                # FC 레이어의 가중치를 평균이 0이고 표준편차가 0.01인 정규 분포로 초기화
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def test():
    net = CustomMobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
