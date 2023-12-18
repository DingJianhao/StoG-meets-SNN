# import sys
# sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *

class BasicBlock(nn.Module):
    def __init__(self, T, in_planes, out_planes, stride, dropRate=0.3, tau=1.0):
        super(BasicBlock, self).__init__()
        self.T = T
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = LIFSpike(T, tau=tau)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act2 = LIFSpike(T, tau=tau)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.convex = ConvexCombination(2)
    def forward(self, x):
        if not self.equalInOut:
            x = self.act1(self.bn1(x))
        else:
            out = self.act1(self.bn1(x))
        out = self.act2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return self.convex(x if self.equalInOut else self.convShortcut(x), out)

class WideResNet(nn.Module):
    def __init__(self, name, T, num_classes, norm, dropRate=0.0, init_c=3, tau=1.0, H=32):
        super(WideResNet, self).__init__()
        self.H = H
        if "16" in name:
            depth = 16
            widen_factor = 4
        elif "20" in name:
            depth = 28
            widen_factor = 10
        else:
            raise AssertionError("Invalid wide-resnet name: " + name)
        
        if init_c == 2: # event data
            self.rescale = nn.AdaptiveAvgPool2d((48, 48))

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            raise AssertionError("Invalid normalization")

        block = BasicBlock
        self.T = T
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)
        self.tau = tau

        # 1st conv before any network block
        if self.H == 32:
            self.conv1 = nn.Conv2d(init_c, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        elif self.H == 64:
            self.conv1 = nn.Conv2d(init_c, nChannels[0], kernel_size=3, stride=2, padding=1, bias=False)

        self.block1 = self._make_layer(block, nChannels[0], nChannels[1], n, 1, dropRate, tau)
        self.block2 = self._make_layer(block, nChannels[1], nChannels[2], n, 2, dropRate, tau)
        self.block3 = self._make_layer(block, nChannels[2], nChannels[3], n, 2, dropRate, tau)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.act = LIFSpike(T, tau=self.tau)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, tau):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(self.T, i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, tau))
        return nn.Sequential(*layers)

    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, mode='bptt', *args, **kwargs):
        # self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                # module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return
    
    def forward(self, input):
        if len(input.shape) == 4:
            input = self.norm(input)
            if self.T > 0:
                input = add_dimention(input, self.T)
                input = self.merge(input)
        else:
            input = input.permute((1, 0, 2, 3, 4))
            input = self.merge(input)
            input = self.rescale(input)
        out = self.conv1(input)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.act(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.T > 0:
            out = self.expand(out)
        return self.fc(out)

# if __name__ == "__main__":
#     N = WideResNet(name='16', T=8, num_classes=10, norm=((0.480, 0.448, 0.398), (0.272, 0.266, 0.274)), dropRate=0.0, init_c=3, tau=1.0)
#     print(N(torch.rand(7, 3, 64, 64)).shape)