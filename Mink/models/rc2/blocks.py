import torch.nn as nn
import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply


class SyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, if_dist=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, if_dist=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, if_dist=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc * self.expansion, kernel_size=1, dilation=1, stride=stride),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, if_dist=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=1, bias=False),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(outc, outc, kernel_size=ks, stride=stride, bias=False, dilation=dilation),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(outc, outc * self.expansion, kernel_size=1, bias=False),
            SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc * self.expansion, kernel_size=1, dilation=1, stride=stride),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


BLOCK_MAP = {
    'ResBlock': ResidualBlock,
    'Bottleneck': Bottleneck,
}
