"""RC2 MinkNet backbone + classifier (DA / loss modules removed)."""

import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

from .blocks import (
    BLOCK_MAP,
    BasicConvolutionBlock,
    BasicDeconvolutionBlock,
    BatchNorm,
    SyncBatchNorm,
)


class MinkNet(nn.Module):
    def __init__(self, model_cfgs, num_class: int):
        super().__init__()
        self.model_cfgs = model_cfgs
        self.num_class = num_class
        self.in_feature_dim = model_cfgs.IN_FEATURE_DIM
        self.num_layer = model_cfgs.get('NUM_LAYER', [2, 3, 4, 6, 2, 2, 2, 2])
        self.block = BLOCK_MAP[model_cfgs.get('BLOCK', 'ResBlock')]

        cr = model_cfgs.get('cr', 1.0)
        cs = model_cfgs.get('PLANES', [32, 32, 64, 128, 256, 256, 128, 96, 96])
        cs = [int(cr * x) for x in cs]
        if_dist = model_cfgs.get('IF_DIST', False)

        self.stem = nn.Sequential(
            spnn.Conv3d(self.in_feature_dim, cs[0], kernel_size=3, stride=1),
            SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.in_channels = cs[0]
        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1, if_dist=if_dist),
            *self._make_layer(self.block, cs[1], self.num_layer[0], if_dist=if_dist),
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1, if_dist=if_dist),
            *self._make_layer(self.block, cs[2], self.num_layer[1], if_dist=if_dist),
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1, if_dist=if_dist),
            *self._make_layer(self.block, cs[3], self.num_layer[2], if_dist=if_dist),
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1, if_dist=if_dist),
            *self._make_layer(self.block, cs[4], self.num_layer[3], if_dist=if_dist),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(self.in_channels, cs[5], ks=2, stride=2, if_dist=if_dist),
        ])
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(nn.Sequential(*self._make_layer(self.block, cs[5], self.num_layer[4], if_dist=if_dist)))

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(self.in_channels, cs[6], ks=2, stride=2, if_dist=if_dist),
        ])
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(nn.Sequential(*self._make_layer(self.block, cs[6], self.num_layer[5], if_dist=if_dist)))

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(self.in_channels, cs[7], ks=2, stride=2, if_dist=if_dist),
        ])
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(nn.Sequential(*self._make_layer(self.block, cs[7], self.num_layer[6], if_dist=if_dist)))

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(self.in_channels, cs[8], ks=2, stride=2, if_dist=if_dist),
        ])
        self.in_channels = cs[8] + cs[0]
        self.up4.append(nn.Sequential(*self._make_layer(self.block, cs[8], self.num_layer[7], if_dist=if_dist)))

        self.dropout = nn.Dropout(model_cfgs.get('DROPOUT_P', 0.0), True)
        self.cs = cs
        self._weight_initialization()

    def _make_layer(self, block, out_channels, num_block, stride=1, if_dist=False):
        layers = [block(self.in_channels, out_channels, stride=stride, if_dist=if_dist)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels, if_dist=if_dist))
        return layers

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        x = batch_dict['lidar']
        x.F = x.F[:, :self.in_feature_dim]

        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x4.F = self.dropout(x4.F)
        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y2.F = self.dropout(y2.F)
        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        return {'out_feature': y4.F}


class MinkClassifier(nn.Module):
    def __init__(self, model_cfgs, num_class: int):
        super().__init__()
        cr = model_cfgs.get('cr', 1.0)
        cs = model_cfgs.get('PLANES', [32, 32, 64, 128, 256, 256, 128, 96, 96])
        cs = [int(cr * x) for x in cs]
        self.classifier = nn.Sequential(nn.Linear(cs[8], num_class))

    def forward(self, x):
        return self.classifier(x)
