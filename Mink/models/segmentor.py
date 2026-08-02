"""Unified segmentor factory compatible with base_agent's MinkUNet I/O."""

from types import SimpleNamespace

import torch.nn as nn

from Mink.models.minkunet import MinkUNet
from Mink.models.rc2.minknet import MinkNet, MinkClassifier
from Mink.models.rc2.spvcnn import SPVCNN, SPVCNNClassifier


class _CfgView:
    """Attribute + dict.get access over a config class / namespace."""

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, item):
        if item.startswith('_'):
            return super().__getattribute__(item)
        return getattr(self._cfg, item)

    def get(self, key, default=None):
        return getattr(self._cfg, key, default)


class RC2Segmentor(nn.Module):
    """Wrap RC2 backbone + classifier to match MinkUNet outputs."""

    def __init__(self, cfg, num_classes, model_name):
        super().__init__()
        model_cfgs = _CfgView(cfg)
        self.model_name = model_name
        if model_name == 'MinkNet':
            self.backbone = MinkNet(model_cfgs, num_classes)
            self.classifier = MinkClassifier(model_cfgs, num_classes)
        elif model_name == 'SPVCNN':
            self.backbone = SPVCNN(model_cfgs, num_classes)
            self.classifier = SPVCNNClassifier(model_cfgs, num_classes)
        else:
            raise ValueError(f'Unsupported MODEL_NAME={model_name}')

    def forward(self, x):
        ret = self.backbone({'lidar': x})
        feat = ret['out_feature']
        logits = self.classifier(feat)
        return {
            'final': logits,
            'pt_feat': SimpleNamespace(F=feat),
        }


def build_segmentor(cfg):
    """
    Build a segmentor for outdoor RC2 models or the original indoor MinkUNet.
    Outdoor: MODEL_NAME in {MinkNet, SPVCNN}
    Indoor fallback: MinkUNet
    """
    model_name = getattr(cfg, 'MODEL_NAME', 'MinkUNet')
    num_classes = cfg.num_classes
    input_channel = cfg.input_channel

    if model_name in ('MinkNet', 'SPVCNN'):
        return RC2Segmentor(cfg, num_classes, model_name)

    return MinkUNet(num_classes=num_classes, cr=1.0, input_channel=input_channel)
