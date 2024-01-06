from torch import nn as nn
from models.deeplab_v3 import DeepLabV3Plus

class DeepLabBB(nn.Module):
    def __init__(
        self,
        pyramid_channels=256,
        deeplab_ch=256,
        backbone='resnet34',
        backbone_lr_mult=0.1,
    ):
        super(DeepLabBB, self).__init__()
        self.pyramid_on = pyramid_channels > 0
        if self.pyramid_on:
            self.output_channels = [pyramid_channels] * 4
        else:
            self.output_channels = [deeplab_ch] #[256]

        self.deeplab = DeepLabV3Plus(backbone=backbone,
                                     ch=deeplab_ch,
                                     project_dropout=0.2,
                                     norm_layer=nn.BatchNorm2d,
                                     backbone_norm_layer=nn.BatchNorm2d)
        self.deeplab.backbone.apply(LRMult(backbone_lr_mult))

    def forward(self, image, mask, mask_features):
        outputs = list(self.deeplab(image, mask_features))
        return outputs

    def load_pretrained_weights(self):
        self.deeplab.load_pretrained_weights()

class LRMult(object):
    def __init__(self, lr_mult=1.):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult
