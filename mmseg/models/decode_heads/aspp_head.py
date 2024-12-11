# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@MODELS.register_module()
class ASPPHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super().__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        

        '''思路1：自定义卷积层'''
        # channels1 = [256, 512, 1024, 2048]
        # channels2 = [256, 256, 512, 1024]
        # self.convs_enhance = nn.ModuleList()
        # for i in range(4):
        #     self.convs_enhance.append(
        #         ConvModule(
        #             in_channels=channels1[i],
        #             out_channels=channels2[i],
        #             kernel_size=1,
        #             stride=1,
        #             norm_cfg=self.norm_cfg,
        #             act_cfg=self.act_cfg))
        # self.fu_conv = nn.ModuleList()
        # for i in range(3):
        #     self.fu_conv.append(
        #         ConvModule(
        #             in_channels=channels1[i],
        #             out_channels=channels1[i],
        #             kernel_size=1,
        #             norm_cfg=self.norm_cfg))
            



    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""


        # '''特征增强模块'''
        # outs = []
        # outs_enhance = []               # 存储增强后的特征
        # outs_enhance.append(inputs[3])  # 保留最底层特征
        # aerfa = 0.5
        # beta = 0.5
        # for idx in range(len(inputs) - 1):
        #     idx = 3 - idx        # [3,2,1]
        #     x1 = inputs[idx]     # (16,16,256)
        #     x2 = inputs[idx - 1] # (32,32,160)
        #     conv = self.convs_enhance[idx]  # 256→160，（16,16,160）
        #     x1 = resize(input=conv(x1), size=x2.shape[2:], 
        #                 mode='bilinear', align_corners=self.align_corners)  # 先改变通道数，再resize
        #     edge = self.fu_conv[idx-1](x2 + x1)             # 可替换融合方法：cat、×、+、-
        #     if idx == 3:
        #         outs.append(x1)
        #     feature = self.fu_conv[idx-1](beta * edge + outs[3 - idx])
        #     if idx == 3:
        #         feature_enhance = aerfa * edge * feature          #########经过运算后，未进行卷积、BN、ReLU操作！！！########
        #         outs_enhance.append(feature_enhance)
        #         outs.append(resize(input=self.convs_enhance[idx - 1](feature_enhance), size=[64,64], 
        #                            mode='bilinear', align_corners=self.align_corners))
        #     elif idx == 2:
        #         feature_enhance = aerfa * edge * feature
        #         outs_enhance.append(feature_enhance)
        #         outs.append(resize(input=self.convs_enhance[idx - 1](feature_enhance), size=[128,128], 
        #                            mode='bilinear', align_corners=self.align_corners))
        #     else:
        #         feature_enhance = aerfa * edge * feature
        #         outs_enhance.append(feature_enhance)
        #         outs.append(feature_enhance)
        # outs_enhance = outs_enhance[::-1]
        # inputs = outs_enhance



        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
