# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import Upsample, resize
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class FPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        
        '''自定义卷积层'''
        channels1 = [256, 256, 256]
        channels2 = [256, 256, 256]
        self.fu_conv = nn.ModuleList()
        for i in range(3):
            self.fu_conv.append(
                ConvModule(
                    in_channels=channels1[i],
                    out_channels=channels1[i],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg))
        self.fu_conv_enhance = nn.ModuleList()
        for i in range(3):
            self.fu_conv_enhance.append(
                ConvModule(
                    in_channels=channels1[i],
                    out_channels=channels2[i],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg))

    def forward(self, inputs):

        x = self._transform_inputs(inputs)



        edge = x
        for i in range(3, 0, -1):
            prev_shape = x[i-1].shape[2:]
            edge[i - 1] = x[i - 1] + resize(
                x[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        '''思路1:特征增强模块'''
        aerfa = 0.5
        beta = 0.5
        outs_enhance = []               # 存储增强后的特征
        outs_enhance.append(x[3])
        for i in range(3, 0, -1):
            feature_shape = edge[i - 1].shape[2:]
            feature = beta * edge[i - 1] + resize(
                    outs_enhance[3 - i],        # 加增强后的特征outs_enhance
                    #edge[i],                     # 加邻近层的edge
                    size=feature_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)
            feature = self.fu_conv[i-1](feature)
            feature_enhance = feature * (aerfa * edge[i - 1])
            feature_enhance = self.fu_conv_enhance[i-1](feature_enhance)
            outs_enhance.append(feature_enhance)
        outs_enhance = outs_enhance[::-1]        # 调换顺序，和inputs的索引保持一致

        # '''替换增强后的特征层'''
        x = outs_enhance





        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)
        return output
