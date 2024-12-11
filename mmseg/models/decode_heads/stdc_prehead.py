# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from mmseg.models.losses.Center_Loss import *


@MODELS.register_module()
class STDCpreHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            
        self.binary_conv = ConvModule(
            in_channels=32,               # 与output的通道数要对应
            out_channels=1,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        



        '''思路1：自定义卷积层'''
        channels1 = [32, 64, 160, 256]   # 用于segformer
        channels2 = [32, 32, 64, 160]
        # channels1 = [64, 128, 320, 512]   # 用于twins
        # channels2 = [64, 64, 128, 320]
        self.convs_enhance = nn.ModuleList()
        for i in range(4):
            self.convs_enhance.append(
                ConvModule(
                    in_channels=channels1[i],
                    out_channels=channels2[i],
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.fu_conv = nn.ModuleList()
        for i in range(4-1):
            self.fu_conv.append(
                ConvModule(
                    in_channels=channels1[i],
                    out_channels=channels1[i],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg))




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
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):   # 删掉batch_data_samples
        """Forward function."""
        output = inputs[0]       # 与初始化self.binary_conv的通道数要匹配



        '''思路1:特征增强模块（陈欢）'''  ### 重新计算一遍edge，肯定跟原始head的不同，最好是能从原始head传递出来
        outs = []
        outs_enhance = []               # 存储增强后的特征
        outs_enhance.append(inputs[3])  # 保留最底层特征
        aerfa = 0.5
        beta = 0.5
        for idx in range(len(inputs) - 1):
            idx = 3 - idx        # [3,2,1]
            x1 = inputs[idx]     # (16,16,256)
            x2 = inputs[idx - 1] # (32,32,160)
            conv = self.convs_enhance[idx]  # 256→160，（16,16,160）
            x1 = resize(input=conv(x1), size=x2.shape[2:], 
                        mode="bilinear", align_corners=self.align_corners)  # 先改变通道数，再resize
            edge = self.fu_conv[idx-1](x2 + x1)             # 可替换融合方法：cat、×、+、-             注意是减法！！！！！
            if idx == 3:
                outs.append(x1)
            feature = self.fu_conv[idx-1](beta * edge + outs[3 - idx])
            if idx == 3:
                feature_enhance = aerfa * edge * feature
                outs_enhance.append(feature_enhance)
                outs.append(resize(input=self.convs_enhance[idx - 1](feature_enhance), size=[64,64], 
                                   mode="bilinear", align_corners=self.align_corners))
            elif idx == 2:
                feature_enhance = 1 * edge * feature
                outs_enhance.append(feature_enhance)
                outs.append(resize(input=self.convs_enhance[idx - 1](feature_enhance), size=[128,128], 
                                   mode="bilinear", align_corners=self.align_corners))
            else:
                feature_enhance = 1 * edge * feature
                outs_enhance.append(feature_enhance)
                outs.append(feature_enhance)
        outs_enhance = outs_enhance[::-1]
        # output = edge    # 只替换（4,160,32,32）或（4,32,128,128）的增强特征
        output = outs_enhance[0]




        out = self.binary_conv(output)   # 改变通道数到1，与gt的通道数一致
        # out = torch.mean(output, dim=1, keepdim=True)   # 平均池化  

        return out