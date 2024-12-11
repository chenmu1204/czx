# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM

import math
from mmcv.cnn import Linear, build_activation_layer
from mmengine.model import BaseModule


@MODELS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        

        '''自定义卷积层'''
        # channels1 = [768, 768, 768]   # BEiT、DPT
        # channels2 = [768, 768, 768]
        channels1 = [512, 512, 512]   # Twins、Swin
        channels2 = [512, 512, 512]
        # channels1 = [384, 384, 384]   # Segmentor
        # channels2 = [384, 384, 384]
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
        

        # '''DPT算法的自定义模块，还需删除250行：class ReassembleBlocks(BaseModule)'''
        # self.convs = nn.ModuleList()
        # for channel in [96, 192, 384, 768]:
        #     self.convs.append(
        #         ConvModule(
        #             channel,
        #             self.channels,
        #             kernel_size=3,
        #             padding=1,
        #             act_cfg=None,
        #             bias=False))
        # self.reassemble_blocks = ReassembleBlocks(in_channels=768,
        #                                           out_channels=[96, 192, 384, 768],
        #                                           readout_type='ignore')
            



    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)
        # build laterals          前3层的64,128,320通道全部转成，和第4层一样的通道数512
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))
        


        # '''DPT算法的自定义模块，还需删除250行：class ReassembleBlocks(BaseModule)'''
        # laterals = self._transform_inputs(inputs)
        # laterals = self.reassemble_blocks(laterals)
        # laterals = [self.convs[i](feature) for i, feature in enumerate(laterals)]   # 4个层次的特征图
        


        origine = laterals
        # build top-down path     临近层加法融合：x2 + resize（x1）
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(          # 改成减法，表征边缘
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs     卷积Conv2d(512, 512, kernel_size=(3, 3)、BN、ReLU
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])
            



        '''边缘特征增强模块'''
        edge = fpn_outs                 # 加法融合后的laterals，经过了卷积+BN+ReLU即为我的edge
        aerfa = 0.3
        beta = 0.5
        outs_enhance = []               # 存储增强后的特征
        outs_enhance.append(origine[3])
        for i in range(used_backbone_levels - 1, 0, -1):
            feature_shape = edge[i - 1].shape[2:]
            feature = beta * edge[i - 1] + resize(
                    #outs_enhance[3 - i],        # 加增强后的特征outs_enhance
                    edge[i],                     # 加邻近层的edge
                    size=feature_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)
            feature = self.fu_conv[i-1](feature)
            feature_enhance = feature * (aerfa * edge[i - 1])
            feature_enhance = self.fu_conv_enhance[i-1](feature_enhance)
            outs_enhance.append(feature_enhance)
        outs_enhance = outs_enhance[::-1]        # 调换顺序，和inputs的索引保持一致
        fpn_outs = outs_enhance


        # '''可视化边缘图'''
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import os
        # from PIL import Image
        # # 设置保存图像的文件夹路径
        # image_names = []
        # for seg_data_sample in batch_data_samples:         ### decode_head.py的第274行，添加传参batch_data_samples，涉及到的所有head都要加 ###
        #     img_path = seg_data_sample.img_path
        #     image_name = os.path.basename(img_path)        # 提取文件名
        #     image_name = os.path.splitext(image_name)[0]   # 去除后缀
        #     image_names.append(image_name)
        # save_folder = "/data3/chenzhenxiang/work_dirs/test/image/"
        # for i, seg_data_sample in enumerate(edge):                     # （4,32,12,8128）
        #     image_data = torch.mean(seg_data_sample, dim=0, keepdim=True)   # 平均池化  
        #     image_data = np.squeeze(image_data).data.cpu().numpy()          # 将数据移动到CPU并转换为numpy数组
        #     plt.figure()
        #     plt.imshow(image_data)  # 使用viridis颜色图，可以根据需要更改, cmap='gray'
        #     save_path = f"{save_folder}{image_names[i]}.png"
        #     plt.savefig(save_path)
        #     plt.close()


        

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],       # 4层统一resize到（4,512,128,128）
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)     # （4,2048,128,128）
        feats = self.fpn_bottleneck(fpn_outs)     # （4,512, 128,128）
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)             # （4,4,128,128）
        return output
    



# '''DPT算法的自定义模块，还需删除250行：class ReassembleBlocks(BaseModule)'''
# class ReassembleBlocks(BaseModule):
#     """ViTPostProcessBlock, process cls_token in ViT backbone output and
#     rearrange the feature vector to feature map.

#     Args:
#         in_channels (int): ViT feature channels. Default: 768.
#         out_channels (List): output channels of each stage.
#             Default: [96, 192, 384, 768].
#         readout_type (str): Type of readout operation. Default: 'ignore'.
#         patch_size (int): The patch size. Default: 16.
#         init_cfg (dict, optional): Initialization config dict. Default: None.
#     """

#     def __init__(self,
#                  in_channels=768,
#                  out_channels=[96, 192, 384, 768],
#                  readout_type='ignore',
#                  patch_size=16,
#                  init_cfg=None):
#         super().__init__(init_cfg)

#         assert readout_type in ['ignore', 'add', 'project']
#         self.readout_type = readout_type
#         self.patch_size = patch_size

#         self.projects = nn.ModuleList([
#             ConvModule(
#                 in_channels=in_channels,
#                 out_channels=out_channel,
#                 kernel_size=1,
#                 act_cfg=None,
#             ) for out_channel in out_channels
#         ])

#         self.resize_layers = nn.ModuleList([
#             nn.ConvTranspose2d(
#                 in_channels=out_channels[0],
#                 out_channels=out_channels[0],
#                 kernel_size=4,
#                 stride=4,
#                 padding=0),
#             nn.ConvTranspose2d(
#                 in_channels=out_channels[1],
#                 out_channels=out_channels[1],
#                 kernel_size=2,
#                 stride=2,
#                 padding=0),
#             nn.Identity(),
#             nn.Conv2d(
#                 in_channels=out_channels[3],
#                 out_channels=out_channels[3],
#                 kernel_size=3,
#                 stride=2,
#                 padding=1)
#         ])
#         if self.readout_type == 'project':
#             self.readout_projects = nn.ModuleList()
#             for _ in range(len(self.projects)):
#                 self.readout_projects.append(
#                     nn.Sequential(
#                         Linear(2 * in_channels, in_channels),
#                         build_activation_layer(dict(type='GELU'))))

#     def forward(self, inputs):
#         assert isinstance(inputs, list)
#         out = []
#         for i, x in enumerate(inputs):
#             assert len(x) == 2
#             x, cls_token = x[0], x[1]
#             feature_shape = x.shape
#             if self.readout_type == 'project':
#                 x = x.flatten(2).permute((0, 2, 1))
#                 readout = cls_token.unsqueeze(1).expand_as(x)
#                 x = self.readout_projects[i](torch.cat((x, readout), -1))
#                 x = x.permute(0, 2, 1).reshape(feature_shape)
#             elif self.readout_type == 'add':
#                 x = x.flatten(2) + cls_token.unsqueeze(-1)
#                 x = x.reshape(feature_shape)
#             else:
#                 pass
#             x = self.projects[i](x)
#             x = self.resize_layers[i](x)
#             out.append(x)
#         return out