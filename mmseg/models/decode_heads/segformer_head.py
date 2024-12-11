# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from mmseg.models.losses.Center_Loss import *


@MODELS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)



        '''思路1：自定义卷积层'''
        channels1 = [32, 64, 160, 256]
        channels2 = [32, 32, 64, 160]
        self.convs_enhance = nn.ModuleList()
        for i in range(num_inputs):
            self.convs_enhance.append(
                ConvModule(
                    in_channels=channels1[i],
                    out_channels=channels2[i],
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.fu_conv = nn.ModuleList()
        for i in range(num_inputs-1):
            self.fu_conv.append(
                ConvModule(
                    in_channels=channels1[i],
                    out_channels=channels1[i],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg))




        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)


        # self.counter = 1  # 用于可视化的计数器




    def forward(self, inputs):   # 删掉batch_data_samples
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)



        # '''inputs[0]的可视化'''
        # if self.counter > 0:
        #     import matplotlib.pyplot as plt
        #     import numpy as np
        #     import os
        #     from PIL import Image
        #     # 设置保存图像的文件夹路径
        #     image_names = []
        #     for seg_data_sample in batch_data_samples:         ### decode_head.py的第274行，添加传参batch_data_samples，涉及到的所有head都要加 ###
        #         # img_path = seg_data_sample.img_path
        #         img_path = seg_data_sample['img_path']
        #         image_name = os.path.basename(img_path)        # 提取文件名
        #         image_name = os.path.splitext(image_name)[0]   # 去除后缀
        #         image_names.append(image_name)
        #     save_folder = "/data3/chenzhenxiang/work_dirs/test/image/"
        #     for i, seg_data_sample in enumerate(inputs[0]):                     # （4,32,128,128）
        #         image_data = torch.mean(seg_data_sample, dim=0, keepdim=True)   # 平均池化  
        #         image_data = np.squeeze(image_data).data.cpu().numpy()          # 将数据移动到CPU并转换为numpy数组
        #         # image_data = image_data / np.max(image_data)
        #         # image_data = Image.fromarray(np.uint8(image_data * 255))
        #         #image_data = image_data.convert("L")
        #         # 显示图像
        #         plt.figure()
        #         plt.imshow(image_data)  # 使用viridis颜色图，可以根据需要更改, cmap='gray'
        #         # 保存图像为PNG文件
        #         save_path = f"{save_folder}iters{self.counter}_{image_names[i]}.png"
        #         plt.savefig(save_path)
        #         plt.close()  # 关闭当前图形，确保下一次循环时创建新图形
        # self.counter = self.counter + 1      # 初始化定义要打开      





        '''思路1:特征增强模块'''
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
                        mode=self.interpolate_mode, align_corners=self.align_corners)  # 先改变通道数，再resize
            edge = self.fu_conv[idx-1](x2 + x1)             # 可替换融合方法：cat、×、+、-
            if idx == 3:
                outs.append(x1)
            feature = self.fu_conv[idx-1](beta * edge + outs[3 - idx])
            if idx == 3:
                feature_enhance = aerfa * edge * feature          #########经过运算后，未进行卷积、BN、ReLU操作！！！########
                outs_enhance.append(feature_enhance)
                outs.append(resize(input=self.convs_enhance[idx - 1](feature_enhance), size=[64,64], 
                                   mode=self.interpolate_mode, align_corners=self.align_corners))
            elif idx == 2:
                feature_enhance = aerfa * edge * feature
                outs_enhance.append(feature_enhance)
                outs.append(resize(input=self.convs_enhance[idx - 1](feature_enhance), size=[128,128], 
                                   mode=self.interpolate_mode, align_corners=self.align_corners))
            else:
                feature_enhance = aerfa * edge * feature
                outs_enhance.append(feature_enhance)
                outs.append(feature_enhance)
        outs_enhance = outs_enhance[::-1]
        inputs = outs_enhance     # 只替换（4,160,32,32）或（4,32,128,128）的增强特征



        # '''edge边缘图的可视化'''
        # if self.counter > 0:
        #     import matplotlib.pyplot as plt
        #     import numpy as np
        #     import os
        #     from PIL import Image
        #     # 设置保存图像的文件夹路径
        #     image_names = []
        #     for seg_data_sample in batch_data_samples:         ### decode_head.py的第274行，添加传参batch_data_samples，涉及到的所有head都要加 ###
        #         # img_path = seg_data_sample.img_path
        #         img_path = seg_data_sample['img_path']
        #         image_name = os.path.basename(img_path)        # 提取文件名
        #         image_name = os.path.splitext(image_name)[0]   # 去除后缀
        #         image_names.append(image_name)
        #     save_folder = "/data3/chenzhenxiang/work_dirs/test/image/"
        #     for i, seg_data_sample in enumerate(edge):                     # （4,32,128,128）
        #         image_data = torch.mean(seg_data_sample, dim=0, keepdim=True)   # 平均池化  
        #         image_data = np.squeeze(image_data).data.cpu().numpy()          # 将数据移动到CPU并转换为numpy数组
        #         # image_data = image_data / np.max(image_data)
        #         # image_data = Image.fromarray(np.uint8(image_data * 255))
        #         #image_data = image_data.convert("L")
        #         # 显示图像
        #         plt.figure()
        #         plt.imshow(image_data)  # 使用viridis颜色图，可以根据需要更改, cmap='gray'
        #         # 保存图像为PNG文件
        #         save_path = f"{save_folder}iters{self.counter}_{image_names[i]}.png"
        #         plt.savefig(save_path)
        #         plt.close()  # 关闭当前图形，确保下一次循环时创建新图形
        # self.counter = self.counter + 1      # 初始化定义要打开

        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        output = self.fusion_conv(torch.cat(outs, dim=1))  # （4,256,128,128）
        out = self.cls_seg(output)        # (4,4,128,128)，预测结果

        return out