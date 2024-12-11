# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList
from .fcn_head import FCNHead
from .segformer_head import SegformerHead
from .stdc_prehead import STDCpreHead

@MODELS.register_module()
class STDCHead(STDCpreHead):                                                  # 记得修改：STDC 的 FCNHead，Segformer 的 SegformerHead，改进的 STDCpreHead
    """This head is the implementation of `Rethinking BiSeNet For Real-time
    Semantic Segmentation <https://arxiv.org/abs/2104.13188>`_.

    Args:
        boundary_threshold (float): The threshold of calculating boundary.
            Default: 0.1.
    """

    def __init__(self, boundary_threshold=0.1, **kwargs):
        super().__init__(**kwargs)
        self.boundary_threshold = boundary_threshold
        # Using register buffer to make laplacian kernel on the same
        # device of `seg_label`.
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                         dtype=torch.float32,
                         requires_grad=False).reshape((1, 1, 3, 3)))
        self.fusion_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                         dtype=torch.float32).reshape(1, 3, 1, 1),
            requires_grad=False)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute Detail Aggregation Loss."""
        # Note: The paper claims `fusion_kernel` is a trainable 1x1 conv
        # parameters. However, it is a constant in original repo and other
        # codebase because it would not be added into computation graph
        # after threshold operation.
        seg_label = self._stack_batch_gt(batch_data_samples).to(            # (4,1,512,512)
            self.laplacian_kernel)
        
        seg_label[seg_label == 255] = 0       # 255的值全部转为0，忽略填充造成的图像边界

        boundary_targets = F.conv2d(seg_label, self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > self.boundary_threshold] = 1    # 阈值self.boundary_threshold=0.1
        boundary_targets[boundary_targets <= self.boundary_threshold] = 0   # (4,1,512,512)

        boundary_targets_x2 = F.conv2d(seg_label, self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)              # (4,1,256,256)

        boundary_targets_x4 = F.conv2d(seg_label, self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)              # (4,1,128,128)

        boundary_targets_x4_up = F.interpolate(
            boundary_targets_x4, boundary_targets.shape[2:], mode='nearest') # (4,1,512,512)
        boundary_targets_x2_up = F.interpolate(
            boundary_targets_x2, boundary_targets.shape[2:], mode='nearest') # (4,1,512,512)

        boundary_targets_x2_up[
            boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[
            boundary_targets_x2_up <= self.boundary_threshold] = 0

        boundary_targets_x4_up[
            boundary_targets_x4_up > self.boundary_threshold] = 1
        boundary_targets_x4_up[
            boundary_targets_x4_up <= self.boundary_threshold] = 0

        boundary_targets_pyramids = torch.stack(
            (boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)     # (4,3,1,512,512)

        boundary_targets_pyramids = boundary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boundary_targets_pyramids, self.fusion_kernel)  # (4,3,512,512)

        boudary_targets_pyramid[
            boudary_targets_pyramid > self.boundary_threshold] = 1
        boudary_targets_pyramid[
            boudary_targets_pyramid <= self.boundary_threshold] = 0

        seg_labels = boudary_targets_pyramid.long()   # (4,1,512,512)
        batch_sample_list = []
        for label in seg_labels:
            seg_data_sample = SegDataSample()
            seg_data_sample.gt_sem_seg = PixelData(data=label)
            batch_sample_list.append(seg_data_sample)

        loss = super().loss_by_feat(seg_logits, batch_sample_list)    # seg_logits（4,1,64,64），4个gt_sem_seg（512,512）




        # '''edge边缘图的可视化'''
        # # if self.counter > 0:
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
        # for i, data_edge in enumerate(seg_logits):                     # （4,1,128,128）
        #     # image_data = torch.mean(seg_data_sample, dim=0, keepdim=True)   # 平均池化  
        #     image_data = np.squeeze(data_edge).data.cpu().numpy()          # 将数据移动到CPU并转换为numpy数组
        #     image_data = image_data*255
        #     # 显示图像
        #     plt.figure()
        #     plt.imshow(image_data, cmap='gray')  # 使用viridis颜色图，可以根据需要更改, cmap='gray'
        #     # 保存图像为PNG文件
        #     save_path = f"{save_folder}{image_names[i]}.png"
        #     plt.savefig(save_path)
        #     plt.close()  # 关闭当前图形，确保下一次循环时创建新图形




        # '''gt边缘图的可视化'''
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import os
        # image_names = []
        # for seg_data_sample in batch_data_samples:
        #     img_path = seg_data_sample.img_path
        #     image_name = os.path.basename(img_path)        # 提取文件名
        #     image_name = os.path.splitext(image_name)[0]   # 去除后缀
        #     image_names.append(image_name)
        # # 设置保存图像的文件夹路径
        # save_folder = "/data3/chenzhenxiang/image/"
        # for i, seg_data_sample in enumerate(batch_sample_list):
        #     image_data = seg_data_sample._gt_sem_seg.data.cpu().numpy()  # 将数据移动到CPU并转换为numpy数组
        #     image_data = np.squeeze(image_data)                          # 去掉单通道的维度
        #     image_data = image_data*255
        #     # 显示图像
        #     plt.figure(figsize=(8, 8))
        #     plt.imshow(image_data, cmap='gray')  # 使用viridis颜色图，可以根据需要更改
        #     plt.title(f'{image_names[i]}')
        #     #plt.colorbar()  # 显示颜色条，如果图像是分割标签，则显示标签对应的颜色
        #     # 保存图像为PNG文件
        #     save_path = f"{save_folder}{image_names[i]}.png"
        #     plt.savefig(save_path)
        #     plt.close()  # 关闭当前图形，确保下一次循环时创建新图形

        return loss    # 返回三个Loss值：loss_ce，loss_dice，acc_seg