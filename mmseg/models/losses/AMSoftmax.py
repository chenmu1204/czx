import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from mmseg.registry import MODELS


@MODELS.register_module()
class AMSoftmax(nn.Module):      # 两个超参数：一个是尺度s，另一个是边距m
    def __init__(self,
                 in_feats=128,
                 num_classes=4,  # 需要去除背景类0，在self.ce前
                 m=0.5,          # 原来是0.5。在softmax的基础上引入了额外的参数 m，一般较大的margin值，模型的判别性更高。
                 s=10,           # 原来是10，改为30以加速和稳定优化。
                 loss_name='loss_AMSoft', loss_weight=1.0):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, num_classes), requires_grad=True)   # 需要优化，计算余弦相似度：将输入特征映射到多维空间，每个维度对应一个类别
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.num_class = num_classes
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        nn.init.xavier_normal_(self.W, gain=1)   #  Xavier初始化方法，以确保模型在训练开始时具有合适的权重值

    def forward(self, x, lb, **kwargs):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        batchsize = x.size(0)
        height = x.size(-2)
        weight = x.size(-1)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)   # 元素级除法操作
        x_norm = x_norm.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
        W = self.W.to(x_norm.device)
        w_norm = torch.norm(W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(W, w_norm)
        costh = torch.matmul(x_norm, w_norm)   # （归一化后）初始化的权重矩阵self.W 与 输入特征相乘，计算余弦相似度

        not_ignore_spatial_mask = lb.int() != 255
        not_ignore_spatial_mask = not_ignore_spatial_mask.view(-1, 1)
        lb_view = lb.view(-1, 1)
        lb_view = lb_view * not_ignore_spatial_mask   # 所有255的值都变成了0

        if lb_view.is_cuda:
            lb_view = lb_view.cpu()

        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)  # 散点操作：.scatter_（在哪个维度散点操作，位置索引，分散到目标张量上的张量值）

        if x.is_cuda:
            delt_costh = delt_costh.to(x.device)
            delt_costh = delt_costh * not_ignore_spatial_mask

        costh_m = costh - delt_costh      # cos - m
        costh_m_s = self.s * costh_m      # 乘以尺度，论文中固定为30
        costh_m_s = costh_m_s.view(batchsize, height, weight, self.num_class)
        costh_m_s = costh_m_s.permute(0, 3, 1, 2)   # 等同于过了分类器的预测结果，将其送入CE计算损失值

        lb = lb.squeeze(1)
        # 去除背景类0
        # lb = torch.where(lb == 0, 255, lb)   # 去除背景后，初始loss会增大1，精度下降3.1%
        loss = self.ce(costh_m_s, lb)     # 交叉熵损失忽略255的值（包括背景0），可以试一下背景类能否有效聚类
        return self.loss_weight * loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name