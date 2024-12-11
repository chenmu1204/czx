import numpy as np
import torch
import torch.nn as nn
from mmseg.registry import MODELS
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from openTSNE import TSNE
import pandas as pd


@MODELS.register_module()
class CenterLoss(nn.Module):
    """Center loss.

    Reference:

    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=4, feat_dim=4, use_gpu=True, loss_name='loss_center', loss_weight=1.0):  # 修改权重时记得除权更新！学习率大点=0.5
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes - 1    # 背景0不计算，就不给中心点，共计3类需要计算centerLoss
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.ignore_label = 255      # 需要忽略的标签值255，后续需把0也忽略
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())     # 随机种子未固定，每次经过TSNE映射的二维坐标数量级差异巨大    # 中心点需要放入优化器更新，4类会产生4个中心，实际只有3类？？？？
        # 新增的参数
        # self.iter_num_center = 1
        # self.feat_loader = []
        # self.labels_loader = []
        # self.cent = []


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        '''存储特征numpy'''
        # # 存中心点
        # self.cent.append(self.centers.cpu().detach().numpy().reshape((1, 6)))
        # if self.iter_num_center % 937 == 0:
        #     centers = self.centers.cpu().detach().numpy()
        #     np.save(r'C:\Users\chenmu\Desktop\center-dim2/centers_{}.npy'.format(self.iter_num_center // 937), centers)
        #     df = pd.DataFrame(np.array(self.cent).squeeze(1), columns=["X1", "Y1", "X2", "Y2", "X3", "Y3"])
        #     df.to_excel(r"C:\Users\chenmu\Desktop\center-dim2/cent_{}.xlsx".format(self.iter_num_center // 937), index=False)
        #     self.cent = []
        #
        #
        # # 存训练集特征点
        # feat_result = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
        # labels_result = labels.view(1, -1).t()
        # mask = labels_result
        # for i in range(1, 4):
        #     if (mask == i).sum() != 0:
        #         # feat_mask = feat_result[mask == i]
        #         feat_mask = feat_result[(mask == i).expand(feat_result.size(0), feat_result.size(1))].reshape(
        #             (mask == i).sum(), feat_result.size(1))
        #         # labels_mask = labels_result[mask == i]
        #         labels_mask = labels_result[(mask == i).expand(labels_result.size(0), labels_result.size(1))].reshape(
        #             (mask == i).sum(), labels_result.size(1))
        #
        #         num = feat_mask.shape[0]
        #         sample_list = list(range(num))
        #         np.random.shuffle(sample_list)  # 打乱顺序，取前十个点
        #         list_sample = sample_list[0:10]
        #         feat_sample = feat_mask[list_sample]
        #         labels_sample = labels_mask[list_sample]
        #         self.feat_loader.append(feat_sample)
        #         self.labels_loader.append(labels_sample)
        # if self.iter_num_center % 937 == 0:
        #     feats = torch.cat(self.feat_loader, 0).data.cpu().numpy()
        #     label = torch.cat(self.labels_loader, 0).data.cpu().numpy()
        #     # centers_result = centers.clone().data.cpu().numpy()
        #     np.save(r'C:\Users\chenmu\Desktop\center-dim2/feats_{}.npy'.format(self.iter_num_center // 937), feats)
        #     np.save(r'C:\Users\chenmu\Desktop\center-dim2/labels_{}.npy'.format(self.iter_num_center // 937), label)
        #     self.feat_loader = []
        #     self.labels_loader = []  # 保存完之后缓存清空，接收下一个
        # self.iter_num_center += 1


        '''计算center_loss'''
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
        batch_size = x.size(0)
        # 计算每个样本与类别中心之间的欧氏距离 #
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()  # 做平方和，x^2+y^2
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)   # 做乘法，-2xy

        classes = torch.arange(self.num_classes).long().to(distmat.device) + 1   # 生成3个类的标签值[1,2,3]
        labels = labels.contiguous().view(-1).unsqueeze(1).expand(batch_size, self.num_classes)
        labels = torch.where(labels == 0, 255, labels)        # 将背景类‘0’设置为255，后续忽略掉
        not_ignore_spatial_mask = labels.int() != self.ignore_label
        labels = labels * not_ignore_spatial_mask
        mask = labels.eq(classes.expand(batch_size, self.num_classes))   # 值相同的位置返回True（所有实际像素的位置），否则返回False

        dist = distmat * mask.float()      # 过滤掉无关像素的值，只保留实际像素位置的值
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / mask.sum()   # 之前是除以batch_size，由于忽略掉了0和255，   应该是这个mask.sum()会使2、3类精度为0，为什么
                                                                     # 实际参与计算centerLoss的像素点大量减少，mask.sum()为实际参与计算点的数量
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