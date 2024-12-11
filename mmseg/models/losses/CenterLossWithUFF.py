import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import Function
from feature_fusion_loss import FeatureFusionBlock


class CenterLossWithUFF(nn.Module):
    """Center loss.

    Reference:

    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=24, feat_dim=2, use_gpu=True):
        super(CenterLossWithUFF, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.center_data = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.center_data_spec = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.ignore_label = -1
        self.uff = FeatureFusionBlock(xyz_dim=feat_dim, rgb_dim=feat_dim).cuda()

    def forward(self, data, data_spec, label):
        """
        Args:
            data: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        data = data.permute(0, 2, 3, 1).contiguous().view(-1, data.size(1))
        data_spec = data_spec.permute(0, 2, 3, 1).contiguous().view(-1, data_spec.size(1))

        batch_size = data.size(0)

        center_data = self.center_data.to(data.device)
        center_data_spec = self.center_data_spec.to(data_spec.device)

        distmat_data = torch.pow(data, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(center_data, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat_data.addmm_(data, center_data.t(), beta=1, alpha=-2)

        distmat_data_spec = torch.pow(data_spec, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(center_data_spec, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat_data_spec.addmm_(data_spec, center_data_spec.t(), beta=1, alpha=-2)
        # distmat.addmm_(1, -2, x, centers.sum(dim=1, keepdim=True).expand(self.num_classes, x.size(-1)).t())

        classes = torch.arange(self.num_classes).long().to(data.device)

        labels = label.contiguous().view(-1).unsqueeze(1).expand(batch_size, self.num_classes)
        not_ignore_spatial_mask = labels.int() != self.ignore_label
        labels = labels * not_ignore_spatial_mask
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist_data = distmat_data * mask.float()
        dist_data = dist_data * not_ignore_spatial_mask
        dist_data = dist_data.clamp(min=1e-12, max=1e+12)
        dist_data_spec = distmat_data_spec * mask.float()
        dist_data_spec = dist_data_spec * not_ignore_spatial_mask
        dist_data_spec = dist_data_spec.clamp(min=1e-12, max=1e+12)

        loss_data = dist_data.sum() / batch_size
        loss_data_spec = dist_data_spec.sum() / batch_size

        loss_center = loss_data + loss_data_spec

        # UFF
        feature_data, feature_data_spec = center_data, center_data_spec
        loss_uff = self.uff(feature_data, feature_data_spec)

        return loss_center, loss_uff