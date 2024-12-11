import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
from mmseg.registry import MODELS


@MODELS.register_module()
class LMCL_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes=4, feat_dim=4, s=7.00, m=0.2, loss_name='loss_LMC'):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.ignore_label = -1
        self._loss_name = loss_name

    def forward(self, feat, label, **kwargs):
        B, H, W = feat.size(0), feat.size(2), feat.size(3)
        feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, feat.size(1))
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        centers = self.centers.to(feat.device)
        norms_c = torch.norm(centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        # label = label.contiguous().view(-1).unsqueeze(1).expand(batch_size, self.num_classes)
        label = label.contiguous().view(-1)
        not_ignore_spatial_mask = label.int() != self.ignore_label
        label = label * not_ignore_spatial_mask

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        y_onehot = y_onehot * not_ignore_spatial_mask.unsqueeze(-1)
        margin_logits = self.s * (logits - y_onehot)
        margin_logits = margin_logits.view(B, H, W, -1)
        margin_logits = margin_logits.permute(0, 3, 1, 2)

        logits = logits.view(B, H, W, -1)
        logits = logits.permute(0, 3, 1, 2)

        return logits #, margin_logits


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
