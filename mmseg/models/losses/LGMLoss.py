import torch
import torch.nn as nn
from mmseg.registry import MODELS


@MODELS.register_module()
class LGMLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """
    def __init__(self, num_classes=4, feat_dim=256, alpha=0.00, loss_name='loss_LGM'):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.ignore_label = 255
        self._loss_name = loss_name

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self,
                feat,
                label,
                **kwargs):

        feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, feat.size(1))
        batch_size = feat.shape[0]
        log_covs = self.log_covs.to(feat.device)
        centers = self.centers.to(feat.device)
        log_covs = torch.unsqueeze(log_covs, dim=0)


        covs = torch.exp(log_covs) # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1) # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1) #eq.(18)

        label = label.contiguous().view(-1).unsqueeze(1).expand(batch_size, self.num_classes)
        not_ignore_spatial_mask = label.int() != self.ignore_label
        label = label * not_ignore_spatial_mask

        one_hot_label = torch.nn.functional.one_hot(label, num_classes=self.num_classes)  # [NHW, class]
        one_hot_label = one_hot_label * not_ignore_spatial_mask.unsqueeze(-1)
        margin_dist = torch.mul(dist, one_hot_label)

        slog_covs = torch.sum(log_covs, dim=-1) #1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5*(tslog_covs + margin_dist) #eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5*torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0/batch_size) * (cdist + reg)

        return likelihood  #logits, margin_logits,

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
