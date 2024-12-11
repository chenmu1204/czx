import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalLossV1(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLossV1, self).__init__()
        self.gamma = gamma
        alpha = alpha.tolist()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        mask = target.int() != -1
        target = target * mask

        logpt = F.log_softmax(input)    # 这里转成log(pt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss * mask.squeeze()
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLossV2(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLossV2, self).__init__()
        self.gamma = gamma
        alpha = alpha.tolist()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        mask = target.int() != -1
        target = target * mask

        logpt = F.log_softmax(input)    # 这里转成log(pt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss * mask.squeeze()
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class FocalLossV3(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLossV3, self).__init__()
        self.gamma = gamma
        alpha = alpha.tolist()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        mask = target.int() != -1
        target = target * mask

        logpt = F.log_softmax(input)    # 这里转成log(pt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss * mask.squeeze()
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def flatten_hw(inputs):

    x = inputs.contiguous().view(inputs.shape[0], inputs.shape[1] * inputs.shape[2], -1)

    return x


# class FocalLossV3(nn.Module):
#     def __init__(self, gamma=3, alpha=None, size_average=True):
#         super(FocalLossV3, self).__init__()
#         self.gamma = gamma
#         alpha = alpha.tolist()
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1-alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
#         # target = target.view(-1, 1)
#         # mask = target.int() != -1
#         # target = target * mask
#         # label = flatten_hw(target)     # N,H,W => N,H*W
#         label = target.view(-1, 1)       # N,H,W => N,H*W
#         label = torch.tensor(label, dtype=torch.int64)  # [N, HW, 1]
#         label = torch.squeeze(label, dim=-1)
#
#         not_ignore_spatial_mask = label.int() != -1
#         label = label * not_ignore_spatial_mask
#
#         one_hot_label = torch.nn.functional.one_hot(label, num_classes=14)  # [N, HW, class]
#         one_hot_label = one_hot_label * not_ignore_spatial_mask.unsqueeze(-1)
#
#         logpt = F.log_softmax(input)    # 这里转成log(pt)
#         # logpt = logpt.gather(1, one_hot_label)
#         logpt = logpt * one_hot_label
#         # logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 alpha = self.alpha.type_as(input.data)
#                 alpha = alpha.unsqueeze(0).repeat(one_hot_label.size(0), 1)
#                 # alpha = alpha.unsqueeze(0).repeat(one_hot_label.size(0), 1, 1)
#             # at = self.alpha.gather(0, target.data.view(-1))
#             # at = self.alpha.gather(0, target.data.view(-1))
#             at = alpha * one_hot_label
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         # loss = loss * mask.squeeze()
#         return loss.mean(0).sum()

