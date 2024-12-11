import torch
import torch.nn as nn
from CAR_utils.CAR_torch_utils import *
import torch.nn.functional as F
from CAR_utils.vismanager_pytorh import get_visualization_manager
from CAR_utils.normalizations_pytorch import normalization
import torch.distributed as dist
from mmseg.registry import MODELS


@MODELS.register_module()
class CAR_Loss(nn.Module):
    def __init__(
        self,
        train_mode=False,
        use_inter_class_loss=True,
        use_intra_class_loss=True,
        intra_class_loss_remove_max=False,
        # '''中心距离'''
        use_inter_c2c_loss=True,
        use_inter_c2p_loss=True,

        intra_class_loss_rate=1,
        inter_class_loss_rate=1,
        num_class=24,
        ignore_label=-1,
        pooling_rates=[1],
        use_batch_class_center=True,
        use_last_class_center=False,
        last_class_center_decay=0.9,
        inter_c2c_loss_threshold=0.5,
        inter_c2p_loss_threshold=0.25,
        filters=None,
        input_channel=None,
        apply_convs=True,
        name=None,
    ):

        super().__init__()

        self.vis_manager = get_visualization_manager()

        self.train_mode = train_mode
        self.use_inter_class_loss = use_inter_class_loss
        self.use_intra_class_loss = use_intra_class_loss
        self.intra_class_loss_rate = intra_class_loss_rate
        self.inter_class_loss_rate = inter_class_loss_rate
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.inter_c2c_loss_threshold = inter_c2c_loss_threshold
        self.inter_c2p_loss_threshold = inter_c2p_loss_threshold

        self.intra_class_loss_remove_max = intra_class_loss_remove_max

        self.use_inter_c2c_loss = use_inter_c2c_loss
        self.use_inter_c2p_loss = use_inter_c2p_loss

        self.filters = filters
        self.input_channels = input_channel
        self.apply_convs = apply_convs

        self.name = name

        if isinstance(pooling_rates, tuple):
            pooling_rates = list(pooling_rates)

        if not isinstance(pooling_rates, list):
            pooling_rates = [pooling_rates]

        self.pooling_rates = pooling_rates
        self.use_batch_class_center = use_batch_class_center
        self.use_last_class_center = use_last_class_center
        self.last_class_center_decay = last_class_center_decay

        # Get input channels from input_shape

        if self.use_last_class_center:
            self.last_class_center = nn.Parameter(torch.empty(1, self.num_class, self.filters))
            nn.init.xavier_uniform_(self.last_class_center)
            self.last_class_center.requires_grad_(False)

        if self.apply_convs:
            self.end_conv = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=(1, 1),
                                 bias=False)
            self.end_norm = normalization(name="batch_norm", num_channels=self.filters, num_groups=32)
            # self.end_norm = nn.BatchNorm2d(num_features=self.filters)

        self.linear_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=self.filters, kernel_size=(1, 1),
                                bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, data, label, extra_prefix=None, training=None):
        # global loss_total
        loss_total = 0

        training = training

        data = self.linear_conv(data)

        # features: [N, H, W, C]
        features = data.permute(0, 2, 3, 1)
        # mask = label.int() != self.ignore_label
        # features = features * mask.unsqueeze(-1)

        loss_name_prefix = f"{self.name}"

        if extra_prefix is not None:
            loss_name_prefix = f"{loss_name_prefix}_{extra_prefix}"

        inputs_shape = features.shape
        height = torch.tensor(inputs_shape[-3], dtype=torch.int)
        width = torch.tensor(inputs_shape[-2], dtype=torch.int)

        # label = resize_image(label.float().unsqueeze(1), (height, width), method="nearest").permute(0, 2, 3, 1)
        label = label.float().unsqueeze(-1)

        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("inputs contains nan or inf")

        flatten_features = flatten_hw(features)

        not_ignore_spatial_mask = label.int() != self.ignore_label  # [N, H, W, 1]
        not_ignore_spatial_mask = flatten_hw(not_ignore_spatial_mask)

        one_hot_label = get_flatten_one_hot_label(
            label, num_class=self.num_class, ignore_label=self.ignore_label
        )  # [N, HW, class]

        class_sum_features, class_sum_non_zero_map = get_class_sum_features_and_counts(
            flatten_features, one_hot_label
        )  # [N, class, C]

        if self.use_batch_class_center:

            class_sum_features_in_cross_batch = class_sum_features.sum(dim=0, keepdim=True)
            class_sum_non_zero_map_in_cross_batch = class_sum_non_zero_map.sum(dim=0, keepdim=True)

            if dist.is_available() and dist.is_initialized():
                class_sum_features_in_cross_batch = dist.all_reduce(class_sum_features_in_cross_batch,
                                                                    op=dist.ReduceOp.SUM)
                class_sum_non_zero_map_in_cross_batch = dist.all_reduce(class_sum_non_zero_map_in_cross_batch,
                                                                        op=dist.ReduceOp.SUM)

            class_avg_features_in_cross_batch = class_sum_features_in_cross_batch / class_sum_non_zero_map_in_cross_batch
            class_avg_features_in_cross_batch = torch.where(torch.isnan(class_avg_features_in_cross_batch),
                                                            torch.zeros_like(class_avg_features_in_cross_batch),
                                                            class_avg_features_in_cross_batch)
            class_avg_features_in_cross_batch = torch.where(torch.isinf(class_avg_features_in_cross_batch),
                                                            torch.zeros_like(class_avg_features_in_cross_batch),
                                                            class_avg_features_in_cross_batch)

            if self.use_last_class_center:
                batch_class_ignore_mask = (class_sum_non_zero_map_in_cross_batch != 0).int()

                class_center_diff = class_avg_features_in_cross_batch - self.last_class_center.type(
                    class_avg_features_in_cross_batch.dtype)
                class_center_diff *= (1 - self.last_class_center_decay) * batch_class_ignore_mask.type(
                    class_center_diff.dtype)

                last_class_center = self.last_class_center + class_center_diff

                class_avg_features_in_cross_batch = last_class_center.float()

            class_avg_features = class_avg_features_in_cross_batch

            class_avg_features = torch.where(torch.isnan(class_avg_features), torch.zeros_like(class_avg_features),
                                             class_avg_features)
            class_avg_features = torch.where(torch.isinf(class_avg_features), torch.zeros_like(class_avg_features),
                                             class_avg_features)

        else:
            class_avg_features = class_sum_features / class_sum_non_zero_map

            class_avg_features = torch.where(torch.isnan(class_avg_features), torch.zeros_like(class_avg_features),
                                             class_avg_features)
            class_avg_features = torch.where(torch.isinf(class_avg_features), torch.zeros_like(class_avg_features),
                                             class_avg_features)

        if self.use_inter_class_loss and training:

            inter_class_relative_loss = 0

            if self.use_inter_c2c_loss:
                inter_class_relative_loss += get_inter_class_relative_loss(
                    class_features_query=class_avg_features, inter_c2c_loss_threshold=self.inter_c2c_loss_threshold,
                )

            if self.use_inter_c2p_loss:
                inter_class_relative_loss += get_pixel_inter_class_relative_loss(
                    x=flatten_features, class_avg_feature=class_avg_features, one_hot_label=one_hot_label,
                    inter_c2p_loss_threshold=self.inter_c2p_loss_threshold,
                )

            loss_total = inter_class_relative_loss * self.inter_class_loss_rate

            # self.add_loss(inter_class_relative_loss * self.inter_class_loss_rate)
            # self.add_metric(inter_class_relative_loss, name=f"{loss_name_prefix}_orl")

        if self.use_intra_class_loss:

            if torch.isnan(class_avg_features).any() or torch.isinf(class_avg_features).any():
                raise ValueError("inputs contains nan or inf")

            same_avg_value = torch.matmul(one_hot_label, class_avg_features)

            if torch.isnan(same_avg_value).any() or torch.isinf(same_avg_value).any():
                raise ValueError("inputs contains nan or inf")

            self_absolute_loss = get_intra_class_absolute_loss(
                flatten_features,
                same_avg_value,
                remove_max_value=self.intra_class_loss_remove_max,
                not_ignore_spatial_mask=not_ignore_spatial_mask,
            )

            if training:
                loss_total = loss_total + self_absolute_loss * self.intra_class_loss_rate
                # loss_total = loss_total + self_absolute_loss * 100
                # self.add_loss(self_absolute_loss * self.intra_class_loss_rate)
                # self.add_metric(self_absolute_loss, name=f"{loss_name_prefix}_sal")

            # print("Using self-loss")

        return loss_total