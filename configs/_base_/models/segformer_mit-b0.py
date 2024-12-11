# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # 分割框架通常使用 SyncBN
data_preprocessor = dict(
    type='SegDataPreProcessor',                     # 数据预处理的类型
    mean=[123.675, 116.28, 103.53],                 # 用于归一化输入图像的平均值
    std=[58.395, 57.12, 57.375],                    # 用于归一化输入图像的标准差
    bgr_to_rgb=True,                                # 是否将图像从 BGR 转为 RGB
    pad_val=0,                                      # 图像的填充值
    seg_pad_val=255)                                # 'gt_seg_map'的填充值！！！这个要注意
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    

    # auxiliary_head=dict(       ### 添加的边缘监督loss ### 
    #     type='STDCHead',       # 修改了 STDC 的 FCNHead，Segformer 的 SegformerHead
    #     in_channels=64, # [32, 64, 160, 256],
    #     in_index=0, #[0, 1, 2, 3],
    #     channels=256,
    #     dropout_ratio=0.1,
    #     num_classes=1,          # 原本是4，decode_head.py第275行，输入STDC_head的特征图不再是（4,4,128,128），而是（4,1,128,128）
    #     norm_cfg=norm_cfg,  
    #     boundary_threshold=0.1,
    #     align_corners=True,
    #     loss_decode=[
    #         dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=True, loss_weight=1.0),   # use_sigmoid=True：一般二分类才使用二值交叉熵损失，这里计算了二值边界的损失
    #         # dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
    #         ]),

    # model training and testing settings
    train_cfg=dict(),                      # train_cfg 当前仅是一个占位符
    test_cfg=dict(mode='whole'))           # 测试模式，可选参数为 'whole' 和 'slide'.
                                           # 'whole': 在整张图像上全卷积(fully-convolutional)测试
                                           # 'slide': 在输入图像上做滑窗预测
    # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0      type='CenterLoss', num_classes=4, loss_name='loss_center', loss_weight=1.0