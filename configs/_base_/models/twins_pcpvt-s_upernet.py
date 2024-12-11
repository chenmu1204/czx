checkpoint = '/home1/users/chenzhenxiang02/mmsegmentation-main/weights/pcpvt_small_20220308-e638c41c.pth'  # noqa
#'/home1/users/chenzhenxiang02/mmsegmentation-main/work_dirs/twins_pcpvt-s_uperhead-inputs[0]/iter_60000.pth'

# model settings
backbone_norm_cfg = dict(type='LN')
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PCPVT',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[4, 2, 2, 2],
        strides=[4, 2, 2, 2],
        mlp_ratios=[8, 8, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        norm_after_stage=False,
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.2),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=
        dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        
        # dict(       ### 添加的边缘监督loss ### 
        # type='STDCHead',
        # in_channels=320,   # 这个需要手动改，确保用于计算loss的特征层通道数与卷积核一致
        # in_index=2,
        # channels=256,
        # # num_convs=1,
        # # concat_input=False,
        # dropout_ratio=0.1,
        # num_classes=1,          # 原本是4，decode_head.py第275行，输入STDC_head的特征图不再是（4,4,21,8,128），而是（4,1,21,8,128） 
        # boundary_threshold=0.1,
        # align_corners=True,
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=True, loss_weight=1.0),   # use_sigmoid=True：一般二分类才使用二值交叉熵损失，这里计算了二值边界的损失
        #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        #     ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))