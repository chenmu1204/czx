_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/potsdam.py', '../_base_/default_runtime.py',   # 【1】改数据集remoteDataset.py  whdld.py   potsdam.py  vaihingen.py
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = "/home1/users/chenzhenxiang02/mmsegmentation-main/weights/vit_small_p16_384_20220308-410f6037.pth"

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        img_size=(512, 512),
        embed_dims=384,
        num_heads=6,
    ),
    # 添加：特征转换为金字塔
    neck=dict(type='Feature2Pyramid', embed_dim=384, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=384,
        dropout_ratio=0.1,
        num_classes=6,                                    # 【2】改类别数！
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # decode_head=dict(
    #     type='SegmenterMaskTransformerHead',
    #     in_channels=384,
    #     channels=384,
    #     num_classes=6,                                   # 【2】改类别数！
    #     num_layers=2,
    #     num_heads=6,
    #     embed_dims=384,
    #     dropout_ratio=0.0,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        )

# optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(_delete_=True, type='OptimWrapper', 
                     optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01))
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
