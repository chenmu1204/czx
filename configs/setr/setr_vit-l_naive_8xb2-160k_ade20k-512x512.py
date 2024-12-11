_base_ = [
    '../_base_/models/setr_naive.py', '../_base_/datasets/remoteDataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        img_size=(512, 512),
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint="/home1/users/chenzhenxiang02/mmsegmentation-main/weights/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth")
            ),
    decode_head=dict(num_classes=4),
    auxiliary_head=#[
        dict(
            _delete_=True,
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=4,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        # dict(
        #     type='SETRUPHead',
        #     in_channels=1024,
        #     channels=256,
        #     in_index=1,
        #     num_classes=4,
        #     dropout_ratio=0,
        #     norm_cfg=norm_cfg,
        #     act_cfg=dict(type='ReLU'),
        #     num_convs=2,
        #     kernel_size=1,
        #     align_corners=False,
        #     loss_decode=dict(
        #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        # dict(
        #     type='SETRUPHead',
        #     in_channels=1024,
        #     channels=256,
        #     in_index=2,
        #     num_classes=4,
        #     dropout_ratio=0,
        #     norm_cfg=norm_cfg,
        #     act_cfg=dict(type='ReLU'),
        #     num_convs=2,
        #     kernel_size=1,
        #     align_corners=False,
        #     loss_decode=dict(
        #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    # ],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

#optimizer = dict(_delete_=True, lr=0.01, weight_decay=0.0)
optimizer = dict(_delete_=True, type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
