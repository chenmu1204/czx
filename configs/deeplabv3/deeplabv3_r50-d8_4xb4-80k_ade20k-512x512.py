_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/remoteDataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained="/home1/users/chenzhenxiang02/mmsegmentation-main/weights/resnet50_v1c-2cccc1ad.pth",
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4))
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)

# 因为代码报错新加的
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.000006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))