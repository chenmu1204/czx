_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/potsdam.py',   # 【1】改数据集remoteDataset.py  whdld.py  potsdam.py  vaihingen.py
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = '/home1/users/chenzhenxiang02/mmsegmentation-main/weights/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=6))                                         # 【2】改类别数

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.000006, betas=(0.9, 0.999), weight_decay=0.01),
                # dict(type='AdamW', lr=0.000006, betas=(0.9, 0.999), weight_decay=0.01),
                # dict(type='SGD', lr=0.5)),    # transformer不建议使用SGD优化器
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),   # 指定哪些模型组件需要特殊的学习率和权重衰减设置
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# 使用2个参数调度策略是为了灵活地控制不同参数的学习率和权重衰减
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),         # 线性学习率衰减
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=160000, by_epoch=False)  # 多项式学习率衰减
]
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
