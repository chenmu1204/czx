checkpoint = '"/home1/users/chenzhenxiang02/mmsegmentation-main/weights/stdc1_20220308-5368626c.pth"'  # noqa
_base_ = './stdc1_4xb12-80k_cityscapes-512x1024.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))))
