# dataset settings
dataset_type = 'remote_sense'
data_root = '/data3/chenzhenxiang/mydata/RSMI_Fine_4/'                 # /data3/chenzhenxiang/mydata/RSMI_Fine/
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),                                    # 第1个流程，从文件路径里加载图像
    dict(type='LoadAnnotations', reduce_zero_label=False),             # 第2个流程，对于当前图像，加载它的标注图像
    dict(type='RandomResize',                                          # 调整输入图像大小(resize)和其标注图像的数据增广流程
         scale=(640, 640),                                             # 图像裁剪的大小
         ratio_range=(0.5, 2.0),                                       # 数据增广的比例范围
         keep_ratio=True),                                             # 调整图像大小时是否保持纵横比
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  # 随机裁剪当前图像和其标注图像的数据增广流程、大小、单个类别可以填充的最大区域的比
    dict(type='RandomFlip', prob=0.5),                                 # 翻转图像和其标注图像的数据增广流程、翻转图像的概率
    dict(type='PhotoMetricDistortion'),                                # 光学上使用一些方法扭曲当前图像和其标注图像的数据增广流程
    dict(type='PackSegInputs')                                         # 打包用于语义分割的输入数据
]
test_pipeline = [
    dict(type='LoadImageFromFile'),                                    # 第1个流程，从文件路径里加载图像
    dict(type='Resize', scale=(512, 512), keep_ratio=True),            # 使用调整图像大小(resize)增强     之前是(640, 640)
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),             # 加载数据集提供的语义分割标注
    dict(type='PackSegInputs')                                         # 打包用于语义分割的输入数据
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,                                               # 【注意】本地电脑CPU只有12核
    persistent_workers=True,                                     # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),          # 训练时进行随机洗牌(shuffle)
    dataset=dict(                                                # 训练数据集配置
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train', seg_map_path='mask/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),          # 验证、测试时不进行随机洗牌(shuffle)
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='mask/test'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
