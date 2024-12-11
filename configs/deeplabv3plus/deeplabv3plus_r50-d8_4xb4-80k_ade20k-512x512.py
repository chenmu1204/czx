_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/remoteDataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4))

# 我加的
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))
train_dataloader = dict(batch_size=4,
                        num_workers=4)
val_dataloader = dict(batch_size=1,
                      num_workers=4)
test_dataloader = val_dataloader