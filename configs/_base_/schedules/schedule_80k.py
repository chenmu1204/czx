# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),   # lr=0.01
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)  # 优化器包装器(Optimizer wrapper)为更新参数提供了一个公共接口
                                                                                # 如果 'clip_grad' 不是None，它将是 ' torch.nn.utils.clip_grad' 的参数
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',    # 调度流程的策略，同样支持 Step, CosineAnnealing, Cyclic 等
        eta_min=1e-4,     # 训练结束时的最小学习率
        power=0.9,        # 多项式衰减 (polynomial decay) 的幂
        begin=0,          # 开始更新参数的时间步(step)
        end=80000,        # 停止更新参数的时间步(step)
        by_epoch=False)   # 是否按照 epoch 计算训练时间
]

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=120000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),                                        # 记录迭代过程中花费的时间
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),  # 从'Runner'的不同组件收集和写入日志
    param_scheduler=dict(type='ParamSchedulerHook'),                         # 更新优化器中的一些超参数，例如学习率
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),   # 定期保存检查点(checkpoint)
    sampler_seed=dict(type='DistSamplerSeedHook'),                           # 用于分布式训练的数据加载采样器
    visualization=dict(type='SegVisualizationHook'))                         # 可视化工具配置
