# 将注册表的默认范围设置为 mmseg
default_scope = 'mmseg'

# environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None        # 从文件中加载检查点(checkpoint)
resume = False          # 是否从已有的模型恢复

tta_model = dict(type='SegTTAModel')
