# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    min_lr=0.0,
    periods=[16000,16000,16000,16000,16000],
    restart_weights=[1,0.5,0.25,0.125,0.0625],
    by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU')
