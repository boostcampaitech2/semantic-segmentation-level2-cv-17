# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.001,
    step=[8, 11])
# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=20000)
runner = dict(type='EpochBasedRunner', max_epochs=3)
# checkpoint_config = dict(by_epoch=False, interval=2000)
checkpoint_config = dict(interval=1)
# evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)
