_base_ = [
    '../_base_/models/upernet_swin_rmi.py', '../_base_/datasets/coco-trash.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        convert_weights=True,
        patch_norm=True),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=11),
    auxiliary_head=dict(in_channels=768, num_classes=11))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-6,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-7,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
