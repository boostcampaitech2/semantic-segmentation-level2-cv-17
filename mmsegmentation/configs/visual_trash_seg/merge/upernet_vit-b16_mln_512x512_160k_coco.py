_base_ = [
    '../../_base_/models/upernet_vit-b16_ln_mln.py',
    '../datasets/coco-trash.py', '../runtime/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py'
]

model = dict(
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_ln_mln_512x512_160k_ade20k/upernet_deit-b16_ln_mln_512x512_160k_ade20k_20210623_153535-8a959c14.pth',
    decode_head=dict(num_classes=11),
    auxiliary_head=dict(num_classes=11),
    backbone=dict(patch_size=8, img_size=(224, 224)))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.) 
        }))

norm_cfg = dict(type='BN', requires_grad=True)
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
seed=1995