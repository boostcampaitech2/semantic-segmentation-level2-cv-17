_base_ = './upernet_vit-b16_mln_512x512_160k_coco.py'

model = dict(
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_80k_ade20k/',
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    decode_head=dict(num_classes=11, in_channels=[384, 384, 384, 384]),
    neck=None,
    auxiliary_head=dict(num_classes=11, in_channels=384))
