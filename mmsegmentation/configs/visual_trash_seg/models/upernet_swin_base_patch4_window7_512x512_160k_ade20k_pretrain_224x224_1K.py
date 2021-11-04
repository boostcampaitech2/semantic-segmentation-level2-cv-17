_base_ = [
    './upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_'
    'pretrain_224x224_1K.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth',
    backbone=dict(
        embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=11, norm_cfg = norm_cfg),
    auxiliary_head=dict(in_channels=512, num_classes=11, norm_cfg = norm_cfg))
