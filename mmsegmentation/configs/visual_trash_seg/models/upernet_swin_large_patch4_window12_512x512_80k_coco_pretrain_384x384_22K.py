_base_ = [
    '../../_base_/models/upernet_swin.py', '../datasets/coco-trash.py',
    '../runtime/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
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
        drop_path_rate=0.2,
        convert_weights=True,
        patch_norm=True),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=11),
    auxiliary_head=dict(in_channels=768, num_classes=11))

data = dict(
    train=dict(
        img_dir='train_all/img',
        ann_dir='train_all/ann'))
