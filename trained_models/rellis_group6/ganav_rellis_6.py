norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='OursHeadClassAtt',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=384,
        mask_size=(97, 97),
        psa_type='bi-direction',
        compact=False,
        shrink_factor=2,
        normalization_factor=1.0,
        psa_softmax=True,
        dropout_ratio=0.1,
        num_classes=6,
        input_transform='multiple_select',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        attn_split=1,
        strides=(2, 1),
        size_index=1,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            static_weight=False),
        img_size=(375, 600)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=160,
            channels=32,
            num_convs=1,
            num_classes=6,
            in_index=-2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=6,
            in_index=-3,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'RELLISDataset_Group6'
data_root = 'data/rellis/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = (1920, 1200)
crop_size = (375, 600)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1200), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(375, 600)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(375, 600), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(375, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(375, 600), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    img_size=(375, 600),
    train=dict(
        type='RELLISDataset_Group6',
        data_root='data/rellis/',
        img_dir='image',
        ann_dir='annotation',
        split='train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(1920, 1200), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(375, 600)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(375, 600), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='RELLISDataset_Group6',
        data_root='data/rellis/',
        img_dir='image',
        ann_dir='annotation',
        split='val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(375, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(375, 600), pad_val=0, seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RELLISDataset_Group6',
        data_root='data/rellis/',
        img_dir='image',
        ann_dir='annotation',
        split='test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(375, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(375, 600), pad_val=0, seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=240000)
total_iters = 240000
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=240000, metric='mIoU')
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0001,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    by_epoch=False)
work_dir = './work_dirs/ganav_rellis_6'
gpu_ids = [6]
auto_resume = False
