# dataset settings
dataset_type = 'GOOSEDataset_Group6'
data_root = 'data/goose/'
img_norm_cfg = dict(
    mean=[80.66, 77.73, 76.87], std=[61.40, 61.08, 64.47], to_rgb=True)
img_size = (2048, 1000)
crop_size = (375, 600)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    img_size=crop_size,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train/',
        ann_dir='labels/train/',
        split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val/',
        ann_dir='labels/val/',
        split='val.txt',
        pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='image',
    #     ann_dir='labels',
    #     split='test.txt',
    #     pipeline=test_pipeline)
    )
