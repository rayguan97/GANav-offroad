_base_ = [
    '../_base_/models/ours_class_att.py', '../_base_/datasets/goose_group6.py',
    '../_base_/default_runtime.py'
]


optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict()
# learning policy
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=240000)
total_iters = 240000
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=240000, metric='mIoU')

# optimizer
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    by_epoch=False)

model=dict(
    decode_head=dict(img_size=(375, 600)))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)
