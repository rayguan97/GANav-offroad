_base_ = [
    '../_base_/models/ours_class_att.py', '../_base_/datasets/rugd_group6.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model=dict(
    decode_head=dict(
        num_classes=6),
    auxiliary_head=dict(num_classes=6))

data = dict(
    samples_per_gpu=4)
