# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='FastSCNN',
        downsample_dw_channels=(32, 48),
        global_in_channels=64,
        global_block_channels=(64, 96, 128),
        global_block_strides=(1, 2, 1),
        global_out_channels=128,
        higher_in_channels=64,
        lower_in_channels=128,
        fusion_out_channels=128,
        out_indices=(0, 1, 2),
        norm_cfg=norm_cfg,
        align_corners=False),
    decode_head=dict(
        type='OursHeadClassAtt',
        in_channels=[64, 128, 128],
        in_index=[0, 1, 2],
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
        norm_cfg=norm_cfg,
        align_corners=False,
        attn_split=1,
        strides=(2,2,1),
        size_index=1,
        img_size=(304, 384),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, static_weight=False)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=32,
            num_convs=1,
            num_classes=6,
            in_index=-1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            img_size=(375, 600),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=32,
            num_convs=1,
            num_classes=6,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            img_size=(375, 600),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))