# dataset settings
dataset_type = 'RUGDDataset'

source_data_root = '/l/scratch/khanmdal/RAL/Datasets/RUGD'
data_root = '/l/scratch/khanmdal/RAL/Datasets/mesh'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

source_input_size = (688, 550) # RUGD
target_input_size = (688, 550) # MESH

crop_size = (512,512)


rugd_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=source_input_size),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
mesh_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=target_input_size),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=target_input_size,
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='UDADataset',
        source=dict(
            type='RUGDDataset',
            data_root=source_data_root,
            img_dir='RUGD_images',
            ann_dir='RUGD_labels',
            pipeline=rugd_train_pipeline),
        target=dict(
            type='MESHDataset',
            data_root=data_root,
            img_dir='Temp/train',
            ann_dir=None,
            pipeline=mesh_train_pipeline)),
    
    test=dict(
        type='MESHDataset',
        data_root=data_root,
        img_dir='Temp/val',
        ann_dir=None,
        pipeline=test_pipeline))
