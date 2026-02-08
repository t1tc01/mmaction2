# _base_ = ['../../_base_/default_runtime.py']
_base_ = ['../_base_/default_runtime.py']

url = (
    'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/'
    'vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True, 
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_path_rate=0.2,
        use_mean_pooling=False,
        return_feat_map=True),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            background_class=True,
            in_channels=768,
            num_classes=7,
            topk=(),
            multilabel=True,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        _scope_='mmaction',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVADataset'
data_root = 'data/human_pose_dataset/rawframes'
anno_root = 'data/human_pose_dataset/annotations'
ann_file_train = f'{anno_root}/train.csv'
ann_file_val = f'{anno_root}/val.csv'
label_file = f'{anno_root}/label_map.pbtxt'

proposal_file_train = f'{anno_root}/proposals_train.pkl'
proposal_file_val = f'{anno_root}/proposals_val.pkl' 

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=16, frame_interval=4),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=16, frame_interval=4, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=None,
        pipeline=train_pipeline,
        label_file=label_file,
        num_classes=7,
        filename_tmpl='frame_{:05d}.jpg',
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root)))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=None,
        pipeline=val_pipeline,
        label_file=label_file,
        num_classes=7,
        filename_tmpl='frame_{:05d}.jpg',
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    num_classes=7,
    exclude_file=None)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=0,
        by_epoch=True,
        begin=5,
        end=20,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=(1.25e-4)/8, weight_decay=0.05),
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.75,
        'decay_type': 'layer_wise',
        'num_layers': 12
    },
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
