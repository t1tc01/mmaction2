_base_ = '../_base_/default_runtime.py'

# 1. MODEL CONFIG
url = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8, 
        speed_ratio=8, 
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d', depth=50, lateral=True, conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1), conv1_stride_t=1, pool1_stride_t=1,
            inflate=(0, 0, 1, 1), spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d', depth=50, lateral=False, base_channels=8, 
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1, pool1_stride_t=1, spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D', roi_layer_type='RoIAlign',
            output_size=8, with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA', 
            background_class=False, # Vì ID của bạn từ 1-7 (không có nền)
            in_channels=2304, 
            num_classes=7, 
            multilabel=True, 
            dropout_ratio=0.5, 
            topk=())), # Chỉ tính top-1 vì bạn có ít lớp
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(type='MaxIoUAssignerAVA', pos_iou_thr=0.9, neg_iou_thr=0.9, min_pos_iou=0.9),
            sampler=dict(type='RandomSampler', num=32, pos_fraction=1, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=dict(action_thr=0.002)))

# 2. DATA PATHS
dataset_type = 'AVADataset'
data_root = 'data/human_pose_dataset/rawframes'
anno_root = 'data/human_pose_dataset/annotations'
ann_file_train = f'{anno_root}/train.csv'
ann_file_val = f'{anno_root}/val.csv'
label_file = f'{anno_root}/label_map.pbtxt'

proposal_file_train = f'{anno_root}/proposals_train.pkl'
proposal_file_val = f'{anno_root}/proposals_val.pkl' 

# 3. PIPELINES
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

# 4. DATALOADERS
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        num_classes=7,
        proposal_file=proposal_file_train,
        filename_tmpl='frame_{:05d}.jpg',
        start_index=1, # ĐÃ SỬA: Vì file của bạn là frame_00001.jpg
        data_prefix=dict(img=data_root)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        filename_tmpl='frame_{:05d}.jpg',
        start_index=1, # ĐÃ SỬA: Đồng bộ với tập train
        data_prefix=dict(img=data_root),
        test_mode=True))

test_dataloader = val_dataloader

# 5. EVALUATOR & RUNTIME
val_evaluator = dict(
    type='AVAMetric', 
    ann_file=ann_file_val, 
    label_file=label_file,
    exclude_file=None, 
    num_classes=7)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='MultiStepLR', begin=0, end=20, by_epoch=True, milestones=[10, 15], gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=40, norm_type=2))