_base_ = [
    '../../_base_/default_runtime.py',
     '../../_base_/recog_models/crnn.py',
    '../../_base_/recog_pipelines/crnn_pipeline.py',
    # '../../_base_/recog_datasets/MJ_train.py',
    # '../../_base_/recog_datasets/academic_test.py',
    # '../../_base_/schedules/schedule_adadelta_18e.py'
    '../../_base_/schedules/schedule_sgd_160e.py'
]

img_prefix = 'ocr_data/ocr_dataSet'
train_ann_file = 'ocr_data/annoDir/ImageSet/ocr_train.txt'
val_ann_file = 'ocr_data/annoDir/ImageSet/ocr_val.txt'
dict_file = 'ocr_data/ocr_dataSet/dict.txt'

label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True, dict_file=dict_file)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True, num_classes=10),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)


train = dict(
    type='OCRDataset',
    img_prefix=img_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineStrParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

test = dict(
    type='OCRDataset',
    img_prefix=img_prefix,
    ann_file=val_ann_file,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

train_list = [train]
test_list = [test]

# train_list = {{_base_.train_list}}
# test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')

cudnn_benchmark = True
