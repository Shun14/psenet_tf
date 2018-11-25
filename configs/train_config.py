TRIAN_CONFIG = {
    'batch_size': 16,
    'number_kernel_scales':3,
    'num_preprocessing_threads':4,
    'minimal_scale_ratio':0.5,
    'num_classes':2,
    'net_name':'resnet_v1_101',
    'train_scale':640,
    'scale_ratio':[0.5, 1.0, 2.0 , 3.0],
    'rotated_angle':[-10, 10],
    'OHEM':3,
    'MOVING_AVERAGE_DECAY':0.99,
    'learning_rate': 1e-3,
    'step_boundaries':[100, 200],
    'dataset_format':{
        'H':'horizontal',
        'P':'quadrilateral',
        'C':'curved'
    }
}
