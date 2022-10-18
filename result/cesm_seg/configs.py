# coding=utf-8
import os, sys
import numpy as np

from default_configs import DefaultConfigs


class configs(DefaultConfigs):
    # ============ model setting ============
    model = 'segment/refinenet_4cascade'

    input_channel = 3
    num_seg_classes = 1
    diceloss_weight = 1
    dice_bg_open = False
    dice_size_avg = False

    input_size = (512, 1024)
    # data augment
    da_kwargs = {
        'do_elastic_deform': False,
        'alpha': (0., 1500.),
        'sigma': (30., 50.),
        'do_rotation': False,
        'angle_x': (0., 2 * np.pi),
        'angle_y': (0., 0),  # must be 0!!
        'angle_z': (0., 0),
        'do_scale': True,
        'scale': (0.9, 1.1),
        'random_crop': True,
        'random_flip': True,
    }

    # ============ I/O ============
    cls_csv_dir = ''
    data_path = ''

    # ============ train ============

    batch_size = 4
    n_workers = 1
    do_validation = True

    num_epochs = 50
    weight_decay = 5e-4
    best_model_min_save_thresh = 20
    save_n_best_models = 5
    save_model_per_epochs = 5
    # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
    weight_init = None
    # one of 'SGD' or 'Adam'
    optimizer = 'Adam'

    momentum = 0.9

    default_lr = 1e-3
    learning_rate = []
    for i in range(num_epochs):
        if i < num_epochs * 0.5:
            learning_rate.append(default_lr)
        elif i < num_epochs * 0.8:
            learning_rate.append(default_lr * 0.1)
        else:
            learning_rate.append(default_lr * 0.01)

    # ============ test ============
    test_batch_size = 2
    is_convert_to_world_coord = False

    def __init__(self):
        super(configs, self).__init__()
        pass
