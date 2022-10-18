#coding=utf-8
import os


class DefaultConfigs:

    # ============ model setting ============
    model = ''
    batch_size = 1
    n_workers = 2
    do_validation = False
    pretrain_path = ''

    # ============ I/O ============
    train_data_root = ''
    test_data_root = ''

    mask_data_root = ''

    train_csv_path = ''
    val_csv_path = ''
    test_csv_path = ''

    # ============ train ============
    num_epochs = 100
    weight_decay = 1e-4
    save_n_best_models = 5
    save_model_per_epochs = 5
    best_model_min_save_thresh = 20
    # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
    weight_init = None
    # one of 'SGD' or 'Adam'
    optimizer = 'SGD'
    momentum = 0
    use_pretrain_model = False
    transfer_learning_weight_path = ''
    exclude_pretrain_names = []

    # ============ test ============
    test_batch_size = 2
    is_convert_to_world_coord = False

    def __init__(self):
        """Set values of computed attributes."""
        # model_names = os.listdir('./models')
        # assert self.model+'.py' in model_names, "Specified model don't exist in models."
        self.model_path = 'models/{}.py'.format(self.model)
