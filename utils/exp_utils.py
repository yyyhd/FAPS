import os
import csv
import subprocess
import torch
import numpy as np
import importlib
from collections import OrderedDict
import time


def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prep_exp(exp_source, exp_result, use_stored_settings=False, is_train=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/testence of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :return:
    """

    cf_file = import_module('cf', os.path.join(exp_source, 'configs.py'))
    cf = cf_file.configs()
    if is_train:
        # the first process of an experiment creates the directories and copies the config to exp_path.
        if not os.path.exists(exp_result):
            os.makedirs(exp_result)
            subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'configs.py'), os.path.join(exp_result, 'configs.py')), shell=True)
            subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'dataset.py'), os.path.join(exp_result, 'dataset.py')), shell=True)
            subprocess.call('cp {} {}'.format('default_configs.py', os.path.join(exp_result, 'default_configs.py')), shell=True)
            subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(exp_result, 'model.py')), shell=True)
            # subprocess.call('cp {} {}'.format(cf.backbone_path, os.path.join(exp_result, 'backbone.py')), shell=True)

    cf.dataset_path = os.path.join(exp_source, 'dataset.py')
    if use_stored_settings:
        cf_file = import_module('cf', os.path.join(exp_result, 'configs.py'))
        cf = cf_file.configs()
        cf.model_path = os.path.join(exp_result, 'model.py')  # todo 暂时不使用拷贝的model，因为难以控制引用的backbone,想到更好的解决办法再改
        # cf.backbone_path = os.path.join(exp_result, 'backbone.py')
        cf.dataset_path = os.path.join(exp_result, 'dataset.py')

    data_name = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    cf.output_files_dir = os.path.join(exp_result, 'outputs_'+data_name)
    cf.select_model_dir = os.path.join(exp_result, 'select_model')
    if not os.path.exists(cf.select_model_dir):
        os.mkdir(cf.select_model_dir)
    cf.tensorboard_dir = os.path.join(exp_result, 'tensorboard')
    if not os.path.exists(cf.tensorboard_dir):
        os.mkdir(cf.tensorboard_dir)
    # cf.result_csv_path = os.path.join(exp_result, 'result_csv')

    cf.exp_result = exp_result
    cf.exp_source = exp_source
    return cf


def model_select_save(cf, net, optimizer, monitor_metrics, epoch):
    if torch.cuda.device_count() > 1:
        net_state_dict = net.module.state_dict()
    else:
        net_state_dict = net.state_dict()

    if cf.do_validation:
        metrics = monitor_metrics['val']
    else:
        metrics = monitor_metrics['train']

    val_loss = metrics['loss']
    index_ranking = np.argsort(val_loss)
    epoch_ranking = np.array(monitor_metrics['train']['epoch'])[index_ranking]
    epoch_ranking = epoch_ranking[epoch_ranking >= cf.best_model_min_save_thresh]
    
    # check if current epoch is among the top-k epchs.
    if epoch in epoch_ranking[:cf.save_n_best_models]:
        # 更新为最佳模型地址
        torch.save(net.state_dict(), os.path.join(cf.select_model_dir, 'model_%03d.pth' % epoch))
        # np.save(os.path.join(cf.select_model_dir, 'epoch_ranking'), epoch_ranking[:cf.save_n_best_models])
        np.savetxt(os.path.join(cf.select_model_dir, 'epoch_ranking.txt'), epoch_ranking[:cf.save_n_best_models], fmt='%03d')

        # delete params of the epoch that just fell out of the top-k epochs.
        if len(epoch_ranking) > cf.save_n_best_models:
            epoch_rm = epoch_ranking[cf.save_n_best_models]
            if not epoch_rm % cf.save_model_per_epochs == 0:
                subprocess.call('rm {}'.format(os.path.join(cf.select_model_dir, 'model_%03d.pth' % epoch_rm)), shell=True)

    if epoch % cf.save_model_per_epochs == 0 or epoch == cf.num_epochs:
        torch.save(net_state_dict, os.path.join(cf.select_model_dir, 'model_%03d.pth' % epoch))

    state = {
        'epoch': epoch,
        'metrics': monitor_metrics,
        'state_dict': net_state_dict,
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(cf.select_model_dir, 'last_state.pth'))


def get_clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        print(name)
    return new_state_dict


def load_checkpoint(checkpoint_path, net, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['metrics']


def load_pretrain_model(net, weight_path, cf):
    net_state_dict = net.state_dict()
    # load_state_dict = get_clean_state_dict(torch.load(weight_path)['state_dict'])
    load_dict = torch.load(weight_path)
    if 'state_dict' in load_dict:
        load_dict = load_dict['state_dict']
    load_state_dict = get_clean_state_dict(load_dict)
    pretrain_dict = {k: v for k, v in load_state_dict.items() if
                     (k in net_state_dict.keys() and k not in cf.exclude_pretrain_names)}

    net_state_dict.update(pretrain_dict)
    net.load_state_dict(net_state_dict)


def load_test_checkpoint(net, weight_path):
    checkpoint = torch.load(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    net.load_state_dict(get_clean_state_dict(state_dict))
    net.eval()


def prepare_monitoring(cf):
    """
    creates dictionaries, where train/val metrics are stored.
    """
    # metrics = {}
    # # first entry for loss dict accounts for epoch starting at 1.
    # metrics['train'] = {'loss':[], 'class_loss': [], 'regress_loss': []}
    # metrics['val'] = {'loss':[], 'class_loss': [], 'regress_loss': []}

    metrics = {
        'train': {'loss': []},
        'val': {'loss': []}
               }

    return metrics


def write_csv(csv_name, content, mul=True, mod="w"):
    with open(csv_name, mod) as myfile:
        mywriter = csv.writer(myfile)
        if mul:
            mywriter.writerows(content)
        else:
            mywriter.writerow(content)
