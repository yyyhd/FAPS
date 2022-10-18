# encoding=utf-8
import argparse
import os
import time
import math
import torch
import numpy  as np
from torch import optim
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import py3nvml
import pandas as pd
from utils.logger_utils import get_logger
import utils.exp_utils as utils
import utils.evaluator_utils as eval_utils
import utils.image_utils as dutils
from sklearn.model_selection import train_test_split
from apex import amp
from torch.utils.data.sampler import WeightedRandomSampler

import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def train(logger, cf, model, dataset):
    logger.info("performing training with model {}".format(cf.model))

    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


    net = model.Net4(cf, logger).cuda()#2

    if cf.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay,
                              momentum=cf.momentum)
    elif cf.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)

    # prepare monitoring
    monitor_metrics = utils.prepare_monitoring(cf)

    if cf.use_pretrain_model:
        utils.load_pretrain_model(net, cf.transfer_learning_weight_path, cf)


    #AMP training
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    # 多GPU并行训练
    net = DataParallel(net)

    starting_epoch = 1
    if cf.resume_to_checkpoint is not False:
        resume_epoch, monitor_metrics = utils.load_checkpoint(cf.resume_to_checkpoint, net, optimizer)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, resume_epoch))
        starting_epoch = resume_epoch + 1

    # add this , can improve the train speed
    torch.backends.cudnn.benchmark = True

    logger.info('loading dataset and initializing batch generators...')

    data_file = dataset.DataCustom(cf, logger, phase="train")
    val_file = dataset.DataCustom(cf, logger, phase="val")
    target = []
    for i in data_file.data_paths_list:
        file = np.load(i)
        if file["label"] == 1:
            target.append(1)
        if file["label"] == 0:
            target.append(0)

    target = np.array(target)
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    target = torch.from_numpy(target).long()

    dataloaders = {phase: DataLoader(dataset.DataCustom(cf, logger, phase=phase),
                                     batch_size=cf.batch_size,
                                     shuffle={'train': False, 'val': False}[phase],
                                     sampler={'train': sampler, 'val': None}[phase],
                                     num_workers=cf.n_workers,
                                     pin_memory=True,
                                     drop_last=True)
                   for phase in ['train', 'val']}


    tensorboard_writer = SummaryWriter(cf.tensorboard_dir)

    for epoch in range(starting_epoch, cf.num_epochs + 1):
        epoch_start_time = toc_train = time.time()
        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        net.train()
        epoch_results_dict = {'train': {'learning_rate': [cf.learning_rate[epoch - 1]], 'epoch': [epoch]},
                              'val': {}}
        for batchidx, dataset_outputs in enumerate(dataloaders['train']):
            tic_fw = time.time()
            # run model
            batch_outputs = net(dataset_outputs['inputs_LE0'], dataset_outputs['inputs_RE0'], dataset_outputs['inputs_LE1'], dataset_outputs['inputs_RE1'],dataset_outputs['label'],phase='train')
            loss, log_string = eval_utils.analysis_train_output(batch_outputs, epoch_results_dict, 'train')

            tic_bw = time.time()
            optimizer.zero_grad()

            #AMP
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # loss.backward()
            optimizer.step()

            logger.info(
                'tr. batch {0}/{1} (ep. {2}) dl {3: .3f}/ fw {4:.3f}s / bw {5:.3f}s / total {6:.3f}s / lr {7} || {8}'.format(
                    batchidx + 1, len(dataloaders['train']), epoch, tic_fw - toc_train, tic_bw - tic_fw,
                    time.time() - tic_bw,
                    time.time() - tic_fw, optimizer.param_groups[-1]['lr'], log_string))

            torch.cuda.empty_cache()
            toc_train = time.time()

        train_time = time.time() - epoch_start_time
        with torch.no_grad():
            if cf.do_validation:
                logger.info("starting valiadation.")

                net.eval()
                for batchidx, dataset_outputs in tqdm(enumerate(dataloaders['val'])):
                    outputs = net(dataset_outputs['inputs_LE0'], dataset_outputs['inputs_RE0'],
                                        dataset_outputs['inputs_LE1'], dataset_outputs['inputs_RE1'],
                                        dataset_outputs['label'], phase='val')

                    eval_utils.analysis_train_output(outputs, epoch_results_dict, phase='val')
                    torch.cuda.empty_cache()

        # update monitoring and prediction plots
        eval_utils.update_metrics(monitor_metrics, epoch_results_dict)
        eval_utils.update_tensorboard(monitor_metrics, epoch, tensorboard_writer, cf.do_validation)
        utils.model_select_save(cf, net, optimizer, monitor_metrics, epoch)

        epoch_time = time.time() - epoch_start_time
        logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(epoch, epoch_time, train_time,
                                                                                epoch_time - train_time))

    tensorboard_writer.close()


def test(logger, cf, model, dataset):
    logger.info("performing testence with model {}".format(cf.model))

    net = model.Net4(cf, logger)

    try:
        utils.load_test_checkpoint(net, cf.resume_to_checkpoint)
        logger.info('resumed to checkpoint {}'.format(cf.resume_to_checkpoint))
    except Exception as e:
        logger.error('load checkpoint error! %s' % e)
        return None

    net = DataParallel(net)
    net.cuda()

    dataset = dataset.DataCustom(cf, logger, phase='test')

    test_data_loader = DataLoader(dataset,
               batch_size=cf.test_batch_size,
               shuffle=False,
               num_workers=cf.n_workers,
               pin_memory=True)
    with torch.no_grad():
        for index, dataset_outputs in tqdm(enumerate(test_data_loader)):

            result_dict = net(dataset_outputs['inputs_LE0'], dataset_outputs['inputs_RE0'],
                                dataset_outputs['inputs_LE1'], dataset_outputs['inputs_RE1'],
                               phase='test')

            eval_utils.analysis_test_output(cf, dataset_outputs, result_dict)
            torch.cuda.empty_cache()

    # eval_utils.save_test_csv(cf, dict_output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='train',
                        help='one out of : train / test / test')
    parser.add_argument('--exp_source', type=str, default='./experiments/cesm_cls',)
    parser.add_argument('--exp_result', type=str, default='./result/cesm_cls',)
    parser.add_argument('--resume_to_checkpoint', action='store_true', default= False,
                        help='False:不加载； True：加载last_state.pht； 路径：加载指定路径.')
    parser.add_argument('--use_stored_settings', action='store_true', default=True,
                        help='load configs from existing stored_source instead of exp_source. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='number of using gpu.')
    parser.add_argument('--use_multi_cpu', '-mc', action='store_false', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train' or args.mode == 'train_test':
        cf = utils.prep_exp(args.exp_source, args.exp_result, args.use_stored_settings, is_train=True)
        if not args.use_multi_cpu:
            cf.n_workers = 0

        model = utils.import_module('model', cf.model_path)
        dataset = utils.import_module('dataset', cf.dataset_path)
        if args.resume_to_checkpoint is True:
            cf.resume_to_checkpoint = os.path.join(cf.select_model_dir, 'last_state.pth')
        else:
            cf.resume_to_checkpoint = args.resume_to_checkpoint

        logger = get_logger(cf.exp_result)
        train(logger, cf, model, dataset)
        cf.resume_to_checkpoint = False

        if args.mode == 'train_test':
            if not os.path.exists(os.path.dirname(cf.result_csv_path)):
                os.makedirs(os.path.dirname(cf.result_csv_path))
            test(logger, cf, model, dataset)

    elif args.mode == 'test':
        cf = utils.prep_exp(args.exp_source, args.exp_result, args.use_stored_settings, is_train=False)
        model = utils.import_module('model', cf.model_path)
        dataset = utils.import_module('dataset', cf.dataset_path)

        logger = get_logger(cf.exp_result)
        if isinstance(args.resume_to_checkpoint, str):
            cf.resume_to_checkpoint = args.resume_to_checkpoint
        else:
            epoch_ranking = np.loadtxt(os.path.join(cf.select_model_dir, 'epoch_ranking.txt'), dtype=int)[0]
            cf.resume_to_checkpoint = os.path.join(cf.select_model_dir, 'model_%03d.pth' % epoch_ranking)
        if not os.path.exists(cf.resume_to_checkpoint):
            raise ValueError('checkpoint path must exist, when testence.')

        test(logger, cf, model, dataset)



