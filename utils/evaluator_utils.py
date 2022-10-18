import numpy as np
import pandas as pd
import os
import time
import torch
from tqdm import tqdm
import SimpleITK as sitk
# from sklearn.metrics import roc_auc_score
from torchvision.utils import make_grid
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch.nn.functional as F
import csv
from skimage import data,filters
from utils.image_utils_2D import calc_dice
from torch.autograd import Variable


def update_tensorboard(metrics, epoch, tensorboard_writer, do_validation=True):
    for phase in metrics:
        if phase != 'val' or do_validation:
            # scalar_dict = {}
            for key in metrics[phase]:
                if len(metrics[phase][key]) > 0 and key != 'epoch':
                    # scalar_dict[key] = metrics[phase][key][-1]
                    tensorboard_writer.add_scalar(phase + '_' + key, metrics[phase][key][-1], epoch)
            # tensorboard_writer.add_scalars(phase, scalar_dict, epoch)


def update_tensorboard_image(feat_dict, global_step, tensorboard_writer):
    for i, key in enumerate(feat_dict):
        feat_map = torch.mean(feat_dict[key], dim=1)
        # 特征图按通道维度取均值
        feat_size = feat_map.size()
        slice_id = feat_size[1] // 2
        # 取特征图中间层(Z方向)进行可视化
        tensorboard_writer.add_image(key, make_grid(feat_map[:, slice_id, :, :],
                                                    padding=20, normalize=True,
                                                    scale_each=True, pad_value=1), global_step)


def analysis_train_output(outputs_dict, epoch_results_dict, phase='train'):
    """
    获取不同模型对应的log_string,按照网络输出填写result_dict
    :param outputs_dict:
    :param epoch_results_dict:
    :param phase:
    :return:
    """
    sum_tags = ['tp', 'tn', 'fp', 'fn']
    for key in outputs_dict:
        if key not in epoch_results_dict[phase]:
            epoch_results_dict[phase][key] = []
        if key not in sum_tags:
            epoch_results_dict[phase][key].append(outputs_dict[key].mean().item())
        else:
            epoch_results_dict[phase][key].append(outputs_dict[key].sum().item())

    exclude_tags = ['learning_rate', 'epoch'] + ['tp', 'tn', 'fp', 'fn']
    log_string = ' '.join([key + ': %0.2f' % epoch_results_dict[phase][key][-1] for key in epoch_results_dict[phase]
                           if key not in exclude_tags])

    loss = outputs_dict['loss'].mean()
    # loss = Variable(loss, requires_grad=True)
    return loss, log_string


def update_metrics(monitor_metrics, epoch_results_dict):
    # train
    exclude_tags = ['tp', 'tn', 'fp', 'fn']
    for phase in epoch_results_dict:
        for key in epoch_results_dict[phase]:
            if key not in monitor_metrics[phase]:
                monitor_metrics[phase][key] = []
            if key not in exclude_tags:
                monitor_metrics[phase][key].append(np.mean(epoch_results_dict[phase][key]))  # 保留均值
        if 'tp' in epoch_results_dict[phase] and 'tn' in epoch_results_dict[phase] and 'fn' in epoch_results_dict[
            phase] and 'fp' in epoch_results_dict[phase]:
            tp = np.sum(epoch_results_dict[phase]['tp'])
            tn = np.sum(epoch_results_dict[phase]['tn'])
            fn = np.sum(epoch_results_dict[phase]['fn'])
            fp = np.sum(epoch_results_dict[phase]['fp'])
            recall = float(tp) / float(tp + fn)
            precision = float(tp) / float(tp + fp)
            accuracy = float(tp + tn) / float(tp + fp + tn + fn)
            if 'recall' not in monitor_metrics[phase]:
                monitor_metrics[phase]['recall'] = []
            if 'precision' not in monitor_metrics[phase]:
                monitor_metrics[phase]['precision'] = []
            if 'accuracy' not in monitor_metrics[phase]:
                monitor_metrics[phase]['accuracy'] = []
            monitor_metrics[phase]['recall'].append(recall)
            monitor_metrics[phase]['precision'].append(precision)
            monitor_metrics[phase]['accuracy'].append(accuracy)


def write_csv(csv_path, content, mul=True, mod="a"):
    with open(csv_path, mod) as myfile:
        mywriter = csv.writer(myfile)
        if mul:
            mywriter.writerows(content)
        else:
            mywriter.writerow(content)


def analysis_cesm_seg(batch_dict, result_dict, output_dir):
    images = batch_dict['inputs']
    gt_mask = batch_dict['mask']
    infos = batch_dict['infos']

    predict_seg = result_dict['predict']
    predict_seg = F.upsample(input=predict_seg,
                             size=(gt_mask.size(2), gt_mask.size(3)),
                             mode='bicubic')

    # seg
    output_csv_path = os.path.join(output_dir, 'seg.csv')
    if not os.path.exists(output_csv_path):
        write_csv(output_csv_path, [['patientid','voi', 'dice']], mod='w')
    for i in range(images.size(0)):
        sid = infos['voi'][i]
        patientid = infos['patientid'][i]
        gt_mask = np.squeeze(gt_mask)##
        gt_mask_tmp = np.array(gt_mask.cpu())[i]
        predict_seg_tmp = np.array(F.sigmoid(predict_seg.cpu()))[i][0]

        thred = filters.threshold_otsu(predict_seg_tmp)
        predict_seg_tmp = predict_seg_tmp > thred

        predict_seg_tmp = (predict_seg_tmp + int(0)).astype(np.int16)
        predict_mask_sitk = sitk.GetImageFromArray(predict_seg_tmp)
        sitk.WriteImage(predict_mask_sitk,'/dd' + sid + '/'+patientid+ '.nii.gz' )
        dice = calc_dice(gt_mask_tmp!=0, predict_seg_tmp)

        write_csv(output_csv_path, [[patientid, sid, dice]], mod='a')

def analysis_cesm_cls(cf, batch_dict, result_dict, output_dir):

    images = batch_dict['inputs_LE0']
    label = batch_dict['label']
    patientid = batch_dict['patientid']
    patientid = (' ').join(patientid)
    predict_cls = result_dict['outputs']
    # class
    output_csv_path = os.path.join(output_dir, 'test_class.csv')
    if not os.path.exists(output_csv_path):
        write_csv(output_csv_path, [['patientid','gt0', 'gt1', 'prob0', 'prob1']], mod='w')
    for i in range(images.size(0)):
        gt_tmp = np.array(label.cpu())[i]
        if hasattr(cf, 'cls_output_map') and cf.cls_output_map == 'Sigmoid':
            prob_tmp = np.array(F.sigmoid(predict_cls).cpu())[i]
        else:
            prob_tmp = np.array(F.softmax(predict_cls, dim=1).cpu())[i]
        write_csv(output_csv_path, [[patientid,gt_tmp[0], gt_tmp[1], prob_tmp[0], prob_tmp[1]]], mod='a')

def analysis_test_output(cf, batch_dict, result_dict):
    if not os.path.exists(cf.output_files_dir):
        os.mkdir(cf.output_files_dir)
    output_dir = cf.output_files_dir

    if 'cesm_seg' == os.path.basename(cf.exp_source):
        analysis_cesm_seg(batch_dict, result_dict, output_dir)
    if 'cesm_cls' == os.path.basename(cf.exp_source):
        analysis_cesm_cls(cf,batch_dict, result_dict, output_dir)