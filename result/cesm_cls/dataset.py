# encoding=utf-8
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import cv2
from PIL import Image
import time
import random
import glob
import torch
import deepdish as dd
import matplotlib.pyplot as plt
# from utils.image_utils import normalize_image, CreateTransformer
from utils.image_utils_2D import normalize_image, CreateTransformer
from utils.logger_utils import NoneLog



def get_lung_offset(lungmask_arr, spacing_zyx, extend_mm=16):
    lung_mask_l = lungmask_arr == 1
    lung_mask_r = lungmask_arr == 2
    lung_mask_b = (lung_mask_l + lung_mask_r) > 0
    zz, yy, xx = np.where(lung_mask_b)

    # extend 16 mm.
    extend_zyx = np.round(extend_mm / np.array(spacing_zyx)).astype(int)
    zz, yy, xx = np.array(zz), np.array(yy), np.array(xx)
    voi_zz = [max(0, zz.min() - extend_zyx[0]), zz.max() + extend_zyx[0] + 1]
    voi_yy = [max(0, yy.min() - extend_zyx[1]), yy.max() + extend_zyx[1] + 1]
    voi_xx = [max(0, xx.min() - extend_zyx[2]), xx.max() + extend_zyx[2] + 1]

    recommed_offset_zyxzyx = [voi_zz[0], voi_yy[0], voi_xx[0], voi_zz[1], voi_yy[1], voi_xx[1]]
    return recommed_offset_zyxzyx


class DataCustom(Dataset):

    def __init__(self, cf, logger=None, phase='train'):
        assert phase in ['train', 'val', 'test'], "phase must be one of 'train', 'val' and 'test'."
        if logger is None:
            logger = NoneLog()
        self.phase = phase
        self.cf = cf
        self.input_size = cf.input_size

        if phase != 'test':
            data_paths_list0 = []
            data_paths_list1 = []
            cls_csv_dir = os.path.join(cf.cls_csv_dir, phase)
            csv_paths = glob.glob(os.path.join(cls_csv_dir, '*.csv'))
            df_cls = pd.DataFrame()
            for csv_path in csv_paths:
                df_cls = df_cls.append(pd.read_csv(csv_path))
            data_paths = cf.data_path1
            for idx, item in df_cls.iterrows():
                data_path0 = os.path.join(data_paths, item[0])
                data_path1 = os.path.join(data_paths, item[1])
                if os.path.exists(data_path0):
                    data_paths_list0.append(data_path0)
                if os.path.exists(data_path1):
                    data_paths_list1.append(data_path1)
            # data_paths_list = list(zip(data_paths_list0,  data_paths_list1,data_paths_list2)
            if cf.black_list:
                black_list = pd.read_csv(cf.black_list)['seriesuid'].tolist()
                df_cls = df_cls[[True if sid not in black_list else False for sid in df_cls['seriesuid'].tolist()]]
            # logger.info(phase + ' label0 num: %s, label1 num: %s, label2 num: %s' % (sum(df_cls['label'] == 0), sum(df_cls['label'] == 1), sum(df_cls['label'] == 2)))
            self.data_paths_list0 = data_paths_list0
            self.data_paths_list1 = data_paths_list1

        if phase == 'test':
            data_paths_list0 = []
            data_paths_list1 = []
            cls_csv_dir = os.path.join(cf.cls_csv_dir, phase)
            csv_paths = glob.glob(os.path.join(cls_csv_dir, '*.csv'))
            df_cls = pd.DataFrame()
            for csv_path in csv_paths:
                df_cls = df_cls.append(pd.read_csv(csv_path))
            data_paths = cf.data_path2
            for idx, item in df_cls.iterrows():
                data_path0 = os.path.join(data_paths, item[0])
                data_path1 = os.path.join(data_paths, item[1])
                if os.path.exists(data_path0):
                    data_paths_list0.append(data_path0)
                if os.path.exists(data_path1):
                    data_paths_list1.append(data_path1)
            self.data_paths_list0 = data_paths_list0
            self.data_paths_list1 = data_paths_list1

    def __len__(self):
        return len(self.data_paths_list0)

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))

        if self.phase != 'test':

                h5_path0 = self.data_paths_list0[idx]
                h5_path1 = self.data_paths_list1[idx]
                h5_file0 = np.load(h5_path0, allow_pickle=True)
                h5_file1 = np.load(h5_path1, allow_pickle=True)

                if h5_file0["label"] == 0:
                    label = np.array([1, 0])
                if h5_file0["label"] == 1:
                    label = np.array([0, 1])

                img_LE0, img_RE0 = self.__preprocess(h5_file0)
                img_LE0 = img_LE0.astype(np.float32)
                img_RE0 = img_RE0.astype(np.float32)

                img_LE1, img_RE1 = self.__preprocess(h5_file1)
                img_LE1 = img_LE1.astype(np.float32)
                img_RE1 = img_RE1.astype(np.float32)
                patientid = h5_file0['patientid'].tolist()
                return {'inputs_LE0': img_LE0,'inputs_RE0': img_RE0, 'inputs_LE1': img_LE1,'inputs_RE1': img_RE1,'label': label, 'patientid': patientid}

        else:

            h5_path0 = self.data_paths_list0[idx]
            h5_path1 = self.data_paths_list1[idx]
            h5_file0 = np.load(h5_path0, allow_pickle=True)
            h5_file1 = np.load(h5_path1, allow_pickle=True)

            if h5_file0["label"] == 0:
                label = np.array([1, 0])
            if h5_file0["label"] == 1:
                label = np.array([0, 1])


            img_LE0, img_RE0 = self.__preprocess(h5_file0)
            img_LE0 = img_LE0.astype(np.float32)
            img_RE0 = img_RE0.astype(np.float32)

            img_LE1, img_RE1 = self.__preprocess(h5_file1)
            img_LE1 = img_LE1.astype(np.float32)
            img_RE1 = img_RE1.astype(np.float32)
            patientid = h5_file0['patientid'].tolist()
            return {'inputs_LE0': img_LE0, 'inputs_RE0': img_RE0, 'inputs_LE1': img_LE1, 'inputs_RE1': img_RE1,
                    'label': label, 'patientid': patientid}

    def __preprocess(self, data):
        img_LE, img_RE, mask = data['new_LE'], data['new_RE'], data['new_lesion']
        img_LE = np.array(img_LE, dtype=np.float32)
        img_RE = np.array(img_RE, dtype=np.float32)

        if self.phase != 'test':
            img_LE, img_RE = self.__aug(img_LE,img_RE)

        img_LE = np.stack((img_LE, img_LE, img_LE))
        img_RE = np.stack((img_RE, img_RE, img_RE))
        return img_LE, img_RE

    def __aug(self, img1, img2):

        if self.phase == 'train':
            transformer = CreateTransformer(self.cf, random_scale=self.cf.da_kwargs['do_scale'],
                                            random_crop=self.cf.da_kwargs['random_crop'],
                                            random_flip=self.cf.da_kwargs['random_flip'])
        elif self.phase == 'val':
            transformer = CreateTransformer(self.cf, random_scale=False,
                                            random_crop=False,
                                            random_flip=False)

        image1, image2 = transformer.image_transform_with_bbox(img1,img2, mask=None, pad_value=0,
                                                            centerd=None, bbox=None)


        return image1, image2


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    from experiments.pneumonia_seg_cls.configs import configs as cf

    dataset = DataCustom(cf, phase='test')
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=False)

    tic = time.time()
    for i, item in enumerate(train_loader):
        a = 1
        pass
        toc = time.time()
        print(toc - tic)
        tic = time.time()
