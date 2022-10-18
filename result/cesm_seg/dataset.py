'''
Dataset for training and testing
'''

import numpy as np
from torch.utils.data import Dataset
import glob
import os, sys
import SimpleITK as sitk
from scipy.ndimage import zoom
import pandas as pd
import matplotlib.pyplot as plt

# from utils.image_utils import normalize_image, CreateTransformer
from utils.image_utils_2D import normalize_image, CreateTransformer
from utils.image_utils import read_dcm_series, image_normalize_pretrain


class DataCustom(Dataset):

    def __init__(self, cf, logger=None, phase='train'):
        assert phase in ['train', 'val', 'test'], "phase must be one of 'train', 'val' and 'test'."
        if logger is None:
            logger = NoneLog()
        self.phase = phase
        self.logger = logger
        self.cf = cf
        self.shape_train = cf.input_size

        if phase != 'test':
            data_paths_list = []
            cls_csv_dir = os.path.join(cf.cls_csv_dir, phase)
            csv_paths = glob.glob(os.path.join(cls_csv_dir, '*.csv'))
            df_cls = pd.DataFrame()
            for csv_path in csv_paths:
                df_cls = df_cls.append(pd.read_csv(csv_path))
            data_paths = cf.data_path
            for idx, item in df_cls.iterrows():
                data_path = os.path.join(data_paths, item[0])
                if os.path.exists(data_path):
                    data_paths_list.append(data_path)

            if cf.black_list:
                black_list = pd.read_csv(cf.black_list)['seriesuid'].tolist()
                df_cls = df_cls[[True if sid not in black_list else False for sid in df_cls['seriesuid'].tolist()]]
            # logger.info(phase + ' label0 num: %s, label1 num: %s, label2 num: %s' % (sum(df_cls['label'] == 0), sum(df_cls['label'] == 1), sum(df_cls['label'] == 2)))
            self.data_paths_list = data_paths_list

        if phase == 'test':
            data_paths_list = []

            cls_csv_dir = os.path.join(cf.cls_csv_dir, phase)
            csv_paths = glob.glob(os.path.join(cls_csv_dir, '*.csv'))
            df_cls = pd.DataFrame()
            for csv_path in csv_paths:
                df_cls = df_cls.append(pd.read_csv(csv_path))
            data_paths = cf.data_path
            for idx, item in df_cls.iterrows():
                data_path = os.path.join(data_paths, item[0])
                if os.path.exists(data_path):
                    data_paths_list.append(data_path)
            self.data_paths_list = data_paths_list

    def __len__(self):
        return len(self.data_paths_list)

    def __getitem__(self, index):
        if self.phase != 'test':
            data = np.load(self.data_paths_list[index])
            img, mask = self.__preprocess(data)
            if self.cf.num_seg_classes > 1:
                img_shape = img.shape
                output_mask_arr = np.zeros([self.cf.num_seg_classes, img_shape[0], img_shape[1], img_shape[2]], dtype=np.int64)
                for i in range(self.cf.num_seg_classes):
                    output_mask_arr[i] = mask == i
                mask = output_mask_arr.astype(np.int64)
            else:
                mask = mask[np.newaxis].astype(np.float32)

            return {'inputs': img, 'mask': mask}
        else:
            data = np.load(self.data_paths_list[index])
            img, mask = self.__preprocess(data)
            img_shape = img.shape
            if self.cf.num_seg_classes > 1:
                output_mask_arr = np.zeros([self.cf.num_seg_classes, img_shape[0], img_shape[1], img_shape[2]], dtype=np.int64)
                for i in range(self.cf.num_seg_classes):
                    output_mask_arr[i] = mask == i
                mask = output_mask_arr.astype(np.int64)
            else:
                mask = mask[np.newaxis].astype(np.float32)

            sid = os.path.basename(self.data_paths_list[index]).replace('.npz', '')
            id = os.path.basename(os.path.dirname(self.data_paths_list[index]))
            seriesuid = str(id) + '_' + sid
            # origin_shape_zyx = np.ascontiguousarray(np.array(data['image_zyx'].shape))
            infos = {
                'patientid': sid,
                'voi': id,
                'image_path': self.data_paths_list[index],
            }
            return {'inputs': img, 'mask': mask, 'infos': infos}


    def __preprocess(self, data):
        img_LE, img_RE, mask = data['LE'], data['RE'], data['lesion']
        img_LE = np.array(img_LE, dtype=np.float32)
        img_RE = np.array(img_RE, dtype=np.float32)
        mask = np.array(mask, dtype=np.int32)
        if self.phase != 'test':
            img_LE, img_RE, mask = self.__aug(img_LE,img_RE, mask)
        image = np.stack((img_LE, img_RE, img_LE))
        mask[mask > 0] = 1####0, 1

        return image, mask

    def __aug(self, img1, img2,mask):
        if self.phase == 'train':
            transformer = CreateTransformer(self.cf, random_scale=self.cf.da_kwargs['do_scale'],
                                            random_crop=self.cf.da_kwargs['random_crop'],
                                            random_flip=self.cf.da_kwargs['random_flip'])
        elif self.phase == 'val':
            transformer = CreateTransformer(self.cf, random_scale=False,
                                            random_crop=False,
                                            random_flip=False)

        image1, image2, mask = transformer.image_transform_with_bbox(img1,img2, mask=mask, pad_value=0,
                                                            centerd=None, bbox=None)
        return image1, image2, mask


if __name__=='__main__':
    from torch.utils.data import DataLoader
    import time

    from experiments.pneumonia_seg.configs import configs as cf
    dataset = DataCustom(cf, phase='train')
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=False)

    tic = time.time()
    for i, item in enumerate(train_loader):
        a=1
        pass
        toc=time.time()
        print(toc-tic)
        tic=time.time()