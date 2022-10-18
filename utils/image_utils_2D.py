# coding=utf-8
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import pydicom
import scipy.ndimage as nd
import os, glob

EXCLUDED_EXTEN = ["jpg", "doc", "docx", "txt", 'xml', "png", "xlsx", "xls", "csv", "md"]


def get_roots(data_dir, extension=None):
    excluded_exten = [".jpg", ".doc", ".docx", ".txt", '.xml', ".png", ".xlsx", ".xls", ".csv", ".md", 'nrrd', 'zip',
                      'gz']
    roots = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if extension is not None:
                if fn.rpartition('.')[-1] != extension:
                    continue
            else:
                if os.path.splitext(fn)[-1] in excluded_exten:
                    continue
            roots.append(root)
    roots = list(set(roots))
    return roots


def read_dcm_series(dcm_dir, logger=None):
    '''
    此函数已考虑dcm中阶矩与斜率的tag
    输出的矩阵已按照世界坐标的顺序排列
    :param dcm_path:
    :param logger:
    :return: sitk格式图像; series_id
    '''
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_dir)
    nb_series = len(series_IDs)
    if nb_series > 1:
        if logger is not None:
            logger.info('nb_series > 1, series ids: %s' % series_IDs)
        else:
            print('nb_series > 1, series ids: %s' % series_IDs)
    elif nb_series == 0:
        series_ID = os.path.basename(dcm_dir)
        series_IDs = [series_ID]
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir)
    if len(series_file_names) == 0:
        series_file_names = glob.glob(os.path.join(dcm_dir, '*.dcm'))
        if len(series_file_names) == 0:
            series_file_names = os.listdir(dcm_dir)
            series_file_names = [os.path.join(dcm_dir, name) for name in series_file_names if
                                 name.rpartition('.')[-1] not in EXCLUDED_EXTEN]
    slices = [pydicom.read_file(s, stop_before_pixels=True) for s in series_file_names]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    series_file_names = [slice.filename for slice in slices]
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image_sitk = series_reader.Execute()
    return image_sitk, series_IDs[0]


def pad_nd_image(image, new_shape=None, mode="edge", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit. by Fabian Isensee

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array(
            [new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in
             range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]] * num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])
    res = np.pad(image, pad_list, mode, **kwargs)
    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


# def image_normalize_pretrain(image, minimum=-1200, maximum=600):
#     voxels = np.reshape(image, -1)
#     percentile_99_5 = np.percentile(voxels, 99.5)
#     percentile_00_5 = np.percentile(voxels, 00.5)
#
#     image = np.clip(image, percentile_00_5, percentile_99_5)
#     image = np.clip(image, minimum, maximum)
#     image = image.astype(np.float32)
#
#     mean = np.mean(image)
#     std = np.std(image)
#     image = (image - mean) / (std + 1e-5)
#     return image


def normalize_image(image, clip_window=(40.0, 350.0), output_range=None, ntype='normalize'):
    if ntype == 'normalize':
        image = np.clip(image, clip_window[0], clip_window[1])
        if output_range is None:
            image = (image - clip_window[0]) / float(clip_window[1] - clip_window[0])
        else:
            image = (image - clip_window[0]) / (
                    float(clip_window[1] - clip_window[0]) / float(output_range[1] - output_range[0])) + \
                    output_range[0]
        return image
    elif ntype == 'standardize':
        image = np.clip(image, clip_window[0], clip_window[1])
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / (std + 1e-5)
        return image
    elif ntype == 'CT':
        mean_intensity = image.mean()
        std_intensity = image.std()
        voxels = np.reshape(image, -1)
        upper_bound = np.percentile(voxels, 99.5)
        lower_bound = np.percentile(voxels, 00.5)
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / std_intensity
        return image
    elif ntype == 'CT2':
        voxels = np.reshape(image, -1)
        upper_bound = np.percentile(voxels, 99.5)
        lower_bound = np.percentile(voxels, 00.5)
        mask = (image > lower_bound) & (image < upper_bound)
        image = np.clip(image, lower_bound, upper_bound)
        mn = image[mask].mean()
        sd = image[mask].std()
        image = (image - mn) / sd
        return image


class CreateTransformer(object):
    def __init__(self, cf, random_scale=False, random_crop=False, random_flip=False):
        self.crop_size = cf.input_size
        if random_scale:
            self.scale = [np.random.uniform(cf.da_kwargs['scale'][0], cf.da_kwargs['scale'][1]),
                          np.random.uniform(cf.da_kwargs['scale'][0], cf.da_kwargs['scale'][1])]
        else:
            self.scale = [1, 1]
        self.random_crop = random_crop
        if random_flip:
            self.random_flip = [np.random.randint(0, 2), np.random.randint(0, 2)]
        else:
            self.random_flip = [0, 0]

    def image_transform_with_bbox(self, data1, data2,mask=None, order=1, pad_value=0, centerd=None, bbox=None):
        """
        :param data:
        :param order:
        :param pad_value:
        :param centerd: [z,y,x,d,h,w]
        :param bbox:  [z1,y1,x1,z2,y2,x2] 右开区间
        :return:
        """
        shape_scale = np.array(self.crop_size) / data1.shape
        self.scale = list(shape_scale * self.scale)
        data1 = zoom(data1, self.scale, order=order)
        data2 = zoom(data2, self.scale, order=order)
        if mask is not None:
            mask = zoom(mask, self.scale, order=0)

        if centerd is not None:
            centerd[:, :6] *= np.array(self.scale * 2)
        if bbox is not None:
            bbox = np.array(bbox, dtype=float)
            bbox[:, :6] *= np.array(self.scale * 2)

        if np.any([data1.shape[dim] < ps for dim, ps in enumerate(self.crop_size)]):
            new_shape = [np.max([data1.shape[dim], ps]) for dim, ps in enumerate(self.crop_size)]
            data1 = pad_nd_image(data1, new_shape, mode='constant', kwargs={'constant_values': pad_value})
            data2 = pad_nd_image(data2, new_shape, mode='constant', kwargs={'constant_values': pad_value})
            if mask is not None:
                mask = pad_nd_image(mask, new_shape, mode='constant', kwargs={'constant_values': 0})
            if centerd is not None:
                centerd[:, :3] += (np.array(new_shape) - np.array(data.shape)) // 2
            if bbox is not None:
                bbox[:, :3] += (np.array(new_shape) - np.array(data.shape)) // 2
                bbox[:, 3:6] += (np.array(new_shape) - np.array(data.shape)) // 2

        for ii in range(len(data1.shape)):
            if self.random_crop:
                if data1.shape[ii] > self.crop_size[ii]:
                    min_crop = np.random.randint(0, data1.shape[ii] - self.crop_size[ii]) // 2
                else:
                    min_crop = 0
            else:
                min_crop = (data1.shape[ii] - self.crop_size[ii]) // 2

            max_crop = min_crop + self.crop_size[ii]
            data1 = np.take(data1, indices=range(min_crop, max_crop), axis=ii)
            data2 = np.take(data2, indices=range(min_crop, max_crop), axis=ii)
            if mask is not None:
                mask = np.take(mask, indices=range(min_crop, max_crop), axis=ii)
            if centerd is not None:
                centerd[:, ii] -= min_crop
            if bbox is not None:
                bbox[:, ii] -= min_crop
                bbox[:, ii + 3] -= min_crop

        for ii, flag in enumerate(self.random_flip):
            if flag:
                data1 = np.flip(data1, ii)
                data2 = np.flip(data2, ii)
                if mask is not None:
                    mask = np.flip(mask, ii)
                if centerd is not None:
                    centerd[:, ii] = data1.shape[ii] - centerd[:, ii] - 1
                if bbox is not None:
                    bbox[:, [ii, ii + 3]] = bbox[:, [ii + 3, ii]]
                    bbox[:, ii] = data1.shape[ii] - bbox[:, ii]
                    bbox[:, ii + 3] = data1.shape[ii] - bbox[:, ii + 3]

        if mask is not None and centerd is not None and bbox is not None:
            return data1, data2, mask, centerd, bbox
        elif mask is not None and centerd is not None:
            return data1, data2, mask, centerd
        elif mask is not None and bbox is not None:
            return data1, data2, mask, bbox
        elif mask is not None:
            return data1, data2, mask
        else:
            return data1, data2


def data_augment(sample, target, bboxes, coord=None, ifflip=True, ifrotate=True, ifswap=True):
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[1:3]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            # 旋转相对于sample中心点旋转
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[0:3]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(1, 2), reshape=False)
                if coord is not None:
                    coord = rotate(coord, angle1, axes=(1, 2), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[0] == sample.shape[1] and sample.shape[0] == sample.shape[2]:
            axisorder = np.random.permutation(3)  # 将维度顺序[0,1,2]打乱
            sample = np.transpose(sample, axisorder)
            coord = np.transpose(coord, axisorder)
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        # flip z, x or y axis
        flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[::flipid[0], ::flipid[1], ::flipid[2]])
        if coord is not None:
            coord = np.ascontiguousarray(coord[::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            # target和bboxes都做与图像相同的翻转
            if flipid[ax] == -1:
                # 128 - target[ax]
                # 128 - bboxes[:, ax]
                target[ax] = np.array(sample.shape[ax] - 1) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax] - 1) - bboxes[:, ax]

    return sample, target, bboxes, coord


class TestDataLoader():
    """
    定义一个加载test数据的dataloader
    """

    def __init__(self, dataset, cf):
        self.dataset = dataset
        self.batch_size = cf.batch_size
        self.dim = cf.dim
        self.cf = cf

    def __getitem__(self, idx):
        if idx < len(self):
            imgs, start_coords_yxz, *datas = self.dataset[idx]
            batch_imgs, batch_start_coord_zyxs = [], []
            n_splite = len(imgs)
            while n_splite > 0:
                batch_img = imgs[self.batch_size * len(batch_imgs): self.batch_size * (len(batch_imgs) + 1)]
                batch_img = np.concatenate([img[np.newaxis, np.newaxis] for img in batch_img], 0)
                batch_start_coord_zyx = start_coords_yxz[
                                        self.batch_size * len(batch_imgs): self.batch_size * (len(batch_imgs) + 1)]
                batch_imgs.append(torch.from_numpy(batch_img))
                batch_start_coord_zyxs.append(np.array(batch_start_coord_zyx))
                n_splite -= self.batch_size
            return batch_imgs, batch_start_coord_zyxs, datas
        raise StopIteration()

    def __len__(self):
        return len(self.dataset)


def calc_dice(mask1, mask2):
    inter = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)

    if union == 0:
        return 1

    return 2 * float(inter) / float(union)