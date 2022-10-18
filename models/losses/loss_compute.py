import torch.nn.functional as F
import torch.nn as nn
import torch

from models.losses.dice_loss import MultiClassDiceLoss, DiceLoss


class SegLoss_2D(torch.nn.Module):
    """
    计算分割loss,二分类和多分类均可使用.若预测结果和目标尺寸不一致,会对预测结果插值之后再计算loss
    """

    def __init__(self, cf):
        super(SegLoss_2D, self).__init__()
        self.diceloss_weight = cf.diceloss_weight if hasattr(cf, 'diceloss_weight') else 0.5
        self.dice_bg_open = cf.dice_bg_open if hasattr(cf, 'dice_bg_open') else True
        self.dice_size_avg = cf.dice_size_avg if hasattr(cf, 'dice_size_avg') else True
        self.num_seg_classes = cf.num_seg_classes if hasattr(cf, 'num_seg_classes') else 2

    def forward(self, predict, target):
        ph, pw = predict.size(2), predict.size(3)
        h, w = target.size(2), target.size(3)

        if ph != h or pw != w:
            predict = F.upsample(input=predict, size=(h, w), mode='trilinear')

        if self.num_seg_classes > 1:
            cc_loss_function = nn.CrossEntropyLoss()
            cc_loss = cc_loss_function(predict, torch.argmax(target, dim=1))

            diceloss_fuc = MultiClassDiceLoss()
            dice_loss = diceloss_fuc(predict, target, optimize_bg=self.dice_bg_open, size_average=self.dice_size_avg)

            loss = cc_loss * (1 - self.diceloss_weight) + dice_loss * self.diceloss_weight
            return loss, cc_loss, dice_loss
        else:
            cc_loss_function = nn.BCEWithLogitsLoss()
            cc_loss = cc_loss_function(predict, target.type(torch.cuda.FloatTensor))

            diceloss_fuc = DiceLoss()
            dice_loss = diceloss_fuc(predict, target)

            loss = cc_loss * (1 - self.diceloss_weight) + dice_loss * self.diceloss_weight
            return loss, cc_loss, dice_loss


