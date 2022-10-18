from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# if there is not any code in __init__, just use def rather than class
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, gt):
        predict = predict.float()
        gt = gt.float()
        probability = torch.nn.functional.sigmoid(predict)
        intersection = torch.sum(probability*gt)
        union = torch.sum(probability*probability) + torch.sum(gt*gt)
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        return 1 - dice



class MultiClassDiceLoss(nn.Module):
    """

    定义实例分割Dice损失函数

    """

    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, predict, target, optimize_bg=True, weight=None, size_average=True, output_map='Softmax'):
        """
        predict : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class

        target : is a 1-hot representation of the groundtruth, shoud have same size
        as the input

        """

        assert predict.size() == target.size(), 'Input sizes must be equal.'
        assert predict.dim() == 5, 'Input must be a 4D Tensor.'

        if output_map == 'Softmax':
            predict = F.softmax(predict, dim=1)
        else:
            predict = F.sigmoid(predict)

        predict = predict.float()
        target = target.float()

        intersection = predict * target
        intersection = torch.sum(intersection, dim=4)
        intersection = torch.sum(intersection, dim=3)
        intersection = torch.sum(intersection, dim=2)

        union_1 = predict * predict
        union_1 = torch.sum(union_1, dim=4)
        union_1 = torch.sum(union_1, dim=3)
        union_1 = torch.sum(union_1, dim=2)

        union_2 = target * target
        union_2 = torch.sum(union_2, dim=4)
        union_2 = torch.sum(union_2, dim=3)
        union_2 = torch.sum(union_2, dim=2)

        dice_coefficient = (2. * intersection + 1e-5) / (union_1 + union_2 + 1e-5)

        if not optimize_bg:
            # ignore bg dice, and take the fg
            dice_coefficient = dice_coefficient[:, 1:]

        if not isinstance(weight, type(None)):
            if not optimize_bg:
                weight = weight[1:]  # ignore bg weight
            weight = weight.size(0) * weight / weight.sum()  # normalize fg weights
            dice_coefficient = dice_coefficient * weight  # weighting

        dice_loss = 1 - dice_coefficient.mean(1)

        if size_average:
            return dice_loss.mean()

        return dice_loss.sum()
