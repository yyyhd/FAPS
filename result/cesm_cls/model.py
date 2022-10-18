"""
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
import torch.nn.functional as F
import numpy as np
from models.segment.pspnet import PSPModule
# __all__ = ['xception']

model_urls = {
    # 'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
    'xception': 'https://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)

        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)

        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)
    def forward(self, inp):
        x = self.rep(inp)


        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, input_channel, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(input_channel, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.psp = PSPModule(2048, 3069, (1,2,3,6))
        self.fc = nn.Linear(3096, num_classes)
        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x, cli=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.psp(x)

        x = self.dropout(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        self.feature = x
        # x = x.view(x.size(0), -1)
        #
        # x = self.fc(x)

        return x


def xception(pretrained=False, input_channel=3, num_classes=2):
    """
    Construct Xception.
    """

    model = Xception(input_channel, num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model

class Net4(nn.Module):
    def __init__(self, cf, logger):
        super(Net4, self).__init__()
        self.cf = cf
        self.logger = logger
        self.net = xception(pretrained=False,input_channel=cf.input_channel, num_classes=cf.num_classes)
        self.fc2 = torch.nn.Linear(in_features=6138, out_features=cf.num_classes)##+psp
        self.compute_loss_func = nn.CrossEntropyLoss()

    def forward(self, image0, image1, image2, image3,labels=None, cli=None, phase='train'):
        if phase != 'test':
            cout0 = self.net(image0)
            cout1 = self.net(image1)
            cout2 = self.net(image2)
            cout3 = self.net(image3)
            data_list1 = torch.cat((cout0, cout1), 1)
            data_list1 = F.adaptive_avg_pool2d(data_list1, (1, 1))
            data_list1 = data_list1.view(data_list1.size(0), -1)
            result0 = self.fc2(data_list1)
            data_list2 = torch.cat((cout2, cout3), 1)
            data_list2 = F.adaptive_avg_pool2d(data_list2, (1, 1))
            data_list2 = data_list2.view(data_list2.size(0), -1)
            result1 = self.fc2(data_list2)
            result = torch.add(result0,result1)/2.0
            loss = self.compute_loss_func(result, torch.argmax(labels, dim=1))
            assert self.cf.num_classes == 2, '下面计算tp, tn, fp, fn均仅限于二分类'
            predict = F.sigmoid(result).squeeze(1).cpu().detach().numpy()

            labels = np.array(labels.cpu()) > 0
            th = 0.5
            predict = predict > th
            tp = torch.tensor(np.sum((predict == True) & (labels == True))).cuda()
            tn = torch.tensor(np.sum((predict == False) & (labels == False))).cuda()
            fp = torch.tensor(np.sum((predict == True) & (labels == False))).cuda()
            fn = torch.tensor(np.sum((predict == False) & (labels == True))).cuda()
            acc = torch.true_divide((tp + tn), (tp + tn + fp + fn))
            result_dict = {
                'loss': loss,
                'acc': acc,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn}


            return result_dict
        else:

            cout0 = self.net(image0)
            cout1 = self.net(image1)
            cout2 = self.net(image2)
            cout3 = self.net(image3)
            data_list1 = torch.cat((cout0, cout1), 1)
            data_list1 = F.adaptive_avg_pool2d(data_list1, (1, 1))
            data_list1 = data_list1.view(data_list1.size(0), -1)
            result0 = self.fc2(data_list1)
            data_list2 = torch.cat((cout2, cout3), 1)
            data_list2 = F.adaptive_avg_pool2d(data_list2, (1, 1))
            data_list2 = data_list2.view(data_list2.size(0), -1)
            result1 = self.fc2(data_list2)

            cout = torch.add(result0, result1) / 2.0

            result_dict = {
                'outputs': cout
            }
            return result_dict
