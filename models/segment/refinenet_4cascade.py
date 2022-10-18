import torch
import torch.nn as nn
import torchvision.models as models
from models.segment.blocks import un_pool, ResidualConvUnit, RefineNetBlock, RefineNetBlockImprovedPooling
from models.losses.loss_compute import SegLoss_2D


class BaseRefineNet4Cascade(nn.Module):
    """Multi-path 4-Cascaded RefineNet for image segmentation

    Args:
        input_shape ((int, int, int)): (channel, h, w) assumes input has
            equal height and width
        refinenet_block (block): RefineNet Block
        num_classes (int, optional): number of classes
        features (int, optional): number of features in net
        resnet_factory (func, optional): A Resnet model from torchvision.
            Default: models.resnet101
        pretrained (bool, optional): Use pretrained version of resnet
            Default: True
        freeze_resnet (bool, optional): Freeze resnet model
            Default: True

    Raises:
        ValueError: size of input_shape not divisible by 32
    """
    def __init__(self,
                 input_shape,
                 refinenet_block=RefineNetBlock,
                 num_classes=10,
                 features=256,
                 input_channel=3,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):

        super().__init__()

        input_h, input_w = input_shape

        if input_h % 32 != 0:
            raise ValueError("{} not divisble by 32".format(input_shape))

        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = refinenet_block(2 * features,
                                         (2 * features, input_h // 32, input_w // 32))
        self.refinenet3 = refinenet_block(features,
                                         (2 * features, input_h // 32, input_w // 32),
                                         (features, input_h // 16, input_w // 16))
        self.refinenet2 = refinenet_block(features,
                                         (features, input_h // 16, input_w // 16),
                                         (features, input_h // 8, input_w // 8))
        self.refinenet1 = refinenet_block(features,
                                         (features, input_h // 8, input_w // 8),
                                         (features, input_h // 4, input_w // 4))

        self.segBranch = nn.Sequential(
            ResidualConvUnit(features),
            ResidualConvUnit(features),
            nn.Conv2d(features, num_classes, kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.Sigmoid())

        self.depthBranch = nn.Sequential(
            ResidualConvUnit(features),
            ResidualConvUnit(features),
            nn.Conv2d(features, 1, kernel_size=1, stride=1,
                      padding=0, bias=True))

        self.initialize_weights()

        resnet = resnet_factory(pretrained=pretrained)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))

    def forward(self, x):

        layer_1 = self.layer1(x)
        # print('layer_1',layer_1.shape)
        layer_2 = self.layer2(layer_1)
        # print('layer_2',layer_2.shape)
        layer_3 = self.layer3(layer_2)
        # print('layer_3',layer_3.shape)
        layer_4 = self.layer4(layer_3)
        # print('layer_4',layer_4.shape)
        layer_1_rn = self.layer1_rn(layer_1)
        # print('layer_1_rn',layer_1_rn.shape)
        layer_2_rn = self.layer2_rn(layer_2)
        # print('layer_2_rn',layer_2_rn.shape)
        layer_3_rn = self.layer3_rn(layer_3)
        # print('layer_3_rn',layer_3_rn.shape)
        layer_4_rn = self.layer4_rn(layer_4)
        # print('layer_4_rn',layer_4_rn.shape)
        path_4 = self.refinenet4(layer_4_rn)
        # print('path_4',path_4.shape)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        # print('path_3',path_3.shape)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        # print('path_2',path_2.shape)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        # print('path_1',path_1.shape)
        path_1 = un_pool(path_1, 4)
        # print('path_1',path_1.shape)
        seg = self.segBranch(path_1)
        depth = self.depthBranch(path_1)

        out = self.output_conv(path_1)
        # print('out',out.shape)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def named_parameters(self):
    #     """Returns parameters that requires a gradident to update."""
    #     return (p for p in super().named_parameters() if p[1].requires_grad)


class RefineNet4Cascade(BaseRefineNet4Cascade):
    """Multi-path 4-Cascaded RefineNet for image segmentation

    Args:
        input_shape ((int, int, int)): (channel, h, w) assumes input has
            equal height and width
        refinenet_block (block): RefineNet Block
        num_classes (int, optional): number of classes
        features (int, optional): number of features in net
        resnet_factory (func, optional): A Resnet model from torchvision.
            Default: models.resnet101
        pretrained (bool, optional): Use pretrained version of resnet
            Default: True
        freeze_resnet (bool, optional): Freeze resnet model
            Default: True

    Raises:
        ValueError: size of input_shape not divisible by 32
    """
    def __init__(self,
                 input_shape,
                 num_classes=10,
                 features=256,
                 input_channel=3,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):

        super().__init__(
            input_shape,
            refinenet_block=RefineNetBlock,
            num_classes=num_classes,
            features=features,
            input_channel=input_channel,
            resnet_factory=resnet_factory,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):
    """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling

    Args:
        input_shape ((int, int, int)): (channel, h, w) assumes input has
            equal height and width
        refinenet_block (block): RefineNet Block
        num_classes (int, optional): number of classes
        features (int, optional): number of features in net
        resnet_factory (func, optional): A Resnet model from torchvision.
            Default: models.resnet101
        pretrained (bool, optional): Use pretrained version of resnet
            Default: True
        freeze_resnet (bool, optional): Freeze resnet model
            Default: True

    Raises:
        ValueError: size of input_shape not divisible by 32
    """
    def __init__(self,
                 input_shape,
                 num_classes=10,
                 features=256,
                 input_channel=3,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):

        super().__init__(
            input_shape,
            refinenet_block=RefineNetBlockImprovedPooling,
            num_classes=num_classes,
            features=features,
            input_channel=input_channel,
            resnet_factory=resnet_factory,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)

class Net(nn.Module):

    def __init__(self, cf, logger):

        super(Net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.net = RefineNet4Cascade(input_shape=cf.input_size, input_channel=cf.input_channel,num_classes=cf.num_seg_classes)
        self.loss_func = SegLoss_2D(cf)

    def forward(self, x, mask_gt=None, phase='train'):
        output = self.net(x)
        if phase != 'test':
            loss, cc_loss, dice_loss = self.loss_func(output, mask_gt)

            result_dict = {
                'loss': loss,
                'cc_loss': cc_loss,
                'dice_loss': dice_loss,
            }
        else:
            result_dict = {
                'predict': output,
            }
        return result_dict
