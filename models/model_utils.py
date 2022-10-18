import torch.nn as nn
import numpy as np


def initialize_weights(net, weight_init='kaiming_uniform'):
    """
    Initialize model weights.
    """
    if weight_init is None:
        weight_init = 'kaiming_uniform'
    for m in net.modules():
        if type(m) in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear]:
            if weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif weight_init == 'xavier_normal':
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif weight_init == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity='relu', a=0)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.uniform_(m.bias, -bound, bound)

            elif weight_init == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu', a=0)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

            elif weight_init == 'normal':
                nn.init.normal_(m.weight, std=0.001)

        elif isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()



