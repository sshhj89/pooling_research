import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


''' from: 
https://github.com/rentainhe/pytorch-pooling/blob/master/Pooling/pooling_method/lip_pooling.py
https://github.com/sebgao/LIP/blob/5e85b9e55b9212fdf6abccb1fa8783d23620f53a/imagenet/lip_densenet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def lip2d(x, logit, kernel=2, stride=2, padding=0):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)

class SoftGate(nn.Module):
    def __init__(self, coeff=12):
        super(SoftGate, self).__init__()
        self.coeff = coeff

    def forward(self, x):
        return torch.sigmoid(x).mul(self.coeff)

class SimplifiedLIP(nn.Module):
    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()

        rp = channels
        # nn.Sequential + OrderedDict
        self.logit = nn.Sequential(
            OrderedDict((
                ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac

def lip():
    print("You are using Lip Pooling Method")
    return SimplifiedLIP
