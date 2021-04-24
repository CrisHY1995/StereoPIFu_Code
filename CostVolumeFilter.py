'''
Description: 
Author: HongYang
Email: hymath@mail.ustc.edu.cn
Date: 2020-10-07 15:21:54
LastEditTime: 2020-10-07 15:21:56
'''
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from networks import convbn_3d, hourglass3D


class CostVolumeFilter(nn.Module):
    def __init__(self, in_nfeat, out_nfeat=32, num_stacks=4):
        super(CostVolumeFilter, self).__init__()

        self.dres0 = nn.Sequential(nn.Conv3d(in_nfeat, out_nfeat, kernel_size=3, padding=1, stride=(1, 2, 2), bias=False),
                                  nn.BatchNorm3d(out_nfeat),
                                  nn.ReLU(inplace=True),
                                  convbn_3d(out_nfeat, out_nfeat, 3, 1, 1),
                                  nn.ReLU(inplace=True)
                                  )

        self.dres1 = nn.Sequential(convbn_3d(out_nfeat, out_nfeat, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(out_nfeat, out_nfeat, 3, 1, 1)) 

        self.num_stacks = num_stacks

        for i in range(self.num_stacks):
            self.add_module("HG3D_%d"%i, hourglass3D(out_nfeat))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, cost):

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        res = []
        out, pre, post = self._modules["HG3D_0"](cost0, None, None)
        out = out + cost0
        res.append(out)

        for i in range(1, self.num_stacks):
            out, _, post = self._modules["HG3D_%d"%i](cost0, pre, post)
            out = out + cost0
            res.append(out)

        return res


    