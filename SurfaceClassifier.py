'''
Description: 
Email: hymath@mail.ustc.edu.cn
Date: 2020-10-07 15:20:12
LastEditTime: 2020-10-07 15:20:25
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, no_residual=False, last_op=nn.Sigmoid()):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        # self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def _forward(self, feature, return_inter_var = False):
        
        y = feature
        tmpy = feature

        num_filter = len(self.filters)

        for i in range(num_filter):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](y if i == 0 else torch.cat([y, tmpy], 1))
            
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            
            if return_inter_var and i == num_filter // 2:
                return y

        if self.last_op:
            y = self.last_op(y)

        return y

    def forward(self, feature_1, feature_2 = None):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        if feature_2 is None:
            return self._forward(feature_1)
        else:
            inter_var1 = self._forward(feature_1, return_inter_var=True)
            inter_var2 = self._forward(feature_2, return_inter_var=True)

            y = torch.stack([inter_var1, inter_var2], dim=1).mean(dim=1)
            tmpy = torch.stack([feature_1, feature_2], dim=1).mean(dim=1)

            num_filter = len(self.filters)
            inter_layer_index = 1 + num_filter // 2
            
            for i in range(inter_layer_index, num_filter):
                if self.no_residual:
                    y = self._modules['conv' + str(i)](y)
                else:
                    y = self._modules['conv' + str(i)](y if i == 0 else torch.cat([y, tmpy], 1))
                
                if i != len(self.filters) - 1:
                    y = F.leaky_relu(y)

        if self.last_op:
            y = self.last_op(y)

        return y
