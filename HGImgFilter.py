import torch.nn as nn
import torch.nn.functional as F
from networks import ConvBlock, HourGlass

class HGFilter(nn.Module):
    def __init__(self, in_nfeat = 3, num_stack = 4, norm_type = 'group', hg_down = 'ave_pool', num_hourglass = 2, hourglass_dim = 256):
        super(HGFilter, self).__init__()
        
        self.num_modules = num_stack
        self.norm_type = norm_type
        self.hg_down = hg_down
        self.num_hourglass = num_hourglass
        self.hourglass_dim = hourglass_dim

        # Base part
        self.conv1 = nn.Conv2d(in_nfeat, 64, kernel_size=7, stride=2, padding=3)

        if self.norm_type== 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm_type == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.norm_type)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.norm_type)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.norm_type)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.norm_type)
        self.conv4 = ConvBlock(128, 256, self.norm_type)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, self.num_hourglass, 256, self.norm_type))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.norm_type))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.norm_type == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.norm_type == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
                
            self.add_module('l' + str(hg_module), nn.Conv2d(256, self.hourglass_dim, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(self.hourglass_dim,256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, tmpx.detach(), normx
