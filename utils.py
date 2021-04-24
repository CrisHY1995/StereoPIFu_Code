import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ZEDProject(nn.Module):
    def __init__(self, camera, use_oriZ) -> None:
        super().__init__()

        ExterMat = torch.from_numpy(camera["ExterMat"]).unsqueeze(0)
        InterMat = torch.from_numpy(camera["InterMat"]).unsqueeze(0)

        self.register_buffer("exter_rot", ExterMat[:, :3, :3])
        self.register_buffer("exter_trans", ExterMat[:, :3, 3:4])
        self.register_buffer("inter_scale", InterMat[:, :2, :2])
        self.register_buffer("inter_trans", InterMat[:, :2, 2:])
        self.use_oriZ = use_oriZ

    def __call__(self, points):
        """
        points: [B, 3, N]
        """
        batch_size = points.size(0)
        exter_trans = self.exter_trans.expand(batch_size, -1, -1)
        exter_rot = self.exter_rot.expand(batch_size, -1, -1)
        inter_trans = self.inter_trans.expand(batch_size, -1, -1)
        inter_scale = self.inter_scale.expand(batch_size, -1, -1)

        homo = torch.baddbmm(exter_trans, exter_rot, points)  # [B, 3, N]
        xy = homo[:, :2, :] / homo[:, 2:3, :]
        uv = torch.baddbmm(inter_trans, inter_scale, xy)
        if self.use_oriZ:
            return uv, points[:, 2:3, :]
            # xyz = torch.cat([uv, points[:, 2:3, :]], 1)
        else:
            return uv, homo[:, 2:3, :]
            # xyz = torch.cat([uv, homo[:, 2:3, :]], 1)

        # return xyz
        
class DilateMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = [[0.00, -0.25, 0.00],
                  [-0.25, 1.00, -0.25],
                  [0.00, -0.25, 0.00]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding_func = nn.ReplicationPad2d(1)
    

    def forward(self, batch_mask, iter_num):
        for _ in range(iter_num):
            padding_mask = self.padding_func(batch_mask)
            res = F.conv2d(padding_mask, self.weight, bias=None, stride = 1, padding=0)
            batch_mask[res.abs() > 0.0001] = 1.0
        return batch_mask

class ImgNormalizationAndInv(nn.Module):
    def __init__(self, RGB_mean, RGB_std):
        super().__init__()

        mean=torch.FloatTensor(RGB_mean).view(1,3,1,1)
        std=torch.FloatTensor(RGB_std).view(1,3,1,1)
        self.register_buffer("rgb_mean", mean)
        self.register_buffer("rgb_std", std)

    
    def forward(self, image_tensor, inv):
        if inv:
            image_tensor = image_tensor * self.rgb_std + self.rgb_mean
        else:
            image_tensor = (image_tensor - self.rgb_mean)/self.rgb_std

        return image_tensor
    
class ExtractDepthEdgeMask(nn.Module):
    def __init__(self, thres_) -> None:
        super().__init__()
        self.thres = thres_

    def forward(self, batch_depth):
        B, _, H, W = batch_depth.size()
        patch = F.unfold(batch_depth, kernel_size=3, padding=1, stride=1)
        min_v, _ = patch.min(dim=1, keepdim = True)
        max_v, _ = patch.max(dim=1, keepdim = True)

        mask = (max_v - min_v) > self.thres
        mask = mask.view(B, 1, H, W).float()
        
        return mask