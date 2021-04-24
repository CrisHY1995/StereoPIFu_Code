import torch
import torch.nn as nn

from AANetPlusFeature import AANetPlusFeature
from HGImgFilter import HGFilter
from CostVolumeFilter import CostVolumeFilter
from SurfaceClassifier import SurfaceClassifier

from ZEDCamera import zed_camera_normalized, zed_camera_regu
from utils import ZEDProject, ImgNormalizationAndInv, DilateMask, ExtractDepthEdgeMask

class StereoPIFuNet(nn.Module):
    def __init__(self, use_VisualHull, use_bb_of_rootjoint):

        super().__init__()
        
        self.use_VisualHull = use_VisualHull
        self.use_bb_of_rootjoint = use_bb_of_rootjoint

        self.max_disp = 72
        self.num_stacks = 4
        self.normalize_rgb = True
        self.use_OriZ = False
        self.img_size = int(zed_camera_regu.l_camera["CameraReso"][0])
        self.baseline = zed_camera_regu.baseline

        self.Stereo_RGB_mean = [0.485, 0.456, 0.406]
        self.Stereo_RGB_std = [0.229, 0.224, 0.225]
        self.HG_RGB_mean = [0.5, 0.5, 0.5]
        self.HG_RGB_std = [0.5, 0.5, 0.5]

        self.depth_edge_thres = 0.08
        self.num_3Dnfeat = 64
        self.point_feature = 256 + self.num_3Dnfeat + 1 + 1 #[image_feature, cost_feature, abso_depth_z, confidence]
        self.im_nfeat = 3
        
        self.sigmoid_coef = 50.0
        if self.sigmoid_coef > 0:
            self.use_sigmoid_z = True
        else:
            self.use_sigmoid_z = False

        self.LeftFeature_List = None
        self.CostVolume3D_List = None
        self.ConfidVolume3D = None
        self.DepthTensor = None
        self.DepthEdgeMask = None
        self.LMaskTensor = None
        self.RMaskTensor = None
        self.l_rgb = None
        self.r_rgb = None
        self._build_tool_funcs()
    
    def _build_tool_funcs(self):
        self.Cost3DFilter = CostVolumeFilter(in_nfeat=224, out_nfeat=self.num_3Dnfeat, num_stacks=self.num_stacks)
        self.image_filter = HGFilter(in_nfeat=self.im_nfeat, num_stack=self.num_stacks)
        self.surface_classifier = SurfaceClassifier(filter_channels=[self.point_feature, 1024, 512, 256, 128, 1])

        self.l_projection = ZEDProject(zed_camera_normalized.l_camera, self.use_OriZ)
        self.r_projection = ZEDProject(zed_camera_normalized.r_camera, self.use_OriZ)
        
        self.aanet_feat_extractor = AANetPlusFeature(self.img_size, self.max_disp)
        self.dilate_mask_func = DilateMask()
        self.StereoRGBNormalizer = ImgNormalizationAndInv(self.Stereo_RGB_mean, self.Stereo_RGB_std)
        self.HGNormalizer = ImgNormalizationAndInv(self.HG_RGB_mean, self.HG_RGB_std)

        self.extract_depth_edge_mask_func = ExtractDepthEdgeMask(thres_ = self.depth_edge_thres)

    @staticmethod
    def calc_mean_z(depth_tensor, mask_c1b):
        batch_size = depth_tensor.size(0)
        res = []
        for i in range(batch_size):
            res.append(torch.mean(depth_tensor[i, :, :, :][mask_c1b[i, :, :,:]]))

        return torch.stack(res, dim=0).view(batch_size, 1, 1)

    def DepthDispConvertor(self, batch_input, batch_mask):
        if batch_input.dim() == 3:
            batch_input = batch_input.unsqueeze(1)

        assert 4 == batch_mask.dim()
        if batch_mask.dtype != torch.bool:
            batch_mask = batch_mask > 0.5
            
        mask = batch_mask & (batch_input > 0.0001)
        res = batch_input.clone()
        res[mask] = self.baseline / (batch_input[mask])
        res[~mask] = 0.0
        return res
    
    def update_feature_by_imgs(self, l_rgb, r_rgb, l_mask, r_mask):
        l_mask_c1b = l_mask > 0.5
        l_mask_c3b = l_mask_c1b.expand(-1, 3, -1, -1)
        
        left_feature, right_feature, cost_volume3D, confidence_volume, disparity_map = self.aanet_feat_extractor(l_rgb, r_rgb)
        depth_tensor = self.DepthDispConvertor(disparity_map.detach(), l_mask)
        
        ori_l_rgb = self.StereoRGBNormalizer(l_rgb, inv = True)
        hg_l_rgb = self.HGNormalizer(ori_l_rgb, inv = False)
        hg_l_rgb[~l_mask_c3b] = 0.0

        im_input = hg_l_rgb
        self.LeftFeature_List = self.image_filter(im_input)[0]
        self.CostVolume3D_List = self.Cost3DFilter(cost_volume3D.detach())
        self.ConfidVolume3D = confidence_volume.detach()

        l_mask_tensor = self.dilate_mask_func(l_mask, 2).detach() if self.use_VisualHull else None
        r_mask_tensor = self.dilate_mask_func(r_mask, 2).detach() if self.use_VisualHull else None
        
        self.DepthTensor = depth_tensor.detach()
        self.DepthEdgeMask = self.extract_depth_edge_mask_func(self.DepthTensor)
        self.LMaskTensor = l_mask_tensor
        self.RMaskTensor = r_mask_tensor
        self.mean_z = self.calc_mean_z(self.DepthTensor, l_mask_c1b)        

    @staticmethod
    def index2D(feat, uv, mode="bilinear"):
        uv = uv.transpose(1, 2)  # [B, N, 2]
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True, mode=mode)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]            

    @staticmethod
    def index3D(feat, uv3d, mode = "bilinear"):
        grid_ = uv3d.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)  #[B, 1, 1, N, 3]
        samples = torch.nn.functional.grid_sample(feat, grid_, mode=mode) # [B, C, 1, 1, N]
        return samples[:, :, 0, 0, :]  #[B, C, N]
    
    def get_feat2D(self, feat, uv, edge_mask):
        
        if edge_mask is not None:
            feat_bilinear = self.index2D(feat, uv, mode = "bilinear")
            feat_nearest = self.index2D(feat, uv, mode="nearest")
            feat = torch.where(edge_mask, feat_nearest, feat_bilinear)
        else:
            feat = self.index2D(feat, uv, mode = "bilinear")
        return feat

    def get_feat3D(self, feat, uv, edge_mask):
        if edge_mask is not None:
            feat_bilinear = self.index3D(feat, uv, mode = "bilinear")
            feat_nearest = self.index3D(feat, uv, mode="nearest")
            feat = torch.where(edge_mask, feat_nearest, feat_bilinear)
        else:
            feat = self.index3D(feat, uv, mode = "bilinear")
        return feat
    
    def sample_feature(self, points):

        assert self.DepthTensor is not None
        assert self.DepthEdgeMask is not None

        point_num = points.size(-1)
        batch_size = points.size(0) 

        left_uv, left_z = self.l_projection(points)
        right_uv, right_z = self.r_projection(points)
        disp = (left_uv[:, 0:1, :] - right_uv[:, 0:1, :]) * self.img_size / self.max_disp - 1.0

        uv3d = torch.cat([left_uv, disp], dim=1)
        in_img = (uv3d[:, 0] >= -1.0) & (uv3d[:, 0] <= 1.0) & (uv3d[:, 1] >= -1.0) & (uv3d[:, 1] <= 1.0) & (uv3d[:, 2] >= -1.0) & (uv3d[:, 2] <= 1.0)
        if self.use_VisualHull:
            assert self.LMaskTensor is not None
            assert self.LMaskTensor.dtype == torch.float32
            assert self.RMaskTensor is not None
            assert self.RMaskTensor.dtype == torch.float32
            l_mask_value = self.index2D(self.LMaskTensor, left_uv, mode="nearest")
            r_mask_value = self.index2D(self.RMaskTensor, right_uv, mode="nearest")
            in_img = in_img & (l_mask_value[:, 0] > 0.5) & (r_mask_value[:, 0] > 0.5)
        
        if self.use_bb_of_rootjoint:
            temp_z = left_z - self.mean_z
            in_img = in_img & (temp_z[:, 0, :] > -0.70) &(temp_z[:, 0, :] < 0.70) #rela_j0z, [B, 1, N]

        point_feat_mask_c1f = in_img.view(batch_size, 1, point_num).float()
        
        edge_mask_c1f = self.index2D(self.DepthEdgeMask, left_uv, mode="bilinear")
        edge_mask_c1b = edge_mask_c1f > 0.01

        z_predict = self.get_feat2D(self.DepthTensor, left_uv, edge_mask=edge_mask_c1b)
        left_z = left_z - z_predict
        
        if self.use_sigmoid_z:
            left_z = (2.0 / (1.0 + torch.exp(-1.0 * self.sigmoid_coef * left_z)) - 1.0)
            
        confid_feature = self.get_feat3D(self.ConfidVolume3D, uv3d, edge_mask_c1b)

        res = []
        for i in range(self.num_stacks):
            left_feature = self.index2D(self.LeftFeature_List[i], left_uv)
            cost_feature = self.index3D(self.CostVolume3D_List[i], uv3d)
            res.append(torch.cat([left_feature, cost_feature, left_z, confid_feature], dim=1))

        return res, point_feat_mask_c1f

    def query(self, points):
        assert self.LeftFeature_List is not None
        assert self.DepthTensor is not None

        point_feat_list, point_feat_mask_c1f = self.sample_feature(points)
        pred = point_feat_mask_c1f * self.surface_classifier(point_feat_list[-1])
        return pred
    
    