import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np

from .AANet_utils import GANetFeature, FeaturePyrmaid, CostVolumePyramid, AdaptiveAggregation, CostVolume, DisparityEstimation, HourglassRefinement


class AANetPlusFeature(nn.Module):

    def __init__(self, input_reso, max_disp = 72) -> None:
        super().__init__()
        
        num_scales = 3
        num_fusions = 6
        num_stage_blocks = 1
        num_deform_blocks = 3
        mdconv_dilation = 2
        deformable_groups = 2
        no_intermediate_supervision = False
        no_feature_mdconv = False
        feature_similarity = "correlation"

        self.refinement_type = "hourglass"
        self.feature_type = "ganet"

        self.num_downsample = 2
        self.aggregation_type = "adaptive"
        self.num_scales = num_scales

        self.max_disp = max_disp // 3
        self.feature_extractor = GANetFeature(feature_mdconv=(not no_feature_mdconv))
        self.feature_reso_list = [math.ceil(input_reso/3), math.ceil(input_reso/6), math.ceil(input_reso/12)]
        self.fpn = FeaturePyrmaid()
        self.cost_volume = CostVolumePyramid(self.max_disp, feature_similarity=feature_similarity)
        self.cost_volume3D = CostVolume(self.max_disp, feature_similarity="difference")

        self.aggregation = AdaptiveAggregation(max_disp=self.max_disp,
                                                num_scales=num_scales,
                                                num_fusions=num_fusions,
                                                num_stage_blocks=num_stage_blocks,
                                                num_deform_blocks=num_deform_blocks,
                                                mdconv_dilation=mdconv_dilation,
                                                deformable_groups=deformable_groups,
                                                intermediate_supervision=not no_intermediate_supervision)

        self.disparity_estimation = DisparityEstimation(self.max_disp, match_similarity = True)
        refine_module_list = nn.ModuleList()
        for _ in range(self.num_downsample):
            refine_module_list.append(HourglassRefinement())
        self.refinement = refine_module_list

    def feature_extraction(self, img):
        feature = self.feature_extractor(img)
        feature = self.fpn(feature)
        return feature

    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)
        return cost_volume

    def Construct3DFeatureVolume(self, left_feature_list, right_feature_list):

        target_reso = self.feature_reso_list[0]
        num = len(left_feature_list)

        left_res = []
        right_res = []

        for i in range(num):
            cur_left_feature = left_feature_list[i]
            if cur_left_feature.size(-1) != target_reso:
                left_res.append(F.interpolate(cur_left_feature, size=(target_reso, target_reso), mode="bilinear"))
            else:
                left_res.append(cur_left_feature)

            cur_right_feature = right_feature_list[i]
            if cur_right_feature.size(-1) != target_reso:
                right_res.append(F.interpolate(cur_right_feature, size=(target_reso, target_reso), mode="bilinear"))
            else:
                right_res.append(cur_right_feature)

        left_feature = torch.cat(left_res, dim=1)
        right_feature = torch.cat(right_res, dim=1)

        cost_volume3D = self.cost_volume3D(left_feature, right_feature)
        return left_feature, right_feature, cost_volume3D

    def ConstructConfindenceVolume(self, aggregation_list):
        target_reso = self.feature_reso_list[0]
        res = []
        for aggregation in aggregation_list:
            confidence_volume = F.softmax(aggregation, dim=1)
            confidence_volume = confidence_volume.unsqueeze(1)
            if confidence_volume.size(-1) !=  target_reso:
                res.append(F.interpolate(confidence_volume, size=(self.max_disp, target_reso, target_reso), mode="trilinear"))
            else:
                res.append(confidence_volume)

        temp_confidence_volume = torch.cat(res, dim=1).mean(dim=1, keepdim=True)
        return temp_confidence_volume
        
    def disparity_computation(self, aggregation):
        assert isinstance(aggregation, list)
        disparity_pyramid = []
        length = len(aggregation)  # D/3, D/6, D/12
        for i in range(length):
            disp = self.disparity_estimation(aggregation[length - 1 - i])  # reverse
            disparity_pyramid.append(disp)  # D/12, D/6, D/3

        return disparity_pyramid

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []

        for i in range(self.num_downsample):
            scale_factor = 1. / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(left_img, scale_factor=scale_factor, mode='bilinear')
                curr_right_img = F.interpolate(right_img, scale_factor=scale_factor, mode='bilinear')
            
            inputs = (disparity, curr_left_img, curr_right_img)
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid

    def forward(self, left_img, right_img):

        left_feature_list = self.feature_extraction(left_img)
        right_feature_list = self.feature_extraction(right_img)
        
        left_feature, right_feature, cost_volume3D = self.Construct3DFeatureVolume(left_feature_list, right_feature_list)

        cost_volume = self.cost_volume_construction(left_feature_list, right_feature_list)
        aggregation = self.aggregation(cost_volume)
        confidence_volume = self.ConstructConfindenceVolume(aggregation)

        disparity_pyramid = self.disparity_computation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])
        # print(cost_volume3D.size())
        # print(confidence_volume.size())

        return left_feature, right_feature, cost_volume3D, confidence_volume, disparity_pyramid[-1]

if __name__ == "__main__":

    device = torch.device("cuda:2")
    # model_path = "./PreTrainModels/AANetPlusFeatures.pth"
    tt = AANetPlusFeature(input_reso=576).to(device)
    RPAXYZ_model_path = "/data2/hongyang/DHRProject/RealData_V2/CheckPoints/RPAXYZ_aanet+_v1.pth"
    tt.load_state_dict(torch.load(RPAXYZ_model_path, map_location=device))

    tt.eval()
    def load_image2tensor(img_path):
        img = cv2.imread(img_path).astype(np.float32)/255.0
        img = cv2.resize(img, (576, 576))
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
        return img.to(device)

    l_img = load_image2tensor("/data2/hongyang/MyData/test/axyz_rigged/CWom0322_M4_axyz_rigged/images/L_RGB_0.png")
    r_img = load_image2tensor("/data2/hongyang/MyData/test/axyz_rigged/CWom0322_M4_axyz_rigged/images/R_RGB_0.png")

    # print(l_img.size(), l_img.dtype)
    # print(r_img.size(), r_img.dtype)

    left_feature_list, right_feature_list, cost_volume3D, confidence_volume, disparity_pyramid = tt(l_img, r_img)

    for ll in disparity_pyramid:
        print(ll.size())