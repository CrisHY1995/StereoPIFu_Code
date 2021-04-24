from os.path import join
import torch.utils.data as data
import cv2
import numpy as np
import os
import torchvision.transforms as transforms
from glob import glob

class EvaDataset(data.Dataset):

    def __init__(self, test_data_dir, transform_rgb, resize512, start_idx = 0, end_idx = -1) -> None:

        self.transform_rgb = transform_rgb
        if transform_rgb:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.test_data_dir = test_data_dir
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        self.resize512 = resize512

        self.build_l_rgb_path_list()

    def build_l_rgb_path_list(self):
        # l_rgb_path_list = glob("/disk2/hongyang/MyData/TestData_small/*/L_RGB*.png")
        # l_rgb_path_list = glob(join(self.test_data_dir, "*/L_RGB*.png"))
        l_rgb_path_list = glob(join(self.test_data_dir, "L_RGB*.png"))
        l_rgb_path_list.sort()

        if self.end_idx != -1:
            l_rgb_path_list = l_rgb_path_list[self.start_idx:self.end_idx]
        else:
            l_rgb_path_list = l_rgb_path_list[self.start_idx:]
            
        self.l_rgb_path_list = l_rgb_path_list

    def __len__(self):
        return len(self.l_rgb_path_list)

    def __getitem__(self, item):
        l_rgb_path = self.l_rgb_path_list[item]

        r_rgb_path = l_rgb_path.replace("L_RGB", "R_RGB")
        l_mask_path = l_rgb_path.replace("L_RGB", "L_mask")
        r_mask_path = l_rgb_path.replace("L_RGB", "R_mask")
        # l_depth_path = l_rgb_path.replace("L_RGB", "L_depth")
        # print(l_rgb_path)
        # exit(0)

        assert os.path.exists(l_rgb_path)
        assert os.path.exists(r_rgb_path)
        # assert os.path.exists(l_depth_path)
        assert os.path.exists(l_mask_path)
        assert os.path.exists(r_mask_path)
        # assert os.path.exists(l_conf_path)

        l_rgb = cv2.imread(l_rgb_path).astype(np.float32) / 255.0
        r_rgb = cv2.imread(r_rgb_path).astype(np.float32) / 255.0
        l_rgb = cv2.cvtColor(l_rgb, cv2.COLOR_BGR2RGB)
        r_rgb = cv2.cvtColor(r_rgb, cv2.COLOR_BGR2RGB)
        
        l_mask = cv2.imread(l_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        r_mask = cv2.imread(r_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0

        if self.resize512:
            l_rgb = cv2.resize(l_rgb, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            r_rgb = cv2.resize(r_rgb, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            l_mask = cv2.resize(l_mask, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            r_mask = cv2.resize(r_mask, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            # l_depth = cv2.resize(l_depth, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

        l_rgb[l_mask < 0.5] = 0
        r_rgb[r_mask < 0.5] = 0
        # l_depth[l_mask < 0.5] = 0

        temp_dict = {
            "l_rgb":self.rgb_transform(l_rgb),
            "r_rgb":self.rgb_transform(r_rgb),
            "l_mask":self.to_tensor_transform(l_mask),
            "r_mask":self.to_tensor_transform(r_mask),
            # "l_depth":self.to_tensor_transform(l_depth),
            "l_rgb_path":l_rgb_path
        }
        
        return temp_dict

if __name__ == "__main__":
    test_data_dir = "/data2/hongyang/StereoPIFu/GenTestData/TestRes/ColorMesh"

    tt = EvaDataset(test_data_dir, transform_rgb=True, resize512 = False)
    print(len(tt))
    for i in range(10):
        data = tt[i]
        print(data["l_rgb"].size())
        print(data["r_rgb"].size())
        print(data["l_mask"].size())
        print(data["r_mask"].size())
        # print(data["l_depth"].size())
        # print(data["l_conf"].size())
        # print(data["sk_disp"].size())
        # print(data["sk_conf"].size())
        # print(data["l_mask_erosion"].size())
        rgb_img = (data["l_rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./temp_res/rrgb.png", rgb_img)
        # cv2.imwrite("../temp_res/ero_mask.png", data["l_mask_erosion"][0].numpy().astype(np.uint8))
        exit(0)