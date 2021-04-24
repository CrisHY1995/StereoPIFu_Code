import torch
from EvaDataset import EvaDataset
from torch.utils.data import DataLoader
from utils import ImgNormalizationAndInv

import numpy as np
from StereoPIFuNet import StereoPIFuNet

from skimage import measure
import os
from tqdm import tqdm
from os.path import join

class EvalResult(object):
    def __init__(self, model_path, save_dir, gpu_idx, test_data_dir, start_idx = 0, end_idx = -1) -> None:
        super().__init__()

        self.use_bb_of_rootjoint = True
        self.use_VisualHull = True

        self.device = torch.device("cuda:%d"%gpu_idx)

        self.RGB_mean = [0.485, 0.456, 0.406]
        self.RGB_std = [0.229, 0.224, 0.225]

        self.B_MIN = np.array([-1.0, -0.5, -1.0])
        self.B_MAX = np.array([1.0, 2.28, 1.0])
        
        self.resolution = 512
        self.use_octree = True
        self.NumOfVP_PerBatch = 5000
        self.test_data_dir = test_data_dir
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.model_path = model_path
        self.save_root_dir = save_dir

        self.build_tool_funcs()
        self.build_net()
        self.build_EvaDataset()

    def build_tool_funcs(self):
        self.img_normalizer = ImgNormalizationAndInv(self.RGB_mean, self.RGB_std).to(self.device)

    def build_EvaDataset(self):
        data_set = EvaDataset(self.test_data_dir, transform_rgb=True, resize512=False, start_idx=self.start_idx, end_idx=self.end_idx)
        self.data_loader = DataLoader(data_set, batch_size=1, num_workers=1, shuffle=False)

    def build_net(self):
        self.netG = StereoPIFuNet(self.use_VisualHull, self.use_bb_of_rootjoint).to(self.device)
        self.netG.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)

    @staticmethod
    def save_obj_mesh(save_path, verts, faces, colors = None, inv_order = True):
        file = open(save_path, 'w')

        if colors is not None:
            for idx, v in enumerate(verts):
                c = colors[idx]
                file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
        else:
            for idx, v in enumerate(verts):
                file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        
        for f in faces:
            f_plus = f + 1
            if inv_order:
                file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
            else:
                file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
        file.close()

    def eval_func(self, points):
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=self.device).float()
        pred = self.netG.query(samples)[0][0]
        return pred.detach().cpu().numpy()

    def batch_eval(self, points):
        num_pts = points.shape[1]
        sdf = np.zeros(num_pts)

        num_samples = self.NumOfVP_PerBatch

        num_batches = num_pts // num_samples
        for i in range(num_batches):
            sdf[i * num_samples:i * num_samples + num_samples] \
                = self.eval_func(points[:, i * num_samples:i * num_samples + num_samples])
        
        if num_pts % num_samples:
            sdf[num_batches * num_samples:] = self.eval_func(points[:, num_batches * num_samples:])
        
        return sdf

    def create_grid(self):
        resX = self.resolution
        resY = self.resolution
        resZ = self.resolution
        b_min = self.B_MIN
        b_max = self.B_MAX

        coords = np.mgrid[:resX, :resY, :resZ]
        coords = coords.reshape(3, -1)
        coords_matrix = np.eye(4)
        length = b_max - b_min
        coords_matrix[0, 0] = length[0] / resX
        coords_matrix[1, 1] = length[1] / resY
        coords_matrix[2, 2] = length[2] / resZ
        coords_matrix[0:3, 3] = b_min
        coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]

        coords = coords.reshape(3, resX, resY, resZ)
        return coords, coords_matrix

    def ReconstrucMesh(self, temp_data, save_path = None):
        l_rgb = temp_data["l_rgb"].to(self.device)
        r_rgb = temp_data["r_rgb"].to(self.device)
        l_mask = temp_data["l_mask"].to(self.device)
        r_mask = temp_data["r_mask"].to(self.device)

        self.netG.update_feature_by_imgs(l_rgb, r_rgb, l_mask, r_mask)

        resolution = self.resolution
        coords, mat = self.create_grid()
        resolutionXYZ = (resolution, resolution, resolution)

        if self.use_octree:
            init_resolution=64
            threshold = 0.01

            sdf = np.zeros(resolutionXYZ)
            dirty = np.ones(resolutionXYZ, dtype=np.bool)
            grid_mask = np.zeros(resolutionXYZ, dtype=np.bool)

            reso = resolutionXYZ[0] // init_resolution

            while reso > 0:
                # subdivide the grid
                grid_mask[0:resolutionXYZ[0]:reso, 0:resolutionXYZ[1]:reso, 0:resolutionXYZ[2]:reso] = True
                # test samples in this iteration
                test_mask = np.logical_and(grid_mask, dirty)
                #print('step size:', reso, 'test sample size:', test_mask.sum())
                points = coords[:, test_mask]

                sdf[test_mask] = self.batch_eval(points)
                dirty[test_mask] = False

                # do interpolation
                if reso <= 1:
                    break
                for x in range(0, resolutionXYZ[0] - reso, reso):
                    for y in range(0, resolutionXYZ[1] - reso, reso):
                        for z in range(0, resolutionXYZ[2] - reso, reso):
                            # if center marked, return
                            if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                                continue
                            v0 = sdf[x, y, z]
                            v1 = sdf[x, y, z + reso]
                            v2 = sdf[x, y + reso, z]
                            v3 = sdf[x, y + reso, z + reso]
                            v4 = sdf[x + reso, y, z]
                            v5 = sdf[x + reso, y, z + reso]
                            v6 = sdf[x + reso, y + reso, z]
                            v7 = sdf[x + reso, y + reso, z + reso]
                            v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                            v_min = v.min()
                            v_max = v.max()
                            # this cell is all the same
                            if (v_max - v_min) < threshold:
                                sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                                dirty[x:x + reso, y:y + reso, z:z + reso] = False
                reso //= 2
        else:
            coords = coords.reshape([3, -1])
            sdf = self.batch_eval(coords)


        sdf = sdf.reshape(resolutionXYZ)

        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)

        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        # verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=self.device).float()
        # uv, _ = self.netG.l_projection(verts_tensor)

        # mean=torch.FloatTensor(self.Stereo_RGB_mean).view(1,3,1,1).to(self.device)
        # std=torch.FloatTensor(self.Stereo_RGB_std).view(1,3,1,1).to(self.device)
        # color_map = (l_rgb * std) + mean
        # color = self.netG.index2D(color_map, uv).detach().cpu().numpy()[0].T
        
        if save_path is not None:
            self.save_obj_mesh(save_path, verts, faces)
            # self.save_obj_mesh(save_path, verts, faces, color)
            
    def eval_evadataset(self):
        self.netG.eval()

        save_root_dir = self.save_root_dir
        
        if not os.path.exists(save_root_dir):
            os.mkdir(save_root_dir)

        for data_dict in tqdm(self.data_loader):

            l_rgb_path = data_dict["l_rgb_path"][0]
            save_dir = join(save_root_dir, l_rgb_path.split("/")[-2])

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            mesh_save_path = join(save_dir, "mesh_%s.obj"%l_rgb_path.split("/")[-1][6:-4])
            self.ReconstrucMesh(data_dict, mesh_save_path)

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gpu_idx", type=int, required=True)
    
    opt = parser.parse_args()

    model_path = opt.model_path
    save_dir = opt.save_dir
    test_data_dir = opt.test_data_dir
    gpu_idx = opt.gpu_idx
    
    # start_idx = int(sys.argv[5])
    # end_idx = int(sys.argv[6])
    
    # tt = EvalResult(model_path, save_dir, gpu_idx, test_data_dir, start_idx, end_idx)
    tt = EvalResult(model_path, save_dir, gpu_idx, test_data_dir)
    tt.eval_evadataset()