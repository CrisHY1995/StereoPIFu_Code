'''
@Description: Generate New Camera
@Author: HongYang
@Email: hymath@mail.ustc.edu.cn
@Date: 2020-07-08 16:12:21
LastEditTime: 2020-10-07 10:25:56
'''

import cv2
import numpy as np

class ZEDCamera(object):
    def __init__(self, reso_X_Y, crop_X_Y_ori, crop_X_Y_lenght, position_3d, normalize_camera):
        super().__init__()
        
        self.normalize_camera = normalize_camera
        self.L_camera_path = "./ConfigData/L_Cam_1920x1080_30fps.yml"
        self.R_camera_path = "./ConfigData/R_Cam_1920x1080_30fps.yml"

        self.position_3d = position_3d
        
        self.crop_X_Y_ori = crop_X_Y_ori
        self.crop_X_Y_lenght = crop_X_Y_lenght
        
        self.reso_X_Y = reso_X_Y
        
        self._build_camera_para(is_left=True)
        self._build_camera_para(is_left=False)
        self._calc_baseline()
        
    def _build_camera_para(self, is_left):
        
        if is_left:
            fs = cv2.FileStorage(self.L_camera_path, cv2.FILE_STORAGE_READ)
        else:
            fs = cv2.FileStorage(self.R_camera_path, cv2.FILE_STORAGE_READ)
        
        ExterMat = fs.getNode("ExterMat").mat().astype(np.float32)
        InterMat = fs.getNode("InterMat").mat().astype(np.float32)
        CameraReso = fs.getNode("CameraSize").mat().astype(np.float32)
        fs.release()
        
        ExterMat[0, 3] += self.position_3d[0]
        ExterMat[1, 3] += self.position_3d[1]
        ExterMat[2, 3] += self.position_3d[2]

        InterMat[0, 2] = InterMat[0, 2] - self.crop_X_Y_ori[0]
        InterMat[1, 2] = InterMat[1, 2] - self.crop_X_Y_ori[1]
        
        if self.normalize_camera:
            InterMat[0, :] = 2.0 * InterMat[0, :]  / self.crop_X_Y_lenght[0]
            InterMat[1, :] = 2.0 * InterMat[1, :] / self.crop_X_Y_lenght[1]
            InterMat[:2, 2] = InterMat[:2, 2] - 1.0
        else:
            InterMat[0, :] = InterMat[0, :] * self.reso_X_Y[0] / self.crop_X_Y_lenght[0]
            InterMat[1, :] = InterMat[1, :] * self.reso_X_Y[1] / self.crop_X_Y_lenght[1]
        
        CameraReso = np.array([self.reso_X_Y[0], self.reso_X_Y[1]])
        
        if is_left:
            self.l_camera = {
                "ExterMat":ExterMat,
                "InterMat":InterMat,
                "CameraReso":CameraReso,
            }
        else:
            self.r_camera = {
                "ExterMat":ExterMat,
                "InterMat":InterMat,
                "CameraReso":CameraReso,
            }

    def Calc_Z(self, one_point, use_l_camera):
        if use_l_camera:
            exter_mat = self.l_camera["ExterMat"]
        else:
            exter_mat = self.r_camera["ExterMat"]
        one_point = one_point.reshape(3, 1)
        res = exter_mat[:3, :3].dot(one_point) + exter_mat[:3, 3:]
        return res[2, 0]
    
    def _calc_baseline(self):
        self.baseline = abs(self.r_camera["ExterMat"][0, 3] - self.l_camera["ExterMat"][0, 3]) * self.r_camera["InterMat"][0,0]
        
    def print_camera_info(self):
        print("=========L_camera===========")
        print()
        print("L_ExterMat")
        print(self.l_camera["ExterMat"])
        print()
        print("L_InterMat")
        print(self.l_camera["InterMat"])
        print()
        print("L_CameraReso")
        print(self.l_camera["CameraReso"])
        print()
        
        print("=========R_camera===========")
        print("R_ExterMat")
        print(self.r_camera["ExterMat"])
        print()
        print("R_InterMat")
        print(self.r_camera["InterMat"])
        print()
        print("R_CameraReso")
        print(self.r_camera["CameraReso"])

zed_camera_normalized = ZEDCamera(reso_X_Y=[576, 576],
                       crop_X_Y_ori=[350, 0],
                       crop_X_Y_lenght=[1080, 1080], 
                       position_3d=[0, 0.9, 2.88],
                       normalize_camera=True)

zed_camera_regu = ZEDCamera(reso_X_Y=[576, 576],
                       crop_X_Y_ori=[350, 0],
                       crop_X_Y_lenght=[1080, 1080], 
                       position_3d=[0, 0.9, 2.88],
                       normalize_camera=False)

# fs = cv2.FileStorage("./temp_res/l_cam_576_para.yml", cv2.FILE_STORAGE_WRITE)
# fs.write('ExterMat', zed_camera_regu.l_camera["ExterMat"])
# fs.write('InterMat', zed_camera_regu.l_camera["InterMat"])
# fs.write('CameraReso', zed_camera_regu.l_camera["CameraReso"])
# fs.release()

# fs = cv2.FileStorage("./temp_res/r_cam_576_para.yml", cv2.FILE_STORAGE_WRITE)
# fs.write('ExterMat', zed_camera_regu.r_camera["ExterMat"])
# fs.write('InterMat', zed_camera_regu.r_camera["InterMat"])
# fs.write('CameraReso', zed_camera_regu.r_camera["CameraReso"])
# fs.release()

# print(zed_camera.baseline)
# zed_camera_regu.print_camera_info()