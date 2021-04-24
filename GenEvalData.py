import openmesh as om
import torch
import cv2
import numpy as np
from ZEDCamera import zed_camera_regu
import sys
import os
from os.path import join

sys.path.append("GenEvalData")
import RenderUtils
    
class RenderColorMesh(object):
    def __init__(self) -> None:
        super().__init__()
        self.B_MIN = np.array([-1.0, -0.5, -1.0])
        self.B_MAX = np.array([1.0, 2.28, 1.0])
        self.light_dire = [0.0, 0.0, -1.0]
        self.ambient_strength = 0.4
        self.light_strength = 0.6

    def render_tex_mesh_func(self, fv_indices, tri_uvs, tri_normals, tex_img, vps, camera_dict):
        
        proj_pixels, z_vals, v_status = self.project_mesh_vps(vps, camera_dict)
        tri_proj_pixels = (proj_pixels[fv_indices]).reshape(-1, 6) 
        tri_z_vals = z_vals[fv_indices] #[n_f, 3]
        tri_status = (v_status[fv_indices]).all(axis=1) #[n_f]

        cam_w = camera_dict["CameraReso"][0]
        cam_h = camera_dict["CameraReso"][1]
        ex_mat = camera_dict["ExterMat"]
        
        depth_img = np.ones((cam_h, cam_w), np.float32) * 100.0
        rgb_img = np.zeros((cam_h, cam_w, 3), np.float32)
        mask_img = np.zeros((cam_h, cam_w), np.int32)
        
        w_light_dx = self.light_dire[0]
        w_light_dy = self.light_dire[1]
        w_light_dz = self.light_dire[2]
        
        c_light_dx = ex_mat[0, 0] * w_light_dx + ex_mat[0, 1] * w_light_dy + ex_mat[0, 2] * w_light_dz
        c_light_dy = ex_mat[1, 0] * w_light_dx + ex_mat[1, 1] * w_light_dy + ex_mat[1, 2] * w_light_dz
        c_light_dz = ex_mat[2, 0] * w_light_dx + ex_mat[2, 1] * w_light_dy + ex_mat[2, 2] * w_light_dz
        
        ambient_strength = self.ambient_strength
        light_strength = self.light_strength
        
        RenderUtils.render_tex_mesh(
            tri_normals, tri_uvs, tri_proj_pixels, tri_z_vals, tri_status, tex_img, depth_img, rgb_img, mask_img,
            c_light_dx,c_light_dy,c_light_dz,ambient_strength,light_strength
        )
        
        depth_img[mask_img < 0.5] = 0
        return rgb_img, depth_img, mask_img
    
    
    def render_tex_mesh(self, mesh_path, tex_path, save_dir, base_name):
        
        om_mesh = om.read_trimesh(mesh_path, halfedge_tex_coord = True)
        
        #Vertex Position
        vps = om_mesh.points()
        # vps = om_mesh.points() 
        if not self.check_mesh_bbox(vps):
            print("Error, the bounding box of the mesh is out of the pre-defined range.")
            exit(0)
        
        n_f = om_mesh.n_faces()
        fv_indices = om_mesh.face_vertex_indices()
        fh_indices = om_mesh.face_halfedge_indices()
        
        #Face texture2D UV
        he_uv = om_mesh.halfedge_texcoords2D()
        tri_uvs = (he_uv[fh_indices]).reshape(n_f, 6)
        
        #Normal
        om_mesh.request_face_normals()
        om_mesh.request_vertex_normals()
        om_mesh.update_normals()
        vns = om_mesh.vertex_normals()
        tri_normals = (vns[fv_indices]).reshape(n_f, 9)
        
        #texture image
        tex_img = cv2.imread(tex_path).astype(np.float32)/255.0
        tex_img = np.ascontiguousarray(tex_img[:, :, ::-1]) #BGR to RGB
        
        l_rgb_img, l_depth_img, l_mask_img = self.render_tex_mesh_func(
            fv_indices, tri_uvs, tri_normals, tex_img, vps, zed_camera_regu.l_camera
        )
        r_rgb_img, r_depth_img, r_mask_img = self.render_tex_mesh_func(
            fv_indices, tri_uvs, tri_normals, tex_img, vps, zed_camera_regu.r_camera
        )
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        cv2.imwrite(join(save_dir, "L_RGB_%s.png"%base_name), (l_rgb_img * 255)[:,:,::-1])
        cv2.imwrite(join(save_dir, "L_depth_%s.png"%base_name), (l_depth_img * 10000).astype(np.uint16))
        cv2.imwrite(join(save_dir, "L_mask_%s.png"%base_name), (l_mask_img).astype(np.uint8))

        cv2.imwrite(join(save_dir, "R_RGB_%s.png"%base_name), (r_rgb_img * 255)[:,:,::-1])
        cv2.imwrite(join(save_dir, "R_depth_%s.png"%base_name), (r_depth_img * 10000).astype(np.uint16))
        cv2.imwrite(join(save_dir, "R_mask_%s.png"%base_name), (r_mask_img).astype(np.uint8))
    
    def render_color_mesh_func(self, fv_indices, tri_colors, tri_normals, vps, camera_dict):
        
        proj_pixels, z_vals, v_status = self.project_mesh_vps(vps, camera_dict)
        tri_proj_pixels = (proj_pixels[fv_indices]).reshape(-1, 6) #[n_f, 6]
        tri_z_vals = z_vals[fv_indices] #[n_f, 3]
        tri_status = (v_status[fv_indices]).all(axis=1) #[n_f]

        cam_w = camera_dict["CameraReso"][0]
        cam_h = camera_dict["CameraReso"][1]
        ex_mat = camera_dict["ExterMat"]
        
        depth_img = np.ones((cam_h, cam_w), np.float32) * 100.0
        rgb_img = np.zeros((cam_h, cam_w, 3), np.float32)
        mask_img = np.zeros((cam_h, cam_w), np.int32)
        
        w_light_dx = self.light_dire[0]
        w_light_dy = self.light_dire[1]
        w_light_dz = self.light_dire[2]
        
        c_light_dx = ex_mat[0, 0] * w_light_dx + ex_mat[0, 1] * w_light_dy + ex_mat[0, 2] * w_light_dz
        c_light_dy = ex_mat[1, 0] * w_light_dx + ex_mat[1, 1] * w_light_dy + ex_mat[1, 2] * w_light_dz
        c_light_dz = ex_mat[2, 0] * w_light_dx + ex_mat[2, 1] * w_light_dy + ex_mat[2, 2] * w_light_dz
        
        ambient_strength = self.ambient_strength
        light_strength = self.light_strength
        
        RenderUtils.render_color_mesh(
            tri_normals, tri_colors, tri_proj_pixels, tri_z_vals, tri_status, depth_img, rgb_img, mask_img,
            c_light_dx,c_light_dy,c_light_dz,ambient_strength,light_strength
        )
        depth_img[mask_img < 0.5] = 0.0
        return rgb_img, depth_img, mask_img
    
    def render_color_mesh(self, mesh_path, save_dir, base_name):
                        
        om_mesh = om.read_trimesh(mesh_path, vertex_color = True)
        
        #Vertex Position
        vps = om_mesh.points()
        if not self.check_mesh_bbox(vps):
            print("Error, the bounding box of the mesh is out of the pre-defined range.")
            exit(0)
        
        n_f = om_mesh.n_faces()
        fv_indices = om_mesh.face_vertex_indices()
        
        #Normal
        om_mesh.request_face_normals()
        om_mesh.request_vertex_normals()
        om_mesh.update_normals()
        vns = om_mesh.vertex_normals()
        tri_normals = (vns[fv_indices]).reshape(n_f, 9)
        
        #Color
        vcs = np.ascontiguousarray(om_mesh.vertex_colors()[:, :3])
        tri_colors = (vcs[fv_indices]).reshape(n_f, 9) 
        
        l_rgb_img, l_depth_img, l_mask_img = self.render_color_mesh_func(
            fv_indices, tri_colors, tri_normals, vps, zed_camera_regu.l_camera
        )
        r_rgb_img, r_depth_img, r_mask_img = self.render_color_mesh_func(
            fv_indices, tri_colors, tri_normals, vps, zed_camera_regu.r_camera
        )
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        cv2.imwrite(join(save_dir, "L_RGB_%s.png"%base_name), (l_rgb_img * 255)[:,:,::-1])
        cv2.imwrite(join(save_dir, "L_depth_%s.png"%base_name), (l_depth_img * 10000).astype(np.uint16))
        cv2.imwrite(join(save_dir, "L_mask_%s.png"%base_name), (l_mask_img).astype(np.uint8))

        cv2.imwrite(join(save_dir, "R_RGB_%s.png"%base_name), (r_rgb_img * 255)[:,:,::-1])
        cv2.imwrite(join(save_dir, "R_depth_%s.png"%base_name), (r_depth_img * 10000).astype(np.uint16))
        cv2.imwrite(join(save_dir, "R_mask_%s.png"%base_name), (r_mask_img).astype(np.uint8))
    
    def check_mesh_bbox(self, vps):
        min_vp = vps.min(axis=0)
        max_vp = vps.max(axis=0)
        
        res = (min_vp > self.B_MIN) * (max_vp < self.B_MAX)
        res = res.all()
        
        return res
        
    
    def project_mesh_vps(self, world_vps, camera_dict):
        ex_mat = camera_dict["ExterMat"]
        in_mat = camera_dict["InterMat"]
        cam_reso = camera_dict["CameraReso"]

        cam_w = cam_reso[0]
        cam_h = cam_reso[1]
        ex_Rmat = ex_mat[:3, :3]
        ex_Tvec = ex_mat[:3, 3:]
        
        fx = in_mat[0, 0]
        fy = in_mat[1, 1]
        cx = in_mat[0, 2]
        cy = in_mat[1, 2]
        
        cam_vps = ex_Rmat.dot(world_vps.T) + ex_Tvec
        pixel_x = fx * (cam_vps[0, :] / cam_vps[2, :]) + cx
        pixel_y = fy * (cam_vps[1, :] / cam_vps[2, :]) + cy
        
        vps_status = (pixel_x > 0) * (pixel_x < cam_w) * (pixel_y > 0) * (pixel_y < cam_h)
        proj_pixel = np.stack([pixel_x, pixel_y], axis=1)
        
        return proj_pixel, cam_vps[2, :], vps_status

if __name__ == "__main__":
    # color_mesh_path = "normalized_mesh_0012.off"
    # tt = RenderColorMesh()
    # tt.render_color_mesh(color_mesh_path, "./TestRes/ColorMesh", "color")
    
    tex_mesh_path = "TempData/SampleData/rp_dennis_posed_004_100k.obj"
    tex_img_path = "TempData/SampleData/rp_dennis_posed_004_dif_2k.jpg"
    tt = RenderColorMesh()
    tt.render_tex_mesh(tex_mesh_path, tex_img_path, "./TempData/TexMesh", "tex")