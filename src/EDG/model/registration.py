import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from skimage import io
from PIL import Image
from timeit import default_timer as timer
import datetime
import argparse
from copy import deepcopy
from .geometry import *
from time import time

import yaml
import matplotlib.pyplot as plt
import torch
# from lietorch import SO3, SE3, LieGroupParameter
import torch.optim as optim
from .loss import *
from .point_render import PCDRender

def R_euler_angle_batch(theta): # [..., 3]

    ori_shape = theta.shape
    theta = theta.reshape(-1, 3)
    res = R_euler_angle(theta)
    return res.reshape(*ori_shape[:-1], 3, 3)

def R_euler_angle(theta_):
    theta = torch.stack([theta_[:,0], -theta_[:,1], theta_[:,2]], dim=1)
    n = theta.shape[0]
    R1 = torch.eye(3).repeat(n, 1, 1).to(theta.device)
    R2 = torch.eye(3).repeat(n, 1, 1).to(theta.device)
    R3 = torch.eye(3).repeat(n, 1, 1).to(theta.device)

    R1[:, 0, 0] = torch.cos(theta[:, 2])  # z
    R1[:, 0, 1] = -torch.sin(theta[:, 2])
    R1[:, 1, 0] = torch.sin(theta[:, 2])
    R1[:, 1, 1] = torch.cos(theta[:, 2])

    R2[:, 0, 0] = torch.cos(theta[:, 1])  # y
    R2[:, 0, 2] = -torch.sin(theta[:, 1])
    R2[:, 2, 0] = torch.sin(theta[:, 1])
    R2[:, 2, 2] = torch.cos(theta[:, 1])

    R3[:, 1, 1] = torch.cos(theta[:, 0])  # x
    R3[:, 1, 2] = -torch.sin(theta[:, 0])
    R3[:, 2, 1] = torch.sin(theta[:, 0])
    R3[:, 2, 2] = torch.cos(theta[:, 0])

    return R3 @ R2 @ R1


def Rx(R, x):
    # print("Rx", R.shape, x.shape)
    return (R * x.unsqueeze(-2)).sum(-1)

def get_mesh_min_dis(mesh):

    min_dis = [1e10 for i in range(len(mesh.vertices))]
    for t in np.asarray(mesh.triangles):
        v1, v2, v3 = mesh.vertices[t[0]], mesh.vertices[t[1]], mesh.vertices[t[2]]
        min_dis[t[0]] = min(min_dis[t[0]], np.linalg.norm(v1 - v2))
        min_dis[t[1]] = min(min_dis[t[1]], np.linalg.norm(v2 - v3))
        min_dis[t[2]] = min(min_dis[t[2]], np.linalg.norm(v3 - v1))
    # print('min_dis', np.mean(min_dis), np.std(min_dis))  # 0.0007826771030806771 0.00045625597323787557
    return torch.Tensor(min_dis).cuda()

def mv(m, v): # [3,3], [N,3]
    assert v.ndim == 2
    vv = (m[:3, :3] * v[:,None]).sum(-1) + m[:3,3]
    return vv

def convert_from_xyz(xyz, fx, fy, cx, cy):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    x_over_z = x / z
    y_over_z = y / z
    xx = x_over_z * fx + cx
    yy = y_over_z * fy + cy
    return torch.stack([xx, yy], -1)

def cam_project(cam_pos, p_3d, fx, fy, cx, cy):
    source_canonical_kpt1_3d = mv(torch.linalg.inv(cam_pos), p_3d)
    pixel_src = convert_from_xyz(source_canonical_kpt1_3d, fx, fy, cx, cy)
    return pixel_src

class Registration():


    def __init__(self, depth_image_path, K, config, extra_info):


        self.device = config.device

        self.deformation_model = config.deformation_model
        self.intrinsics = K

        self.extra_info = extra_info

        self.config = config

        """initialize deformation graph"""
        depth_image = io.imread(depth_image_path)
        image_size = (depth_image.shape[0], depth_image.shape[1])
        data = get_deformation_graph_from_depthmap( depth_image, K, extra_info = extra_info)
        self.graph_edges = data['graph_edges'].to(self.device)
        self.graph_edges_weights = data['graph_edges_weights'].to(self.device)
        self.graph_clusters = data['graph_clusters'] #.to(self.device)

        self.mesh_original = data['mesh_original']


        # """initialize point clouds"""
        # valid_pixels = torch.sum(data['pixel_anchors'], dim=-1) > -4
        # self.source_pcd = data["point_image"][valid_pixels].to(self.device)
        # self.point_anchors = data["pixel_anchors"][valid_pixels].long().to(self.device)
        # self.anchor_weight = data["pixel_weights"][valid_pixels].to(self.device)
        # self.anchor_loc = data["graph_nodes"][self.point_anchors].to(self.device)
        # self.frame_point_len = [ len(self.source_pcd)]
        # print(valid_pixels.shape, self.source_pcd.shape) # [H, W], [H * W, 3]
        # input('ahaa')

        """initialize point clouds"""
        # valid_pixels = torch.sum(data['pixel_anchors'], dim=-1) > -4
        # self.source_pcd = data["point_image"][valid_pixels].to(self.device)
        self.vertices = torch.Tensor(np.asarray(data["mesh_original"].vertices)).cuda()
        self.mesh = data["mesh_original"]
        self.min_dis_mask = data["min_dis_mask"].cuda()
        self.V_fixed_id_list = data["V_fixed_id_list"].cuda()
        self.p_fixed_list = data["p_fixed_list"].cuda()
        self.point_anchors = data["pixel_anchors"].long().to(self.device)
        self.anchor_weight = data["pixel_weights"].to(self.device)
        self.point_anchors_all = data["pixel_anchors_all"].long().to(self.device)
        self.anchor_weight_all = data["pixel_weights_all"].to(self.device)
        self.anchor_loc = data["graph_nodes"][self.point_anchors].to(self.device)
        self.graph_nodes = data["graph_nodes"].to(device=self.device)
        self.node_indices = data["node_indices"].to(device=self.device)
        self.data_name = data["data_name"]
        self.data_basename_src = data["data_basename_src"]
        self.canonical_cam = data["canonical_cam"]
        self.cam_intrinsics = data["cam_intrinsics"]
        self.set_zero = extra_info["set_zero"]
        # self.graph_nodes_control_bool = data["graph_nodes_control_bool"].to(device=self.device)
        # self.frame_point_len = [ len(self.source_pcd)]

        self.post_name = ""
        if self.extra_info['mesh_type'] != "tetra":
            self.post_name = "_" + self.extra_info['mesh_type']

        """pixel to pcd map"""
        # self.pix_2_pcd_map = [ self.map_pixel_to_pcd(valid_pixels).to(config.device) ]


        """define differentiable pcd renderer"""
        self.renderer = PCDRender(K, img_size=image_size)


    def register_a_depth_frame(self, tgt_depth_path, landmarks=None):
        """
        :param tgt_depth_path:
        :return:
        """

        """load target frame"""
        tgt_depth = io.imread( tgt_depth_path )/1000.
        depth_mask = torch.from_numpy(tgt_depth > 0)
        tgt_pcd = depth_2_pc(tgt_depth, self.intrinsics).transpose(1,2,0)
        self.tgt_pcd = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float().to(self.device)
        pix_2_pcd = self.map_pixel_to_pcd( depth_mask ).to(self.device)

        if landmarks is not None:
            s_uv , t_uv = landmarks
            s_id = self.pix_2_pcd_map[-1][ s_uv[:,1], s_uv[:,0] ]
            t_id = pix_2_pcd [ t_uv[:,1], t_uv[:,0]]
            valid_id = (s_id>-1) * (t_id>-1)
            s_ldmk = s_id[valid_id]
            t_ldmk = t_id[valid_id]

            landmarks = (s_ldmk, t_ldmk)

        self.visualize_results(self.tgt_pcd)
        warped_pcd = self.solve(  landmarks=landmarks)
        self.visualize_results( self.tgt_pcd, warped_pcd)

    def mesh_nerf2original(self, mesh_):

        from copy import deepcopy
        mesh = deepcopy(mesh_)
        rotation = torch.eye(4)
        rotation[1:3, 1:3] = torch.Tensor([[0, -1], [1, 0]])

        transform_matrix_, camera_to_world, scale_factor = torch.load(f"../nerfstudio/camera_info/{self.data_basename_src}_train.pt")
        transform_matrix = torch.eye(4)
        transform_matrix[:3] = transform_matrix_[:3].cpu()

        mesh.scale(1.0 / scale_factor, center = [0,0,0])
        mesh.transform(torch.linalg.inv(transform_matrix))
        mesh.transform(torch.linalg.inv(rotation))
        # mesh.compute_vertex_normals()

        return mesh


    def solve_main(self): 
        """
        :param tgt_depth_path:
        :return:
        """
        # self.visualize_results(self.tgt_pcd)
        new_mesh, line_warp = self.solve2()
        print('data_name', self.data_name)
        mesh_name = f"./mesh_dg/mesh_dg_{self.data_name}_{self.extra_info['score_thres']}{self.post_name}.obj"
        mesh_original_name = f"./mesh_dg/mesh_dg_{self.data_name}_original_size_{self.extra_info['score_thres']}{self.post_name}.obj"
        o3d.io.write_triangle_mesh(mesh_name, new_mesh)
        mesh_original = self.mesh_nerf2original(new_mesh)
        o3d.io.write_triangle_mesh(mesh_original_name, mesh_original)
        
        # self.vis_mesh([new_mesh], [line_warp])

    def vis_mesh(self, mesh_list, other_list):

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for mesh in mesh_list:
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)
        for other in other_list:
            vis.add_geometry(other)
        vis.run()
        vis.destroy_window()
        
    def solve(self, **kwargs ):

        if self.deformation_model == "ED":
            # Embeded_deformation, c.f. https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
            return self.optimize_ED(**kwargs)

    def get_mesh_new_pose(self):

        node_all = self.graph_nodes[self.point_anchors_all].cuda() # [M, 8, 3] # point_anchors_all [540479, 8]
        R_mat = R_euler_angle_batch(self.R) # [N, 3, 3]
        anchor_rot_all = R_mat[self.point_anchors_all]
        # print('get mesh debug', self.R.shape, R_mat.shape, self.point_anchors_all.shape, anchor_rot_all.shape) # torch.Size([2359, 3]) torch.Size([2359, 3, 3]) torch.Size([540479, 8]) torch.Size([540479, 8, 3, 3]) 
        anchor_trn_all = self.t[self.point_anchors_all]
        # torch.save([node_all, anchor_rot_all, anchor_trn_all, self.vertices, self.anchor_weight_all], 'debug.pt')
        # input('debug')
        warped_vertices = ED_warp(self.vertices, node_all, anchor_rot_all, anchor_trn_all, self.anchor_weight_all)
        return warped_vertices
    
    def clean_mesh(self, mesh):
        # mesh's vertices contains some NaN, use average of adjacent vertices to remove.
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        edges = [[] for i in range(vertices.shape[0])]
        for t in triangles:
            edges[t[0]].append(t[1])
            edges[t[0]].append(t[2])
            edges[t[1]].append(t[0])
            edges[t[1]].append(t[2])
            edges[t[2]].append(t[0])
            edges[t[2]].append(t[1])
        for id in range(10000):
            exit_flag = 1
            clean_cnt = 0
            for i in range(vertices.shape[0]):
                if np.isnan(vertices[i]).any():
                    exit_flag = 0
                    clean_cnt += 1
                    surrounding_v = vertices[edges[i]]
                    surrounding_v = surrounding_v[~np.isnan(surrounding_v).any(-1)]
                    if surrounding_v.shape[0] == 0:
                        exit_flag = 1
                        print('no surrounddings')
                        break
                    vertices[i] = surrounding_v.mean(0)
            if exit_flag:
                break
            print('clean_cnt', id, clean_cnt)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        

    def solve2(self):
        '''
        :param landmarks:
        :return:
        '''

        """translations"""
        node_translations = torch.zeros_like(self.graph_nodes).uniform_(-1e-6, 1e-6).cuda()
        self.t = torch.nn.Parameter(node_translations)
        self.t.requires_grad = True

        """rotations"""
        phi = torch.zeros_like(self.graph_nodes).cuda()
        # print(phi.shape) # 266, 3
        self.R = torch.nn.Parameter(phi, requires_grad=True)
        # node_rotations = SO3.exp(phi)
        # self.R = LieGroupParameter(node_rotations)
        # warped_vertices = self.get_mesh_new_pose()
        # self.mesh.vertices = o3d.utility.Vector3dVector(warped_vertices.detach().cpu().numpy())
        # return self.mesh
        # input('aha')

        """optimizer setup"""
        optimizer = optim.Adam([self.R, self.t], lr= self.config.lr )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)


        """render reference pcd"""
        # sil_tgt, d_tgt, _ = self.render_pcd(self.tgt_pcd)
        source_v = self.vertices[self.V_fixed_id_list].cuda()
        target_v = self.p_fixed_list.cuda()

        # rigid_coeff = 1000
        # rigid_sum = 0
        # for node_id, edges in enumerate(self.graph_edges):
        #     for id, neighbor_id in enumerate(edges):
        #         if neighbor_id == -1:
        #             break
        #         if self.graph_nodes_control_bool[node_id] and self.graph_nodes_control_bool[neighbor_id]:
        #             rigid_sum += 1
        #             self.graph_edges_weights[node_id, id] *= rigid_coeff

        # for i in range(self.graph_edges.shape[0]):
        #     e = self.graph_edges[i]
        #     if self.graph_nodes_control_bool[e[0]] and self.graph_nodes_control_bool[e[1]]:
        #         self.graph_edges_weights[i] *= 10
        #         rigid_sum += 1
        # print('rigid_sum', rigid_sum, self.graph_nodes_control_bool.shape, self.graph_edges.shape)
        # self.graph_edges_weights
        
        # Transform points
        line_warp = o3d.geometry.LineSet()
        n_iter = 3000
        for i in range(n_iter):#self.config.iters):

            anchor_trn = self.t [self.point_anchors] # point_anchors: [N, 6] kpt对应于哪些graph node id
            # anchor_rot = self.R [ self.point_anchors]
            # anchor_loc: (graph node 3D position)[self.point_anchors]
            # print('ar', anchor_rot.shape, self.point_anchors.shape)
            anchor_rot_all = R_euler_angle_batch(self.R[:])
            anchor_rot = anchor_rot_all[self.point_anchors]
            # print('debug', self.source_pcd.shape, self.anchor_loc.shape, anchor_rot.shape, anchor_trn.shape, self.anchor_weight.shape)
            # torch.Size([31501, 3]) torch.Size([31501, 6, 3]) torch.Size([31501, 6, 3, 3]) torch.Size([31501, 6, 3]) torch.Size([31501, 6])
            # warped_pcd = ED_warp(self.source_pcd, self.anchor_loc, anchor_rot, anchor_trn, self.anchor_weight)
            # err_control = control_cost(anchor_rot_all[self.graph_nodes_control_bool], self.t[self.graph_nodes_control_bool])
            warped_pcd = ED_warp(source_v.detach(), self.anchor_loc, anchor_rot, anchor_trn, self.anchor_weight)
            # print('debug', warped_pcd.mean(), target_v.mean(), anchor_rot.mean())
            err_arap = arap_cost(anchor_rot_all, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights, lietorch=False)
            err_edge = edge_cost(self.graph_nodes, self.t, self.graph_edges, self.graph_edges_weights)
            err_ldmk, err_ldmk_all = landmark_cost_p(warped_pcd, target_v.detach())
            warped_2d = cam_project(self.canonical_cam.detach(), warped_pcd, self.cam_intrinsics['fx'], self.cam_intrinsics['fy'], self.cam_intrinsics['cx'], self.cam_intrinsics['cy'])
            target_2d = cam_project(self.canonical_cam.detach(), target_v.detach(), self.cam_intrinsics['fx'], self.cam_intrinsics['fy'], self.cam_intrinsics['cx'], self.cam_intrinsics['cy'])
            err_ldmk_2d, err_ldmk_2d_all = landmark_cost_p(warped_2d, target_2d)
            

            # warped_vertices = self.get_mesh_new_pose()
            # print(torch.isnan(warped_vertices).float().mean(), 'man warped')
            # node_coors = self.t[:] + self.graph_nodes
            # loss_collision = collision_loss(node_coors, warped_vertices.detach(), self.min_dis_mask)
            
            if i == n_iter - 1:
                line_warp.points = o3d.utility.Vector3dVector(warped_pcd.detach().cpu().numpy().tolist() + target_v.detach().cpu().numpy().tolist())
                line_warp.lines = o3d.utility.Vector2iVector(np.array([[i, i + warped_pcd.shape[0]] for i in range(warped_pcd.shape[0])]).tolist())
                line_warp.paint_uniform_color([1, 0, 0])
            # sil_src, d_src, _ = self.render_pcd(warped_pcd)
            # err_silh = silhouette_cost(sil_src, sil_tgt) if self.config.w_silh > 0 else 0
            # err_depth = projective_depth_cost(d_src, d_tgt) if self.config.w_depth > 0 else 0

            # cd = chamfer_dist(warped_pcd, self.tgt_pcd) if self.config.w_chamfer > 0 else 0
            loss = {
                "arap_loss": err_arap * self.config.w_arap * 1.,
                "ldmk_loss": err_ldmk * self.config.w_ldmk * 0.1,
                # "ldmk_2d_loss": err_ldmk_2d * self.config.w_ldmk * 0.001 * (1.0 / 5000 * 4),
                # "collision_loss": loss_collision,
                # err_edge * 100 # + \
                # err_ldmk_all.max() * 10.
                # err_control * 0.01
                # err_silh * self.config.w_silh + \
                # err_depth * self.config.w_depth + \
                # cd * self.config.w_chamfer
            }
            loss = sum(loss.values())
            print('loss', i, loss.item(), 'arap', err_arap.item(), 'ldmk', err_ldmk.item(), 'ldmk_2d', err_ldmk_2d.item(), err_edge.item(), err_ldmk_all.max().item())# , err_control.item())
            # if loss.item() < 1e-7:
            #     break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if torch.isnan(loss).any() and self.set_zero:
            print('nan loss')
            self.R = torch.zeros_like(self.R)
            self.t = torch.zeros_like(self.t)

        warped_vertices = self.get_mesh_new_pose()
        new_mesh = deepcopy(self.mesh)
        new_mesh.vertices = o3d.utility.Vector3dVector(warped_vertices.detach().cpu().numpy())
        self.clean_mesh(new_mesh)

        valid_vertices = ~torch.isnan(self.anchor_weight_all.mean(-1))
        print(valid_vertices.shape, valid_vertices.sum()) # [N], N-100多
        g = self.graph_nodes[self.point_anchors_all][valid_vertices]
        v = self.vertices[valid_vertices].cpu()
        vd = warped_vertices[valid_vertices].cpu()
        # if loss is nan
        R = R_euler_angle_batch(self.R)[self.point_anchors_all][valid_vertices] # [N, 6, 3, 3]
        Rg = Rx(R, g)
        t = self.t[self.point_anchors_all][valid_vertices]
        valid_anc_w_all = self.anchor_weight_all[valid_vertices][:,:,None]
        g = (g * valid_anc_w_all).sum(1).cpu()
        Rg = (Rg * valid_anc_w_all).sum(1).cpu()
        # R = (R * valid_anc_w_all[:,:,:,None]).sum(1).cpu()
        print(R.shape, valid_anc_w_all.shape, g.shape, v.shape) # torch.Size([277144, 8, 3, 3]) torch.Size([277144, 8, 1]) torch.Size([277144, 3]) torch.Size([277144, 3
        R = calc_average_rotation_matrix(R, valid_anc_w_all.squeeze(-1)).cpu() # [N, 3, 3]
        g = v
        t = (t * valid_anc_w_all).sum(1).cpu()
        vd2 = Rx(R, v) - Rg.detach() + g.detach() + t
        v2 = Rx(torch.linalg.inv(R), vd - g - t + Rg)
        print('debug', (vd2 - vd).abs().max(), (v2 - v).abs().max())
        torch.save([v.float(), vd.float(), R.float(), Rg.float(), g.float(), t.float()], f'./dg/dg_{self.data_name}{self.post_name}.pt')
        # E = torch.Tensor(np.asarray(self.mesh.edges)).cuda()
        print('debugg', valid_vertices.shape, self.anchor_weight_all.shape, g.shape, R.shape, t.shape)#  E.shape)
        # torch.Size([519164]) torch.Size([519164, 8]) torch.Size([519164, 8, 3]) torch.Size([519164, 8, 3, 3]) torch.Size([519164, 8, 3])
        
        return new_mesh, line_warp

    def optimize_ED(self,  landmarks=None):
        '''
        :param landmarks:
        :return:
        '''

        """translations"""
        node_translations = torch.zeros_like(self.graph_nodes)
        self.t = torch.nn.Parameter(node_translations)
        self.t.requires_grad = True

        """rotations"""
        phi = torch.zeros_like(self.graph_nodes)
        print(phi.shape) # 266, 3
        self.R = torch.nn.Parameter(phi, requires_grad=True)
        # node_rotations = SO3.exp(phi)
        # self.R = LieGroupParameter(node_rotations)


        """optimizer setup"""
        optimizer = optim.Adam([self.R, self.t], lr= self.config.lr )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)


        """render reference pcd"""
        sil_tgt, d_tgt, _ = self.render_pcd(self.tgt_pcd)


        # Transform points
        for i in range(self.config.iters):

            anchor_trn = self.t [self.point_anchors]
            anchor_rot = self.R [ self.point_anchors]
            # print('ar', anchor_rot.shape, self.point_anchors.shape)
            anchor_rot = R_euler_angle_batch(anchor_rot)
            # print('debug', self.source_pcd.shape, self.anchor_loc.shape, anchor_rot.shape, anchor_trn.shape, self.anchor_weight.shape)
            # torch.Size([31501, 3]) torch.Size([31501, 6, 3]) torch.Size([31501, 6, 3, 3]) torch.Size([31501, 6, 3]) torch.Size([31501, 6])
            warped_pcd = ED_warp(self.source_pcd, self.anchor_loc, anchor_rot, anchor_trn, self.anchor_weight)

            err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)
            err_ldmk = landmark_cost(warped_pcd, self.tgt_pcd, landmarks) if landmarks is not None else 0

            # sil_src, d_src, _ = self.render_pcd(warped_pcd)
            # err_silh = silhouette_cost(sil_src, sil_tgt) if self.config.w_silh > 0 else 0
            # err_depth = projective_depth_cost(d_src, d_tgt) if self.config.w_depth > 0 else 0

            # cd = chamfer_dist(warped_pcd, self.tgt_pcd) if self.config.w_chamfer > 0 else 0

            loss = \
                err_arap * self.config.w_arap + \
                err_ldmk * self.config.w_ldmk * 0
                # err_silh * self.config.w_silh + \
                # err_depth * self.config.w_depth + \
                # cd * self.config.w_chamfer

            print( i, loss)
            if loss.item() < 1e-7:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return warped_pcd


    def render_pcd (self, x):
        INF = 1e+6
        px, dx = self.renderer(x)
        px, dx  = map(lambda feat: feat.squeeze(), [px, dx ])
        dx[dx < 0] = INF
        mask = px[..., 0] > 0
        return px, dx, mask


    def map_pixel_to_pcd(self, valid_pix_mask):
        ''' establish pixel to point cloud mapping, with -1 filling for invalid pixels
        :param valid_pix_mask:
        :return:
        '''
        image_size = valid_pix_mask.shape
        pix_2_pcd_map = torch.cumsum(valid_pix_mask.view(-1), dim=0).view(image_size).long() - 1
        pix_2_pcd_map [~valid_pix_mask] = -1
        return pix_2_pcd_map


    def visualize_results(self, tgt_pcd, warped_pcd=None):

        import mayavi.mlab as mlab
        c_red = (224. / 255., 0 / 255., 125 / 255.)
        c_pink = (224. / 255., 75. / 255., 232. / 255.)
        c_blue = (0. / 255., 0. / 255., 255. / 255.)
        scale_factor = 0.007
        source_pcd = self.source_pcd.cpu().numpy()
        tgt_pcd = tgt_pcd.cpu().numpy()

        # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
        if warped_pcd is None:
            mlab.points3d(source_pcd[ :, 0], source_pcd[ :, 1], source_pcd[:,  2],resolution=4, scale_factor=scale_factor , color=c_red)
        else:
            warped_pcd = warped_pcd.detach().cpu().numpy()
            mlab.points3d(warped_pcd[ :, 0], warped_pcd[ :, 1], warped_pcd[:,  2], resolution=4, scale_factor=scale_factor , color=c_pink)
        mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2],resolution=4, scale_factor=scale_factor , color=c_blue)
        mlab.show()