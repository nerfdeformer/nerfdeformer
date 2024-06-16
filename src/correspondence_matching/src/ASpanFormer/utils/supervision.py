from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid
import numpy as np

from .geometry import warp_kpts, warp_kpts2

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt

@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device

    w_xyz_0_to_1_cam = data['w_xyz_0_to_1_cam'].to(device)
    xyz_mask0 = data['xyz_mask0'].to(device)
    w_xyz_1_to_0_cam = data['w_xyz_1_to_0_cam'].to(device)
    xyz_mask1 = data['xyz_mask1'].to(device)

    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['ASPAN']['RESOLUTION'][0] # 8
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1]) # 104 or 832
    # torch.save(data['image0'], 'image0.pt')
    # torch.save(data['image1'], 'image1.pt')
    # torch.save(data['depth0'], 'depth0.pt')
    # torch.save(data['depth1'], 'depth1.pt')
    # print('scale0 and 1', data['scale0'], data['scale1'], data['pair_names'], data['pair_names_depth'])
    # exit(0)
    # print(N, H0, W0, H1, W1, scale, scale0, scale1, h0, w0, h1, w1)

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt0_m = scale * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c
    # print('ahahaha', grid_pt0_i[0,1], scale0)
    # input()
    # print('previous grid', grid_pt0_i.shape, grid_pt0_i.max(), grid_pt0_i.min(), grid_pt1_i.max(), grid_pt1_i.min())
    # print(grid_pt0_c.max(), grid_pt0_c.min(), grid_pt1_c.max(), grid_pt1_c.min())
    # print(scale, scale0, scale1, [H0, W0, H1, W1], h0, w0, h1, w1)
    # print('all shape', data['scale0'], data['scale1'], data['image0'].shape, data['depth0'].shape, data['image1'].shape, data['depth1'].shape) # image: 832, depth: 2000
    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    # _, w_pt0_i, w_kpts0_cam = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'], True)
    # _, w_pt1_i, w_kpts1_cam = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'], False)
    _, w_pt0_i, w_kpts0_cam = warp_kpts2(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'], w_xyz_0_to_1_cam, xyz_mask0, True)
    _, w_pt1_i, w_kpts1_cam = warp_kpts2(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'], w_xyz_1_to_0_cam, xyz_mask1, False)
                
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # print('w_kpts0_cam', w_kpts0_cam.shape, w_kpts1_cam.shape) # [1,3,N]
    # import open3d as o3d
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # pcd = o3d.geometry.PointCloud()
    # print('w_kpts0_cam', w_kpts0_cam.shape, w_kpts0_cam.max(), w_kpts0_cam.min())
    # pcd.points = o3d.utility.Vector3dVector(w_kpts0_cam[0].cpu().numpy().T)
    # pcd.paint_uniform_color([1, 0, 0])
    # vis.add_geometry(pcd)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(w_kpts1_cam[0].cpu().numpy().T)
    # pcd.paint_uniform_color([0, 1, 0])
    # vis.add_geometry(pcd)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(w_kpts1_cam[0, :, 23 * 104 + 7:23 * 104 + 7 + 1].cpu().numpy().T)
    # pcd.paint_uniform_color([0, 0, 1])
    # vis.add_geometry(pcd)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(w_kpts0_cam[0, :, 20 * 104 + 6:20 * 104 + 6 + 1].cpu().numpy().T)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # vis.add_geometry(pcd)
    # print(w_kpts1_cam[0, :, 23 * 104 + 7:23 * 104 + 7 + 1], w_kpts0_cam[0, :, 20 * 104 + 6:20 * 104 + 6 + 1])
    # vis.run()
    # input()


    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0
    # print('########## aha', grid_pt0_i.shape)
    # for id, grid_pt0_m_single in enumerate(grid_pt0_m[0]):
    #     # if grid_pt0_m_single[0] + grid_pt0_m_single[1] != 0:
    #     #     print('grid', grid_pt0_m_single)
    #     if int(grid_pt0_m_single[0]) == 48 and int(grid_pt0_m_single[1]) == 160:
    #         print(nearest_index1.shape)
    #         print('debug0', nearest_index1[0, id])
    #         yy = nearest_index1[0, id]//(832 // 8) * 8
    #         xx = nearest_index1[0, id]%(832 // 8) * 8
    #         print('debug0', yy, xx, xx + 832)
    #         input()
            # 20 * 104 + 6

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    # print('output debug')
    # print(data['image0'].shape, data['image1'].shape, b_ids.max(), b_ids.min()) # [1,1,832,832] for both
    # h, w = data['image0'].shape[2:]
    # print(h,w,h0,w0,h1,w1)
    # img_cat = torch.cat([data['image0'], data['image1']], dim=3)[0].repeat(3, 1, 1)
    # for i_id, j_id in zip(i_ids, j_ids):
    #     # print(i_id, j_id)
    #     color = torch.rand(3)
    #     img_cat[:, i_id//(w // 8) * 8, i_id%(w // 8) * 8] = color
    #     img_cat[:, j_id//(w // 8) * 8, j_id%(w // 8) * 8 + w] = color
    # img_cat = img_cat.permute(1, 2, 0)
    # print('img_cat', img_cat.shape)
    # torch.save(img_cat, 'debug.pt')
    # print(img_cat.mean(), img_cat.max(), img_cat.min())
    # import imageio
    # img_cat = (img_cat.cpu().numpy() * 255).astype(np.uint8)
    # imageio.imwrite("debug.png", img_cat)
    # input()
    #   imageio.imwrite(data['image0'])

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })

@torch.no_grad()
def spvs_coarse_old(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['ASPAN']['RESOLUTION'][0] # 8
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1]) # 104 or 832
    # torch.save(data['image0'], 'image0.pt')
    # torch.save(data['image1'], 'image1.pt')
    # torch.save(data['depth0'], 'depth0.pt')
    # torch.save(data['depth1'], 'depth1.pt')
    # print('scale0 and 1', data['scale0'], data['scale1'], data['pair_names'], data['pair_names_depth'])
    # exit(0)

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt0_m = scale * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c
    # print('ahahaha', grid_pt0_i[0,1], scale0)
    # input()
    # print('previous grid', grid_pt0_i.shape, grid_pt0_i.max(), grid_pt0_i.min(), grid_pt1_i.max(), grid_pt1_i.min())
    # print(grid_pt0_c.max(), grid_pt0_c.min(), grid_pt1_c.max(), grid_pt1_c.min())
    # print(scale, scale0, scale1, [H0, W0, H1, W1], h0, w0, h1, w1)
    # print('all shape', data['scale0'], data['scale1'], data['image0'].shape, data['depth0'].shape, data['image1'].shape, data['depth1'].shape) # image: 832, depth: 2000
    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i, w_kpts0_cam = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'], True)
    _, w_pt1_i, w_kpts1_cam = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'], False)
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # print('w_kpts0_cam', w_kpts0_cam.shape, w_kpts1_cam.shape) # [1,3,N]
    # import open3d as o3d
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # pcd = o3d.geometry.PointCloud()
    # print('w_kpts0_cam', w_kpts0_cam.shape, w_kpts0_cam.max(), w_kpts0_cam.min())
    # pcd.points = o3d.utility.Vector3dVector(w_kpts0_cam[0].cpu().numpy().T)
    # pcd.paint_uniform_color([1, 0, 0])
    # vis.add_geometry(pcd)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(w_kpts1_cam[0].cpu().numpy().T)
    # pcd.paint_uniform_color([0, 1, 0])
    # vis.add_geometry(pcd)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(w_kpts1_cam[0, :, 23 * 104 + 7:23 * 104 + 7 + 1].cpu().numpy().T)
    # pcd.paint_uniform_color([0, 0, 1])
    # vis.add_geometry(pcd)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(w_kpts0_cam[0, :, 20 * 104 + 6:20 * 104 + 6 + 1].cpu().numpy().T)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # vis.add_geometry(pcd)
    # print(w_kpts1_cam[0, :, 23 * 104 + 7:23 * 104 + 7 + 1], w_kpts0_cam[0, :, 20 * 104 + 6:20 * 104 + 6 + 1])
    # vis.run()
    # input()


    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0
    # print('########## aha', grid_pt0_i.shape)
    # for id, grid_pt0_m_single in enumerate(grid_pt0_m[0]):
    #     # if grid_pt0_m_single[0] + grid_pt0_m_single[1] != 0:
    #     #     print('grid', grid_pt0_m_single)
    #     if int(grid_pt0_m_single[0]) == 48 and int(grid_pt0_m_single[1]) == 160:
    #         print(nearest_index1.shape)
    #         print('debug0', nearest_index1[0, id])
    #         yy = nearest_index1[0, id]//(832 // 8) * 8
    #         xx = nearest_index1[0, id]%(832 // 8) * 8
    #         print('debug0', yy, xx, xx + 832)
    #         input()
            # 20 * 104 + 6

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    # print('output debug')
    # print(data['image0'].shape, data['image1'].shape, b_ids.max(), b_ids.min()) # [1,1,832,832] for both
    # h, w = data['image0'].shape[2:]
    # print(h,w,h0,w0,h1,w1)
    # img_cat = torch.cat([data['image0'], data['image1']], dim=3)[0].repeat(3, 1, 1)
    # for i_id, j_id in zip(i_ids, j_ids):
    #     # print(i_id, j_id)
    #     color = torch.rand(3)
    #     img_cat[:, i_id//(w // 8) * 8, i_id%(w // 8) * 8] = color
    #     img_cat[:, j_id//(w // 8) * 8, j_id%(w // 8) * 8 + w] = color
    # img_cat = img_cat.permute(1, 2, 0)
    # print('img_cat', img_cat.shape)
    # torch.save(img_cat, 'debug.pt')
    # print(img_cat.mean(), img_cat.max(), img_cat.min())
    # import imageio
    # img_cat = (img_cat.cpu().numpy() * 255).astype(np.uint8)
    # imageio.imwrite("debug.png", img_cat)
    # input()
    #   imageio.imwrite(data['image0'])

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['ASPAN']['RESOLUTION'][1]
    radius = config['ASPAN']['FINE_WINDOW_SIZE'] // 2

    # print('debug', radius, scale, config['ASPAN']['FINE_WINDOW_SIZE'], config['ASPAN']['RESOLUTION'])
    # print(config)

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError
