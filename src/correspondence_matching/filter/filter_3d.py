import json
import os
from copy import deepcopy

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch


def convert_from_uvd_batch(xx, yy, d, fx, fy, cx, cy, direct=False):  # torch
    x_over_z = (xx - cx) / fx
    y_over_z = (yy - cy) / fy
    if direct:
        z = d
    else:
        z = d / torch.sqrt(1.0 + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return torch.stack([x, y, z], -1)


# union_find_algorithm
def find_root(x, parent):
    x_root = x
    while parent[x_root] != -1:
        x_root = parent[x_root]
    return x_root


def union_vertices(x, y, parent, size):
    x_root = find_root(x, parent)
    y_root = find_root(y, parent)
    if x_root != y_root:
        parent[x_root] = y_root
        size[y_root] += size[x_root]


def get_dis(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def get_block_score(kpts, kpts1, mask, src_mask, kpts0_3d, kpts1_3d, depths0, depths1, depths0_grid, depths1_grid):
    dis0_threshold = 0.02
    dis1_threshold = 0.02
    dis0s = []
    dis1s = []

    points = []
    colors = []

    kpts1_left = []

    len_y = 100
    len_x = 100
    width, height = 800, 800
    score_list = []
    kpts = (kpts / 8).astype(int)
    kpts1 = kpts1.astype(int)
    exist = np.zeros((len_y, len_x))
    depth0_grid = np.zeros((len_y, len_x))
    depth1_grid = np.zeros((len_y, len_x))
    kpts0_3d_grid = np.zeros((len_y, len_x, 3))
    kpts1_3d_grid = np.zeros((len_y, len_x, 3))
    parent = [-1] * (len_y * len_x)
    size = [1] * (len_y * len_x)
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    for id, (kpt, kpt1, depth0, depth1) in enumerate(zip(kpts, kpts1, depths0, depths1)):
        x, y = kpt
        x1, y1 = kpt1
        depth0_grid[y, x] = depth0
        depth1_grid[y, x] = depth1
        if x < 0 or x >= len_x or y < 0 or y >= len_y:
            continue
        if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
            continue
        if not mask[y * 8, x * 8]: # correspondence_matching is trying to build the correspondence in 8x lower resolution
            continue
        if not src_mask[y1, x1]:
            continue
        kpts1_left.append([x1, y1])
        exist[y, x] = 1
        kpts0_3d_grid[y, x] = kpts0_3d[id]
        kpts1_3d_grid[y, x] = kpts1_3d[id]
    for kpt in kpts:
        x, y = kpt
        if not exist[y, x]:
            continue
        for i in range(4):
            xx = x + dx[i]
            yy = y + dy[i]
            if xx < 0 or xx >= len_x or yy < 0 or yy >= len_y:
                continue
            if exist[yy, xx] == 0:
                continue
            dis0 = get_dis(kpts0_3d_grid[yy, xx], kpts0_3d_grid[y, x])
            dis1 = get_dis(kpts1_3d_grid[yy, xx], kpts1_3d_grid[y, x])
            dis0s.append(dis0)
            dis1s.append(dis1)
            points.append([x * 8 + dx[i], y * 8 + dy[i]])
            colors.append([0, 0, 0.5])

            if dis0 > dis0_threshold:
                colors[-1][0] += 1
                colors[-1][2] = 0
            if dis1 > dis1_threshold:
                colors[-1][1] += 1
                colors[-1][2] = 0

            if dis0 > dis0_threshold or dis1 > dis1_threshold:
                continue
            union_vertices(y * len_x + x, yy * len_x + xx, parent, size)
    max_size = np.max(size)
    # size_thres = max_size * 0.25
    size_thres = max_size * 0.0
    for kpt in kpts:
        x, y = kpt
        if not exist[y, x]:
            score_list.append(0)
        else:
            root = find_root(y * len_x + x, parent)
            score = size[root]
            if score < size_thres:
                score = 0
            score_list.append(score)
            points.append([x * 8, y * 8])
            grey_rate = min(1.0, score_list[-1] / 100) * 0.8
            colors.append([grey_rate, grey_rate, grey_rate])

    return score_list, points, colors, kpts1_left


def mv(m, v):
    vv = np.array(v)
    vv = np.append(vv, 1).reshape(-1, 1)
    vv = m @ vv
    vv = vv.reshape(-1)
    vv /= vv[-1]
    return vv[:-1]


def mv_batch(m, v):  # m: torch.Tensor[4,4], v: torch.Tensor[N,3]
    v = torch.cat([v, torch.ones_like(v[..., 0:1]).to(m.device)], dim=-1)
    v = torch.transpose(v, -1, -2)
    v = m @ v
    v = torch.transpose(v, -1, -2)
    v = v / v[..., -1:]
    return v[..., :-1]


def cam_to_world(camera_pos, X):  # [4,4], [N, 3]
    X = (camera_pos @ X[:, :, None]).squeeze(-1)
    return X


def max_cat(a, b):
    return np.max([a, b], axis=0)


def get_depth_std(depth):
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    dx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    # calculate the depth of 3*3 surrounding window
    depth_std = np.zeros((9, depth.shape[0], depth.shape[1]))
    depth_std[0] = depth
    depth_std[0, 0, :] = 1e9
    depth_std[0, -1, :] = 1e9
    depth_std[0, :, 0] = 1e9
    depth_std[0, :, -1] = 1e9

    for i in range(1, 9):
        depth_std[i] = np.roll(depth, dx[i], axis=1)
        depth_std[i] = np.roll(depth_std[i], dy[i], axis=0)

    depth_std = np.std(depth_std, axis=0)
    depth_std_shift = np.zeros((4, depth.shape[0], depth.shape[1]))
    depth_std_shift[0, 1:, :] = depth_std[:-1, :]
    depth_std_shift[1, :-1, :] = depth_std[1:, :]
    depth_std_shift[2, :, 1:] = depth_std[:, :-1]
    depth_std_shift[3, :, :-1] = depth_std[:, 1:]
    for i in range(4):
        depth_std = max_cat(depth_std, depth_std_shift[i])

    return depth_std


def threshold_kpts(kpts0, kpts1, scores, threshold=0.3):
    new_kpts0 = []
    new_kpts1 = []
    new_scores = []
    for kpt0, kpt1, score in zip(kpts0, kpts1, scores):
        if score >= threshold:
            if kpt0[1] >= height or kpt0[0] >= width or kpt0[1] < 0 or kpt0[0] < 0:
                continue
            if kpt1[1] >= height or kpt1[0] >= width or kpt1[1] < 0 or kpt1[0] < 0:
                continue
            new_kpts0.append(kpt0)
            new_kpts1.append(kpt1)
            new_scores.append(score)
    return np.array(new_kpts0), np.array(new_kpts1), np.array(new_scores)


def enlarge_pose(camera_pos_):
    camera_pos = np.eye(4)
    camera_pos[:3] = camera_pos_[:]
    camera_pos[:, 1] *= -1
    camera_pos[:, 2] *= -1
    return camera_pos

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--choose_target_id", type=int)
parser.add_argument("--data_basename")
parser.add_argument("--src_time", type=int)
parser.add_argument("--tgt_time", type=int)
parser.add_argument("--img_depth_type", default="gt")
parser.add_argument("--score_method", default="multi_block")
parser.add_argument("--target_depth_type", default="gt")
parser.add_argument("--aspanformer_flag", default="True")
parser.add_argument("--finetune_id", default="99999")


args = parser.parse_args()

choose_target_id = args.choose_target_id
data_basename = args.data_basename
src_time = args.src_time
tgt_time = args.tgt_time
target_depth_type = args.target_depth_type
finetune_id = args.finetune_id

# choose_target_id = 21
# data_basename = "phoenix"
# src_time = 1
# tgt_time = 15 # 2

img_depth_type = args.img_depth_type
print("img_depth_type", img_depth_type)
if args.aspanformer_flag == "True":
    aspanformer_flag = True
else:
    aspanformer_flag = False

data_name = f"{data_basename}_{src_time}"
target_data_name = f"{data_basename}_{tgt_time}"
FF_kpt_name = f"{data_basename}_{src_time}_{tgt_time}_{choose_target_id}"

target_img_dir = f"../data/cmc-render-main/{target_data_name}/{str(choose_target_id).zfill(6)}.png"
source_canonical_img_dir = f"../data/cmc-render-main/{data_name}/{str(choose_target_id).zfill(6)}.png"

src_data_dir = f"../render_temp/{data_name}_original"
src_img_dir = f"../data/cmc-render-main/{data_name}"

if img_depth_type == "gt":
    src_kpt_dir = f"./output_aspanformer/{data_basename}_{src_time}_{tgt_time}_{choose_target_id}"
elif img_depth_type == "nerf":
    src_kpt_dir = f"./output_aspanformer/{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_nerf"
elif img_depth_type == "high":
    src_kpt_dir = f"./output_aspanformer/{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_nerf_high"
elif img_depth_type == "finetune":
    src_kpt_dir = (
        f"./output_aspanformer/{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_nerf_high_{finetune_id}"
    )
elif img_depth_type == "finetune_gt":
    src_kpt_dir = f"./output_aspanformer/{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_{finetune_id}"
elif img_depth_type == "finetune_nerf":
    src_kpt_dir = f"./output_aspanformer/{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_nerf_{finetune_id}"
src_kpt_dir_list = [src_kpt_dir]
if img_depth_type == "hebrid":
    src_kpt_dir_list = [f"./output_aspanformer/{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_nerf_high", f"./output_aspanformer/{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_nerf_high_{finetune_id}"]
camera_dir = f"../../nerfstudio/camera_info/camera_{data_name}.pth"

score_method = args.score_method

task_name = f"{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_{score_method}_img_depth_type_{img_depth_type}_aspanformer_{aspanformer_flag}_target_depth_{target_depth_type}"
if finetune_id != "99999":
    task_name += f"_finetune_id_{finetune_id}"
skip_id_list = [choose_target_id]

output_name = f"./kpt_output/{task_name}/corr_2d_3d_{score_method}.npy"
# if os.path.exists(output_name):
#     print("!!!!!!!! exist, finishing vis_kpt now")
render_dir = f"./kpt_output/{task_name}/kpts_render_{score_method}"
os.makedirs(render_dir, exist_ok=True)

# c2w, _,_,_,_,_,_ = torch.load(camera_dir)

frames = json.load(open(f"{src_img_dir}/transforms_train.json", "r"))["frames"]
transform_matrix, camera_to_world, scale_factor = torch.load(f"./camera_info/{data_name}_train.pt")
c2w_all = torch.Tensor([frame["transform_matrix"] for frame in frames])  # [N, 4, 4]

target_img = imageio.imread(target_img_dir) / 255
target_img_gray = np.mean(target_img, axis=2)
target_img_mask = target_img[..., 3] > 0
source_canonical_img = imageio.imread(source_canonical_img_dir) / 255
source_canonical_img_gray = np.mean(source_canonical_img, axis=2)

if target_depth_type == "gt":
    target_gt_name = (
        f"../data/cmc-render-main/{data_basename}_{tgt_time}/depth/{str(choose_target_id).zfill(6)}_depth.npy"
    )
elif target_depth_type == "zoe":
    target_gt_name = (
        f"../data/cmc-render-main/{data_basename}_{tgt_time}/depth_zoe/{str(choose_target_id).zfill(6)}_depth.npy"
    )
elif "simkinect" in target_depth_type:
    target_gt_name = (
        f"../data/cmc-render-main/{data_basename}_{tgt_time}/depth_{target_depth_type}/{str(choose_target_id).zfill(6)}_depth.npy"
    )
print("target_depth_type", target_depth_type, target_gt_name)
depth_gt = np.load(target_gt_name, allow_pickle=True) / 1000
depth_gt_std = get_depth_std(depth_gt)

fx, fy = 1111.1, 1111.1
cx, cy = 400.0, 400.0
width, height = 800, 800

z = torch.Tensor(depth_gt).reshape(-1).cuda()
x, y = np.meshgrid(np.arange(width), np.arange(height))
x, y = torch.Tensor(x).reshape(-1).cuda(), torch.Tensor(y).reshape(-1).cuda()
camera_pos = torch.Tensor(enlarge_pose(c2w_all[choose_target_id].cpu().numpy())).cuda()

kpts0_3d_can = convert_from_uvd_batch(x, y, z, fx, fy, cx, cy, direct="zoe" not in target_depth_type)
kpts0_3d_grid = mv_batch(camera_pos, kpts0_3d_can).cpu().reshape(height, width, 3).cpu()

corr_2d_3d = {}
pcd_depth = o3d.geometry.PointCloud()

source_canonical_camera_pos_ = c2w_all[choose_target_id].cpu().numpy()
source_canonical_camera_pos = np.eye(4)
source_canonical_camera_pos[:3] = source_canonical_camera_pos_[:]
source_canonical_camera_pos[:, 1] *= -1
source_canonical_camera_pos[:, 2] *= -1

def get_depth(i):
    return (np.load(f"{src_img_dir}/depth/{str(i).zfill(6)}_depth.npy") / 1000)[:, :, None]

def get_kpts_and_score(i, src_kpt_dir, angle=0):

    data = np.load(f"{src_kpt_dir}/match_{i + 1}_{angle}.npy", allow_pickle=True).item()
    kpts0, kpts1, scores = data["corr0"], data["corr1"], data["score"]
    kpts0, kpts1, scores = threshold_kpts(kpts0, kpts1, scores, 0.3)
    
    return kpts0, kpts1, scores


def point_img(img, kpt, outside=True):
    img[kpt[1], kpt[0]] = [1, 0, 0]
    if outside:
        if kpt[1] + 1 < img.shape[0]:
            img[kpt[1] + 1, kpt[0]] = [1, 0, 0]
        if kpt[1] - 1 >= 0:
            img[kpt[1] - 1, kpt[0]] = [1, 0, 0]
        if kpt[0] + 1 < img.shape[1]:
            img[kpt[1], kpt[0] + 1] = [1, 0, 0]
        if kpt[0] - 1 >= 0:
            img[kpt[1], kpt[0] - 1] = [1, 0, 0]


def point_img_batch(img, kpts, color=[1, 0, 0]):
    for kpt in kpts:
        img[kpt[1], kpt[0]] = color


all_kpts_3D = []


for src_kpt_dir in src_kpt_dir_list:
    for i in range(0, 199, 1):
        depth = get_depth(i)
        depth_std = get_depth_std(depth)
        img_mask = imageio.imread(f"{src_img_dir}/{str(i).zfill(6)}.png")[..., 3] > 0
        img_mask[depth[:,:,0] <= 0] = False
        for angle in [-90, -60, -30, 0, 30, 60, 90]:
            img_mask_img = np.ones((height, width, 3)) * img_mask[:, :, None]
            tgt_img_temp = deepcopy(target_img)[:, :, :3]
            kpts0, kpts1, scores = get_kpts_and_score(i, src_kpt_dir, angle)
            if kpts1.shape[0] == 0:
                continue
            original_scores = scores
            print("id", i, angle, len(scores))

            camera_pos = enlarge_pose(c2w_all[i].cpu().numpy())

            kpts0_3d = torch.stack([kpts0_3d_grid[int(y0), int(x0)] for (x0, y0) in kpts0], 0).cuda()
            depths1 = torch.Tensor([depth[int(y1), int(x1)].item() for (x1, y1) in kpts1]).cuda()
            depths0 = torch.Tensor([depth_gt[int(y1), int(x1)].item() for (x1, y1) in kpts0]).cuda()
            kpts1_torch = torch.Tensor(kpts1).cuda().long().float()
            kpts1_3d_can = convert_from_uvd_batch(
                kpts1_torch[:, 0], kpts1_torch[:, 1], depths1, fx, fy, cx, cy, direct=True
            )
            kpts1_3d = mv_batch(torch.Tensor(camera_pos).cuda(), kpts1_3d_can).cpu()

            scores, points_render, colors_render, kpts1_left = get_block_score(
                kpts0,
                kpts1,
                target_img_mask,
                img_mask,
                kpts0_3d.cpu().numpy(),
                kpts1_3d.cpu().numpy(),
                depths0.cpu().numpy(),
                depths1.cpu().numpy(),
                depth_gt,
                depth,
            )

            for kpt0, kpt1, score, original_score, kpt1_3d in zip(kpts0, kpts1, scores, original_scores, kpts1_3d):
                x1, y1 = kpt1
                x0, y0 = kpt0
                x1, y1 = int(x1), int(y1)
                x0, y0 = int(x0), int(y0)
                if y1 >= height or x1 >= width or y1 < 0 or x1 < 0:
                    continue
                if y0 >= height or x0 >= width or y0 < 0 or x0 < 0:
                    continue
                if not img_mask[y1, x1] or not target_img_mask[y0, x0]:
                    continue
                if depth_std[y1, x1] > 0.1:
                    continue
                if depth_gt[y0, x0] == 0:
                    continue

                if target_img_gray[y0, x0] > 0:
                    # point_img(tgt_img_temp, (x0, y0))
                    all_kpts_3D.append(kpt1_3d)
                    if (y0, x0) not in corr_2d_3d:
                        corr_2d_3d[(y0, x0)] = [
                            (kpt1_3d, score, src_kpt_dir, (i, angle, original_score, depth_std[y1, x1], (y0, x0)), (y1, x1))
                        ]
                    else:
                        corr_2d_3d[(y0, x0)].append(
                            (kpt1_3d, score, src_kpt_dir, (i, angle, original_score, depth_std[y1, x1], (y0, x0)), (y1, x1))
                            )

for (y0, x0), kpts_3d in corr_2d_3d.items():
    corr_2d_3d[(y0, x0)] = sorted(kpts_3d, key=lambda x: x[1], reverse=True)

np.save(output_name, corr_2d_3d)
