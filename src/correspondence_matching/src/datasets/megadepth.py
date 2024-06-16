import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
import glob
from copy import deepcopy
import json

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth, read_megadepth_gray2, read_megadepth_depth2


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def random_func2(x, seed):

    result = 101 * x + 10101 * (x + seed) + 1010101 * (x * x + seed * seed) + 101010101 * (x * x * x + seed * seed * seed) + x + seed
    seed = (seed * 101 + 1000000009 + seed * seed * 131) % 1000000007
    return result, seed

def random_func(x, seed):

    return 10101 * x + 101010101 * x * seed + 1000000007 * (x + seed) * (x + seed) + 1000000009 * x * x * x

def generate_map(length, seed):

    seed_state = deepcopy(seed)
    id_map = [i for i in range(length)]
    for i in range(1, length):
        pose, seed = random_func2(i, seed)
        pose = pose % i
        id_map[i], id_map[pose] = id_map[pose], id_map[i]
    return id_map

class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        # print(npz_path)
        # print('len pair info', len(self.pair_infos))
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]
        
        # print('len pair info', len(self.pair_infos))
        # if len(self.pair_infos) >= 9575 * 4 - 4 and len(self.pair_infos) <= 9575 * 4 + 4:
        #     input()
        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        # print('info', self.img_resize, self.df, self.img_padding) # 832, 8, True
        self.depth_max_size = 3000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0, original_size0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1, original_size1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0, rate0 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size, original_size=original_size0)
            depth1, rate1 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size, original_size=original_size1)
        else:
            depth0 = depth1 = torch.tensor([])
        print('####### depth in megadepth', depth0.shape, depth1.shape, image0.shape, image1.shape)
        # import os
        # if os.path.basename(self.scene_info['depth_paths'][idx1]) == "3553841868_b6ee93bf43_o.h5":
        #     print('debug data')
        #     print(self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1])
        #     print(self.scene_info['depth_paths'][idx0], self.scene_info['depth_paths'][idx1])
        #     print(scale0, scale1, image0.shape, image1.shape, depth0.shape, depth1.shape)
        #     input()

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
        K_0[0:2] *= rate0[:,None]
        K_1[0:2] *= rate1[:,None]
        
        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            'pair_names_depth': (self.scene_info['depth_paths'][idx0], self.scene_info['depth_paths'][idx1]),
        }

        # if scale0 > 10:
        #     print('aha')
        #     print(scale0, scale1)
        #     input()

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data


class MegaDepthDataset2(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 rank=0,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        # print(npz_path)
        # print('len pair info', len(self.pair_infos))
        # self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        if "val" in mode:
            self.data_dir = "data_blender_aspan"
        elif "train" in mode:
            self.data_dir = "data_blender_aspan2"
        else:
            raise NotImplementedError
        print('enter glob')
        # self.pair_infos = glob.glob(f"./{self.data_dir}/*.json")
        # np.save(f"./{self.data_dir}_pair_infos.npy", self.pair_infos)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if "train" in mode:
            self.pair_infos = np.load(f"./{self.data_dir}_pair_infos.npy", allow_pickle=True)
            print('train size', len(self.pair_infos))
        if "val" in mode:
            self.pair_infos = glob.glob(f"./{self.data_dir}/*.json")
            
            from copy import deepcopy

            pair_unsorted = deepcopy(self.pair_infos)
            self.pair_infos.sort()
            self.pair_infos = [self.pair_infos[(1000000007 * x + 10000000009 * x * x)% 1200] for x in range(20)] + [pair_unsorted[(1000000007 * x + 10000000009 * x * x)% (len(self.pair_infos) - 0) + 0] for x in range(30)]
            print('val', self.pair_infos)

        self.vis = [0 for i in range(len(self.pair_infos))]
        self.vis_sum = 0
        print('finish glob', len(self.pair_infos))
        self.pair_infos.sort()
        # if "val" in mode:
        #     self.pair_infos = [self.pair_infos[(1000000007 * x + 10000000009 * x * x)% 1200] for x in range(20)]
        # else:
        #     self.pair_infos = self.pair_infos[1200:]
        

        
        print('create dataset', mode, len(self.pair_infos), rank)
        # set_seed(rank)
        self.seed = rank
        if 'train' in mode:
            self.id_map = generate_map(len(self.pair_infos), self.seed)
        else:
            self.id_map = [i for i in range(len(self.pair_infos))]
        
        # print('len pair info', len(self.pair_infos))
        # if len(self.pair_infos) >= 9575 * 4 - 4 and len(self.pair_infos) <= 9575 * 4 + 4:
        #     input()
        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        # print('info', self.img_resize, self.df, self.img_padding) # 832, 8, True
        self.depth_max_size = 850 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, ori_idx):
        
        while 1:
            idx = self.id_map[ori_idx]
            print('ori_idx', self.seed, ori_idx, idx, np.sum(self.vis), np.sum(np.array(self.vis) >= 0.5))
            self.vis[idx] += 1
            if self.vis[idx] == 1:
                self.vis_sum += 1
            # idx = (idx + random_func(idx, self.seed)) % len(self.pair_infos)
            data_info = json.load(open(self.pair_infos[idx]))
            pair_name = data_info['pair_name']
            extra_data = torch.load(rf"./{self.data_dir}/{pair_name}.pth")
            w_xyz_0_to_1_cam, xyz_mask0, w_xyz_1_to_0_cam, xyz_mask1 = extra_data[0], extra_data[1], extra_data[2], extra_data[3]
            # if w_xyz_0_to_1_cam has None, continue
            
            ori_idx = (ori_idx + 3)  % len(self.id_map)
            if torch.isnan(w_xyz_0_to_1_cam).any():
                continue
            if torch.isnan(w_xyz_1_to_0_cam).any():
                continue
            break

        # read grayscale image and mask. (1, h, w) and (h, w)
        # img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        # img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        # img_name0 = osp.join(self.root_dir, data_info['img_src_name'])
        # img_name1 = osp.join(self.root_dir, data_info['img_tgt_name'])
        img_name0 = data_info['img_src_name']
        img_name1 = data_info['img_tgt_name']
        
        print('vis_sum', self.mode, self.seed, ori_idx, idx, self.vis_sum, self.vis[idx], 'img_name', img_name0, img_name1)
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0, original_size0 = read_megadepth_gray2(
            img_name0, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1, original_size1 = read_megadepth_gray2(
            img_name1, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            # depth0, rate0 = read_megadepth_depth(
            #     osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size, original_size=original_size0)
            # depth1, rate1 = read_megadepth_depth(
            #     osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size, original_size=original_size1)
            # depth0, rate0 = read_megadepth_depth2(
            #     osp.join(self.root_dir, data_info['depth_src_name']), pad_to=self.depth_max_size, original_size=original_size0)
            # depth1, rate1 = read_megadepth_depth2(
            #     osp.join(self.root_dir, data_info['depth_tgt_name']), pad_to=self.depth_max_size, original_size=original_size1)
            depth0, rate0 = read_megadepth_depth2(
                data_info['depth_src_name'], pad_to=self.depth_max_size, original_size=original_size0)
            depth1, rate1 = read_megadepth_depth2(
                data_info['depth_tgt_name'], pad_to=self.depth_max_size, original_size=original_size1)
        else:
            depth0 = depth1 = torch.tensor([])
        # print('####### depth in megadepth', depth0.shape, depth1.shape, image0.shape, image1.shape)
        # import os
        # if os.path.basename(self.scene_info['depth_paths'][idx1]) == "3553841868_b6ee93bf43_o.h5":
        #     print('debug data')
        #     print(self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1])
        #     print(self.scene_info['depth_paths'][idx0], self.scene_info['depth_paths'][idx1])
        #     print(scale0, scale1, image0.shape, image1.shape, depth0.shape, depth1.shape)
        #     input()

        # read intrinsics of original size

        fx, fy = 1111.1, 1111.1
        cx, cy = 400., 400.
        width, height = 800, 800

        K = torch.zeros((3,3), dtype=torch.float)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
        K[2,2] = 1

        # K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        # K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
        K_0 = K.clone()
        K_1 = K.clone()
        K_0[0:2] *= rate0[:,None]
        K_1[0:2] *= rate1[:,None]
        
        # read and compute relative poses
        # T0 = self.scene_info['poses'][idx0]
        # T1 = self.scene_info['poses'][idx1]
        T0 = data_info['T0']
        T1 = data_info['T1']
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'w_xyz_0_to_1_cam': w_xyz_0_to_1_cam,
            'xyz_mask0': xyz_mask0,
            'w_xyz_1_to_0_cam': w_xyz_1_to_0_cam,
            'xyz_mask1': xyz_mask1,
            'pair_names': pair_name,
            'idx': idx,
            'idx_val': idx,
            # 'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            # 'pair_names_depth': (self.scene_info['depth_paths'][idx0], self.scene_info['depth_paths'][idx1]),
        }

        # if scale0 > 10:
        #     print('aha')
        #     print(scale0, scale1)
        #     input()

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
