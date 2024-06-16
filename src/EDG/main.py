import mayavi.mlab as mlab
from model.geometry import *
import os
import torch
import argparse
# import cv2
from model.registration import Registration
import  yaml
from easydict import EasyDict as edict



def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)


if __name__ == "__main__":

    extra_info = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    parser.add_argument("--choose_target_id", type = int)
    parser.add_argument("--data_basename")
    parser.add_argument("--src_time", type = int)
    parser.add_argument("--tgt_time", type = int)
    parser.add_argument("--img_depth_type", default = "gt")
    parser.add_argument("--score_method", default = "multi_block")
    parser.add_argument("--target_depth_type", default = "gt")
    parser.add_argument("--score_thres", type = int, default = 13)
    parser.add_argument("--remove_iter", type = int, default = 2)
    parser.add_argument("--mesh_type", default="tetra")
    parser.add_argument("--aspanformer_use_flag", default="True")
    parser.add_argument("--finetune_id", default="99999")
    parser.add_argument("--set-zero", default="False")
    
    
    
    args = parser.parse_args()
    extra_info['data_basename'] = args.data_basename
    extra_info['src_time'] = args.src_time
    extra_info['tgt_time'] = args.tgt_time
    extra_info['choose_target_id'] = args.choose_target_id

    extra_info['score_method'] = args.score_method
    extra_info['img_depth_type'] = args.img_depth_type
    extra_info['target_depth_type'] = args.target_depth_type
    extra_info['aspanformer_flag'] = args.aspanformer_use_flag == "True"
    extra_info['sigma'] = 32
    extra_info['score_thres'] = args.score_thres
    extra_info['remove_iter'] = args.remove_iter
    extra_info['mesh_type'] = args.mesh_type
    extra_info["finetune_id"] = args.finetune_id
    extra_info["set_zero"] = args.set_zero == "True"

    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = edict(config)

    if config.gpu_mode:
        config.device = torch.device("cuda:0")
    else:
        config.device = torch.device('cpu')

    """demo data"""
    intrinsics = np.loadtxt(config.intrinsics)

    """load lepard predicted matches as landmarks"""
    data = np.load(config.correspondence)
    ldmk_src = data['src_pcd'][0][data['match'][:,1]]
    ldmk_tgt = data['tgt_pcd'][0][data['match'][:,2]]
    uv_src = xyz_2_uv(ldmk_src, intrinsics)
    uv_tgt = xyz_2_uv(ldmk_tgt, intrinsics)
    landmarks = ( torch.from_numpy(uv_src).to(config.device),
                  torch.from_numpy(uv_tgt).to(config.device))


    """init model with source frame"""
    model = Registration(config.src_depth, K=intrinsics, config=config, extra_info = extra_info)

    # model.register_a_depth_frame( config.tgt_depth,  landmarks=landmarks)
    model.solve_main()