import os
import sys
import imageio
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.ASpanFormer.aspanformer import ASpanFormer
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
import demo_utils
from copy import deepcopy

import cv2
import torch
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='../configs/aspan/outdoor/aspan_test.py',
  help='path for config file.')
parser.add_argument('--img0_path', type=str, default='../assets/phototourism_sample_images/piazza_san_marco_06795901_3725050516.jpg',
  help='path for image0.')
parser.add_argument('--img1_path', type=str, default='../assets/phototourism_sample_images/piazza_san_marco_15148634_5228701572.jpg',
  help='path for image1.')
parser.add_argument('--weights_path', type=str, default='../weights/outdoor.ckpt',
  help='path for model weights.')
parser.add_argument('--long_dim0', type=int, default=1024,
  help='resize for longest dim of image0.')
parser.add_argument('--long_dim1', type=int, default=1024,
  help='resize for longest dim of image1.')
parser.add_argument('--out_path', type=str, default='./',
  help='path for output.')
parser.add_argument('--rgbd', action='store_true')


args = parser.parse_args()
from torchvision import transforms
rotate = transforms.functional.rotate

def rotate_img(img,angle):
  img=rotate(img,angle)
  return img

if __name__=='__main__':
    
    config = get_cfg_defaults()
    config.merge_from_file(args.config_path)
    _config = lower_config(config)
    matcher = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load(args.weights_path, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict,strict=False)
    matcher.cuda(),matcher.eval()
    
    import glob
    img0_name = args.img0_path
    img_list = [x for x in glob.glob(args.img1_path + '/*.png') if not 'depth' in x]
    img_list.sort()
    
    os.makedirs(args.out_path,exist_ok=True)
    angle_list = [0, 60, -90, 90, -30, 30, -60]

    for rot_angle in angle_list:
      rot_rad = - rot_angle / 180. * np.pi
      for id, img1_name in enumerate(img_list):
        print(rot_angle,img0_name,img1_name)
        file_name = args.out_path + f'/match_{id + 1}_rc_{rot_angle}.png'
        
        img0, img1 = cv2.imread(img0_name), cv2.imread(img1_name) # shape: (H, W, C)
        if img1.shape[0] != args.long_dim1:
          img1 = demo_utils.resize(img1,args.long_dim1)

        # filtering here: seems ASpanFormer has better results on white background than black.
        black = img0.min(2) <= 3
        img0[:,:,0][black] = 255
        img0[:,:,1][black] = 255
        img0[:,:,2][black] = 255

        black = img1.min(2) <= 3
        img1[:,:,0][black] = 255
        img1[:,:,1][black] = 255
        img1[:,:,2][black] = 255
        
      
        img1_original = deepcopy(img1)
        img0_g,img1_g=cv2.imread(img0_name,0),cv2.imread(img1_name,0)
        
        img1 = rotate(torch.from_numpy(img1).permute(2, 0, 1), rot_angle, fill = 255).permute(1, 2, 0).numpy().astype(np.uint8)
        img1_g = rotate(torch.from_numpy(img1_g)[None], rot_angle, fill = 255)[0].numpy().astype(np.uint8)

        img0,img1=demo_utils.resize(img0,args.long_dim0),demo_utils.resize(img1,args.long_dim1)
        img0_g,img1_g=demo_utils.resize(img0_g,args.long_dim0),demo_utils.resize(img1_g,args.long_dim1)
        data={'image0':torch.from_numpy(img0_g/255.)[None,None].cuda().float(),
              'image1':torch.from_numpy(img1_g/255.)[None,None].cuda().float()} 
        
        with torch.no_grad():
          matcher(data,online_resize=True)
          corr0,corr1=data['mkpts0_f'].cpu().numpy(),data['mkpts1_f'].cpu().numpy()

        image_size = np.array([img0.shape[0], img0.shape[1]])
        corr1 -= image_size / 2
        corr1[:,1], corr1[:,0] = np.cos(rot_rad) * deepcopy(corr1[:,1]) - np.sin(rot_rad) * deepcopy(corr1[:,0]), np.sin(rot_rad) * deepcopy(corr1[:,1]) + np.cos(rot_rad) * deepcopy(corr1[:,0])
        corr1 += image_size / 2

        #visualize match
        display=demo_utils.draw_match(img0,img1_original,corr0,corr1, mconf = data['mconf'].cpu().numpy())
        display_rc=demo_utils.draw_match(img0,img1_original,corr0,corr1)
        data = {
          'corr0':corr0,
          'corr1':corr1,
          'score':data['mconf'].cpu().numpy(),
        }
        np.save(args.out_path + f'/match_{id + 1}_{rot_angle}.npy', data)
        cv2.imwrite(args.out_path + f'/match_{id + 1}_{rot_angle}_line.png',display)
        cv2.imwrite(args.out_path + f'/match_{id + 1}_{rot_angle}_dot.png',display_rc)

        print('# of corr', len(corr1))