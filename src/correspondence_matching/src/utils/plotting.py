import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import imageio
from copy import deepcopy

def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth' or dataset_name=='gl3d':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def draw_match2(img1, img2, corr1, corr2,inlier=[True],color=None,radius1=1,radius2=1,resize=None, mconf = None):
    if resize is not None:
        scale1,scale2=[img1.shape[1]/resize[0],img1.shape[0]/resize[1]],[img2.shape[1]/resize[0],img2.shape[0]/resize[1]]
        img1,img2=cv2.resize(img1, resize, interpolation=cv2.INTER_AREA),cv2.resize(img2, resize, interpolation=cv2.INTER_AREA) 
        corr1,corr2=corr1/np.asarray(scale1)[np.newaxis],corr2/np.asarray(scale2)[np.newaxis]
    corr1_key = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], radius1) for i in range(corr1.shape[0])]
    corr2_key = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], radius2) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]
    if color is None:
        color = [(0, 255, 0) if cur_inlier else (0,0,255) for cur_inlier in inlier]
    if mconf is not None:
        color = []
        # print('mconf', mconf)
        for i in range(len(mconf)):
            rate = (mconf[i] - 0.2) / 0.8
            color.append(np.array([0, 255, 0]) * rate + np.array([0, 0, 255]) * (1 - rate)) # BGR
    # print('color', color)
    if color.shape[0] > 0 and color.max() <= 1.1:
        color = (color * 255).astype(np.uint8)
    if len(color)==1 and 0:
        display = cv2.drawMatches(img1, corr1_key, img2, corr2_key, draw_matches, None,
                              matchColor=color[0],
                              singlePointColor=color[0],
                              flags=4
                              )
    else:
        color_rdm = False
        if len(color)==1:
            color_rdm = True
            color = np.random.rand(len(corr1),3) * 255
        height,width=max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1]
        display=np.zeros([height,width,3],np.uint8)
        display[:img1.shape[0],:img1.shape[1]]=img1
        display[:img2.shape[0],img1.shape[1]:]=img2
        n_line = 60
        for i in range(len(corr1)):
            left_x,left_y,right_x,right_y=int(corr1[i][0]),int(corr1[i][1]),int(corr2[i][0]+img1.shape[1]),int(corr2[i][1])
            cur_color=(int(color[i][0]),int(color[i][1]),int(color[i][2]))
            cv2.circle(display, (left_x, left_y), 2, cur_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(display, (right_x, right_y), 2, cur_color, -1, lineType=cv2.LINE_AA)
            cur_color=(int(color[i][0]),int(color[i][1]),int(color[i][2]))
            if np.random.rand() < n_line / len(corr1) and not color_rdm:
                cv2.line(display, (left_x,left_y), (right_x,right_y),cur_color,1,lineType=cv2.LINE_AA)
    return display

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None, cv2 = False):
    
    # print(img0, img0.shape, img1.shape, mkpts0.shape, mkpts1.shape, kpts0.shape, color.shape)
    if cv2:
        img = draw_match2(img0[:,:,None], img1[:,:,None], mkpts0, mkpts1, color=color[:,:3])
        imageio.imwrite(path, img)
        return
    # import torch
    # torch.save([img0, img1, mkpts0, mkpts1, kpts0, kpts1, text, dpi, path, color], 'debug.pt')
    # print('save')
    # input()

    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic', extra_info = None, path = None, cv2 = False):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]
    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    if extra_info is not None:
        extra_info['n_match'] = len(kpts0)
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text, path = path, cv2 = cv2)
    return figure

def _make_evaluation_figure_offset(data, b_id, alpha='dynamic',side=''):
    layer_num=data['predict_flow'][0].shape[0]

    b_mask = data['offset_bids'+side] == b_id
    conf_thr = 2e-3 #hardcode for scannet(coarse level)
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    
    figure_list=[]
    #draw offset matches in different layers
    for layer_index in range(layer_num):
        l_mask=data['offset_lids'+side]==layer_index
        mask=l_mask&b_mask
        kpts0 = data['offset_kpts0_f'+side][mask].cpu().numpy()
        kpts1 = data['offset_kpts1_f'+side][mask].cpu().numpy()
        
        epi_errs = data['epi_errs_offset'+side][mask].cpu().numpy()
        correct_mask = epi_errs < conf_thr
        
        precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
        n_correct = np.sum(correct_mask)
        n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
        recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
        # recall might be larger than 1, since the calculation of conf_matrix_gt
        # uses groundtruth depths and camera poses, but epipolar distance is used here.

        # matching info
        if alpha == 'dynamic':
            alpha = dynamic_alpha(len(correct_mask))
        color = error_colormap(epi_errs, conf_thr, alpha=alpha)
        
        text = [
            f'#Matches {len(kpts0)}',
            f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
            f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
        ]
        
        # make the figure
        #import pdb;pdb.set_trace()
        figure = make_matching_figure(deepcopy(img0), deepcopy(img1) , kpts0, kpts1,
                                    color, text=text)
        figure_list.append(figure)
    return figure

def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode='evaluation', extra_result = None, path = None, cv2 = False):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    if extra_result is not None:
        extra_result['n_match_gt'] = data['spv_b_ids'].shape[0]
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA, 
                extra_info=extra_result,
                path = path,
                cv2 = cv2)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
    figures[mode].append(fig)
    return figures

def make_matching_figures_offset(data, config, mode='evaluation',side=''):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure_offset(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA,side=side)
        elif mode == 'confidence':
            fig = _make_evaluation_figure_offset(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures

def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
