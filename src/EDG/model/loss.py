from typing import Union
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
import matplotlib.pyplot as plt
from time import time

def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals

def compute_truncated_chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        trunc=0.2,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)


    # truncation
    x_mask[cham_x >= trunc] = True
    y_mask[cham_y >= trunc] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0


    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    # cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist

def control_cost(R, t):

    R_mean = R.mean(0)
    t_mean = t.mean(0)
    loss = (R - R_mean).pow(2).mean() + (t - t_mean).pow(2).mean()
    return loss
def edge_cost(g, t, e, w_):

    w = w_ > 0
    g_i = g[:, None]
    g_j = g[e]
    t_i = t[:, None]
    t_j = t[e]
    
    # print(g_i.shape, g_j.shape, t_i.shape, t_j.shape, 'debug')
    # input()
    diff_g = g_i - g_j
    diff_t = g_i + t_i - g_j - t_j
    length_g = (diff_g * diff_g).sum(-1).sqrt()
    length_t = (diff_t * diff_t).sum(-1).sqrt()
    loss = (length_g - length_t).pow(2)
    loss = (loss * w).mean()

    return loss


def argmin_dis_batch(p, q, K=1):
    from pytorch3d import ops
    result = ops.knn_points(p[None], q[None], K=K)
    return result[0][0, :, :], result[1][0, :, :]

# def collision_loss(p, min_dis):
#     valid = ~torch.isnan(p[:,0])
#     p = p[valid]
#     torch.save(p, 'p.pt')
#     # torch.save(min_dis, 'min_dis.pt')
#     # input()
#     t = time()
#     dis = argmin_dis_batch(p, p, K = 2)[0][...,1]
#     print('t c loss', time() - t)
#     print(dis.shape, p.shape, min_dis.shape, dis.mean(), 'dis')
#     dis[dis > min_dis[valid] / 2] = 0.
#     print(dis.mean())
#     return - dis.mean()

def min_minibatch_mask(p, q, mask):
    with torch.no_grad():
        dis_ = (p[:,None] - q).square().sum(-1).sqrt()
        dis = torch.ones_like(dis_) * 1e10
        dis[mask] = dis_[mask]
        dis, min_id = dis.min(1)
    dis = (p[:] - q[min_id]).square().sum(-1).sqrt()
    return dis

def min_batch_mask(p, q, mask):

    batch_size = int(1e8 / q.shape[0])
    dis_list = []
    for i in range(0, p.shape[0], batch_size):
        p_batch = p[i:i+batch_size]
        mask_batch = mask[i:i+batch_size]
        dis = min_minibatch_mask(p_batch, q, mask_batch)    
        dis_list.append(dis)
    dis = torch.cat(dis_list)
    return dis

def collision_loss(nodes_, vertices, min_dis_mask, threshold = 1e-3): # [N, 3], [M, 3]
    print('begin')
    valid = ~torch.isnan(vertices[:,0])
    nodes = nodes_[::2]
    min_dis_mask = min_dis_mask[::2, valid]
    dis_ = min_batch_mask(nodes, vertices[valid], min_dis_mask)
    print(dis_.min(), dis_.max(), dis_.mean(), dis_.shape, 'dis')
    dis = torch.zeros_like(dis_)
    dis[dis_ <= threshold] = dis_[dis_ <= threshold]
    print('ends')
    return - dis.mean()

def arap_cost (R, t, g, e, w, lietorch=True):
    '''
    :param R:
    :param t:
    :param g:
    :param e:
    :param w:
    :return:
    '''

    R_i = R [:, None]
    g_i = g [:, None]
    t_i = t [:, None]

    g_j = g [e]
    t_j = t [e]
    # print('arap', R_i.shape, g_j.shape, g_i.shape, t_i.shape, g_j.shape, t_j.shape, w.shape)
    # # arap torch.Size([7710, 1, 3]) torch.Size([7710, 8, 3]) torch.Size([7710, 1, 3]) torch.Size([7710, 1, 3]) torch.Size([7710, 8, 3]) torch.Size([7710, 8, 3]) torch.Size([7710, 8])
    # print(w.max(), w.min(), w.mean())
    # print('diff', t_i.mean(), t_i.std(), (R_i - torch.eye(3).cuda()).abs().max())

    if lietorch :
        e_ij = ((R_i * (g_j - g_i) + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)
    else:
        e_ij = (((R_i.double() @ (g_j - g_i)[...,None]).squeeze() + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)
        # print(R_i.type(), R_i.shape,  g_j.type(), g_j.shape, g_i.type(), g_i.shape)
        # A = (R_i @ (g_j - g_i)[...,None]).squeeze()
        # B = g_i + t_i
        # C = g_j + t_j
        # print('type', A.type(), B.type(), C.type())
        # e_ij = ((A + B  - C )**2).sum(dim=-1)

    o = (w * e_ij ).mean()

    return o


def projective_depth_cost(dx, dy):

    x_mask = dx> 0
    y_mask = dy> 0
    depth_error = (dx - dy) ** 2
    depth_error = depth_error[y_mask * x_mask]
    silh_loss = torch.mean(depth_error)

    return silh_loss

def silhouette_cost(x, y):

    x_mask = x[..., 0] > 0
    y_mask = y[..., 0] > 0
    silh_error = (x - y) ** 2
    silh_error = silh_error[~y_mask]
    silh_loss = torch.mean(silh_error)

    return silh_loss

def landmark_cost(x, y, landmarks):
    x = x [ landmarks[0] ]
    y = y [ landmarks[1] ]
    loss = torch.mean(
        torch.sum( (x-y)**2, dim=-1 ))
    return loss

def landmark_cost_p(x, y):

    loss = torch.sum( (x-y)**2, dim=-1 )
    return loss.mean(), loss

def chamfer_dist(src_pcd,   tgt_pcd):
    '''
    :param src_pcd: warpped_pcd
    :param R: node_rotations
    :param t: node_translations
    :param data:
    :return:
    '''

    """chamfer distance"""
    samples = 1000
    src=torch.randperm(src_pcd.shape[0])
    tgt=torch.randperm(tgt_pcd.shape[0])
    s_sample = src_pcd[ src[:samples]]
    t_sample = tgt_pcd[ tgt[:samples]]
    cham_dist = compute_truncated_chamfer_distance(s_sample[None], t_sample[None], trunc=0.3)

    return cham_dist


