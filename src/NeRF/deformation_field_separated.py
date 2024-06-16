import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch3d import ops


def argmin_dis_batch(p, q, K=1):
    # print("argmin_dis_batch (pytorch3d)", p.shape, q.shape) # [N, 3], [M, 3]

    result = ops.knn_points(p[None], q[None], K=K)

    return result[0][0, :, :], result[1][0, :, :]

def Rx(R, x):
    # print("Rx", R.shape, x.shape)
    return (R * x.unsqueeze(-2)).sum(-1)


class DeformationGraph(nn.Module):
    def __init__(self, aabb=None, dg_dir = None, **kwargs): # dg_dir: embedded deformation graph directory
        super(DeformationGraph, self).__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.K = 20
        v, vd, R, Rg, g, t = torch.load(dg_dir)

        self.v = v.cuda()
        self.vd = vd.cuda()
        self.R = R.cuda()
        self.Rg = Rg.cuda()
        self.g = g.cuda()
        self.t = t.cuda()
        self.R_inv = torch.inverse(self.R)
        self.n = self.g.shape[0]

    def deform_forward(self, V_id_list, p_deformed):  # from deformed to canonical
        assert p_deformed.ndim == 2
        R_inv = self.R[V_id_list].transpose(-1, -2)
        g = self.g[V_id_list]
        t = self.t[V_id_list]
        p = Rx(R_inv, p_deformed - g - t) + g
        return p, R_inv

    def deform_forward_K(self, dis, V_id_list, p_deformed):
        assert dis.ndim == 2
        K = V_id_list.shape[-1]
        dis_max = dis.max(dim=-1, keepdim=True)[0]
        w = (1.0 - dis / dis_max).square()  # [B, K]
        w /= w.sum(dim=-1, keepdim=True)
        if K == 1:
            w = torch.ones_like(w)
        p, R = [], []
        for i in range(K):
            p_, R_ = self.deform_forward(V_id_list[:, i], p_deformed)
            p.append(p_)
            R.append(R_)
        p = torch.stack(p, dim=1)  # [B, K, 3]
        R = torch.stack(R, dim=1)  # [B, K, 3, 3]
        p = (p * w[:, :, None]).sum(dim=1)  # [B, 3]
        R = (R * w[:, :, None, None]).sum(dim=1)  # [B, 3, 3]

        return p, R

    def deform_direction(self, x, deform_rot): # the function to deform the direction of rays.
        original_shape = x.shape
        x = deform_rot @ x.reshape(-1, 3, 1)
        x = x.reshape(*original_shape)
        return x

    def forward(self, inputs): # inputs: [B, n_point_dim(=3) + n_positional_encoding]
        assert len(inputs.shape) == 2

        pts, positional_encoding = inputs[:, :3], inputs[:, 3:].float()

        min_dis, min_id = argmin_dis_batch(pts_original, self.vd, self.K)
        pts_deformed, R = self.deform_forward_K(min_dis, min_id, pts_original)

        
        pts_deformed[:, 0][min_dis[:, 0] > 2.1e-4] = 1e9 # if the distance to the EDG is farther than the threshold, we map it to a very far place (1e9), that will get an empty occupancy in outside codes. 
        
        return (pts_deformed, (R, self.deform_direction)) # deform_direction is used in outside code.

