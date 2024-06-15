import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from nerfstudio.fields.geom_utils import *


class Deformation(nn.Module):
    def __init__(self, body_config: dict, embedding_config: dict):
        super(Deformation, self).__init__()
        self.embedding, self.input_size, self.input_dim, self.n_freq = get_embedder(**embedding_config)
        body_config["input_ch"] = self.input_size
        body_config["input_dim"] = self.input_dim
        body_config["n_freq"] = self.n_freq
        self.body = body_config["type"](**body_config)

        self.train_step = None

    def forward(self, x):
        original_shape = x.shape
        res = self.forward2D(x.reshape(-1, x.shape[-1]))
        res[0] = res[0].reshape(*original_shape[:-1], -1)
        return res

    def forward2D(self, x):
        deform_rot = None
        new_x = self.body(self.embedding(x, self.train_step))
        if type(new_x) == tuple:
            new_x, deform_rot = new_x
        dx = (new_x - x).detach()
        return [new_x, deform_rot]

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

    def forward(self, inputs): # inputs: [B, 3 + n_positional_encoding]
        assert len(inputs.shape) == 2

        pts, positional_encoding = inputs[:, :3], inputs[:, 3:].float()
        pts_original = to_original(self.aabb, pts)

        min_dis, min_id = argmin_dis_batch(pts_original, self.vd, self.K)
        pts_deformed, R = self.deform_forward_K(min_dis, min_id, pts_original)

        pts_deformed = to_contract(self.aabb, pts_deformed)
        
        pts_deformed[:, 0][min_dis[:, 0] > 2.1e-4] = 1e9 # if the distance to the EDG is farther than the threshold, we map it to a very far place (1e9), that will get an empty occupancy in outside codes. 
        
        return (pts_deformed, (R, self.deform_direction)) # deform_direction is used in outside code.

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        self.increasing_coeff = self.kwargs["increasing_coeff"]
        self.max_increasing = self.kwargs["max_increasing"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]
        self.N_freqs = N_freqs

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)  #  * 0.5

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, step=None):
        if step is None or self.increasing_coeff is None:
            alpha = torch.ones((self.N_freqs,)).numpy()
        else:
            if self.max_increasing is not None and step > self.max_increasing:
                step = self.max_increasing
            alpha = torch.clamp(step * self.increasing_coeff - torch.arange(0, self.N_freqs), 0, 1).numpy()
        # print("alpha", alpha)
        if not self.kwargs["return_alpha"]:
            res = []
            for i, fn in enumerate(self.embed_fns):
                id = (i - 1) // 2
                if id == -1:
                    res.append(fn(inputs))
                else:
                    res.append(fn(inputs) * alpha[id])
            res = torch.cat(res, -1)
            return res
        else:
            res = []
            for i, fn in enumerate(self.embed_fns):
                res.append(fn(inputs))
            res = torch.cat(res, -1)
            return res, alpha


def get_embedder(
    multires, input_dims, increasing_coeff=None, max_freq_log2=None, return_alpha=False, max_increasing=None
):
    if multires == 0:
        return nn.Identity(), input_dims
    if max_freq_log2 is None:
        max_freq_log2 = multires - 1
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": max_freq_log2,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
        "increasing_coeff": increasing_coeff,
        "max_increasing": max_increasing,
        "return_alpha": return_alpha,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, step=None, eo=embedder_obj: eo.embed(x, step)
    return embed, embedder_obj.out_dim, input_dims, multires

