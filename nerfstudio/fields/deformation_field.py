# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instant-NGP field implementations using tiny-cuda-nn, torch, ....
"""


from typing import Optional

import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfacc import ContractionType, contract
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field


class Deformation(nn.Module):
    def __init__(self, body_config: dict, embedding_config: dict):
        super(Deformation, self).__init__()
        # self.temp = nn.Linear(3, 3)
        # torch.nn.init.eye_(self.temp.weight)
        # torch.nn.init.xavier_uniform_(self.temp)
        # torch.nn.init.constant_(self.temp.bias, 0.0)
        self.embedding, self.input_size = get_embedder(**embedding_config)
        body_config["input_ch"] = self.input_size
        self.body = body_config["type"](**body_config)

    def forward(self, x):
        # return self.temp(x)
        # print("enter Deformation")
        new_x = self.body(self.embedding(x))
        dx = (new_x - x).detach()
        print("dx", torch.abs(dx).max(0)[0])
        return new_x


class MLP(nn.Module):
    def __init__(self, input_ch, W, out, D, skips):
        super(MLP, self).__init__()
        self.input_ch = input_ch
        self.W = W
        self.D = D
        self.out = out
        self.skips = skips
        if D == 0:
            self.net = []
            self.net_final = nn.Linear(self.input_ch, self.out)
            torch.nn.init.xavier_uniform_(self.net_final.weight, gain=1e-4)
            torch.nn.init.constant_(self.net_final.bias, 0.0)
        else:
            layers = [nn.Linear(self.input_ch, self.W)]
            for i in range(self.D - 1):
                in_channels = self.W
                if i in self.skips:
                    in_channels += self.input_ch
                layers += [nn.Linear(in_channels, self.W)]
            self.net = nn.ModuleList(layers)
            for x in self.net:
                torch.nn.init.xavier_uniform_(x.weight, gain=1e-3)
                torch.nn.init.constant_(x.bias, 0.0)
            if self.out is not None:
                self.net_final = nn.Linear(self.W, self.out)
                torch.nn.init.xavier_uniform_(self.net_final.weight, gain=1e-3)
                torch.nn.init.constant_(self.net_final.bias, 0.0)

    def forward(self, new_pts):
        h = new_pts
        for i in range(len(self.net)):
            h = self.net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)
        if self.out is not None:
            h = self.net_final(h)
        return h


def get_transform(w, v, theta):

    W = skew(w)
    R = exp_so3(W, theta)
    print(W.shape, R.shape)
    print(
        (
            theta[:, :, None] * torch.eye(3).repeat(W.shape[0], 1, 1).to(W.device)
            + (1.0 - torch.cos(theta[:, :, None])) * W
            + (theta[:, :, None] - torch.sin(theta[:, :, None])) * W @ W
        ).shape,
        v.shape,
        v[..., None].shape,
    )
    t = (
        theta[:, :, None] * torch.eye(3).repeat(W.shape[0], 1, 1).to(W.device)
        + (1.0 - torch.cos(theta[:, :, None])) * W
        + (theta[:, :, None] - torch.sin(theta[:, :, None])) * W @ W
    ) @ v[..., None]
    t = t[:, :, 0]
    return R, t.reshape(R.shape[0], 3)


def exp_so3(W, theta):  # W[N,3,3], theta[N,1]

    return (
        torch.eye(3).repeat(W.shape[0], 1, 1).to(W.device)
        + torch.sin(theta[:, :, None]) * W
        + (1 - torch.cos(theta[:, :, None])) * (W @ W)
    )


def skew(w):

    W = torch.zeros(w.shape[0], 3, 3).to(w.device)
    W[:, 0, 1] = -w[:, 2]
    W[:, 0, 2] = w[:, 1]
    W[:, 1, 0] = w[:, 2]
    W[:, 1, 2] = -w[:, 0]
    W[:, 2, 0] = -w[:, 1]
    W[:, 2, 1] = w[:, 0]
    return W


class DeformationMLPSE3(nn.Module):
    def __init__(self, input_ch, W=128, D=6, skips=(4,), **kwargs):
        super(DeformationMLPSE3, self).__init__()

        self.trunk = MLP(input_ch, W, None, D, skips)
        self.w_net = MLP(W, None, 3, 0, None)
        self.v_net = MLP(W, None, 3, 0, None)

    def forward(self, new_pts):  # 继续debug这里（这里面已经有很多debug信息了，注意检查）
        assert len(new_pts.shape) == 2
        hidden = self.trunk(torch.ones_like(new_pts))
        w = self.w_net(hidden)
        v = self.v_net(hidden)
        print("debug", w.shape, v.shape, w[0], v[0], hidden[0])
        theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # [N, 1]
        eps = 1e-4
        w = w / (theta + eps)
        v = v / (theta + eps)
        # theta = torch.zeros_like(theta)

        print("debug2", w[0], v[0], theta[0])
        # theta = theta * 0
        R, t = get_transform(w, v, theta)  # (N, 3, 3), (N, 3)
        print(R[0:2], t[0:2])
        # R[1:] = R[0]
        # t[1:] = t[0]
        print(R[0:1])
        print(t[0:1])
        # R[:] = torch.eye(3).to(R.device)
        # t[:] = 0
        # print(R[:10], t)
        # print("R", R.shape, t.shape, new_pts[:, :3, None].shape) [N,3,3] [N,3,1]
        pts = (R[0:1] @ new_pts[:, :3, None].float())[:, :, 0] + t[0:1]
        print("A", new_pts[0, :3], pts[0], pts[0] - new_pts[0, :3], t[0])
        return pts


class DeformationMLPDeltaX(nn.Module):
    def __init__(self, input_ch, W, D, skips, **kwargs):
        super(DeformationMLPDeltaX, self).__init__()

        self.input_ch = input_ch
        self.W = W
        self.D = D
        self.skips = skips
        layers = [nn.Linear(self.input_ch, self.W)]
        for i in range(self.D - 1):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.W)]
        self.net = nn.ModuleList(layers)
        self.net_final = nn.Linear(self.W, 3)

    def forward(self, new_pts):
        h = new_pts
        for i in range(len(self.net)):
            h = self.net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return new_pts[:, :3] + 1e-3 * self.net_final(h)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims):
    if multires == 0:
        return nn.Identity(), input_dims

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def main():
    deformation_field = Deformation(
        body_config={"type": DeformationMLPSE3, "input_ch": 3, "D": 6, "W": 128, "skips": [4]},
        embedding_config={"multires": 10, "input_dims": 3,},
    ).cuda()
    x = torch.rand(100, 3).cuda() - 0.5
    new_x = deformation_field(x)
    dx = new_x - x
    print("dx", torch.abs(dx).max(0)[0])


# main()
