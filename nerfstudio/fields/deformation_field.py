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
from nerfstudio.fields.geom_utils import *


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
        deform_rot = None
        new_x = self.body(self.embedding(x))
        # print("type deform output", type(new_x), tuple, self.body)
        if type(new_x) == tuple:
            new_x, deform_rot = new_x
        dx = (new_x - x).detach()
        print("dx", dx.shape, torch.abs(dx).max(0)[0])
        return new_x, deform_rot


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
            torch.nn.init.xavier_uniform_(self.net_final.weight, gain=1e-3)
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


class DeformationMLPSE3(nn.Module):
    def __init__(self, input_ch, W=128, D=6, skips=(4,), aabb=None, **kwargs):
        super(DeformationMLPSE3, self).__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        print("aabb in SE3 deform", self.aabb)
        # self.trunk = MLP(input_ch, W, None, D, skips)
        # self.w_net = MLP(W, None, 3, 0, None)
        # self.v_net = MLP(W, None, 3, 0, None)
        # self.theta_net = MLP(W, None, 1, 0, None)

        # # simple SO(2)
        # self.theta = Parameter(torch.Tensor([0.0]), requires_grad=True)
        # self.forward = self.forward_simple_SO2

        # # SO(3), axis and theta
        # self.theta = Parameter(torch.Tensor([0.0]), requires_grad=True)
        # self.w = Parameter(torch.Tensor([0.0, 0.0, 1.0]), requires_grad=True)
        # self.forward = self.forward_axis_theta

        # SO(3), Euler angle
        self.theta = Parameter(torch.Tensor([0.0, 0.0, 0.0]), requires_grad=True)
        self.forward = self.forward_euler_angle

        # self.v = Parameter(torch.zeros(3,), requires_grad=True)

    def deform_direction(self, x, deform_rot):

        assert len(x.shape) == 2
        print("debug dir norm inside SE3", x.norm(dim=-1).max(), x.norm(dim=-1).min())
        # x *= self.aabb[1] - self.aabb[0]

        print("debug dir norm inside SE3 2", x.norm(dim=-1).max(), x.norm(dim=-1).min())
        print(deform_rot[0])
        x = deform_rot @ x.reshape(-1, 3, 1)
        x = x.reshape(-1, 3)
        # x /= self.aabb[1] - self.aabb[0]
        return x

    def forward_euler_angle(self, new_pts):  # eular angle
        assert len(new_pts.shape) == 2
        theta = self.theta.repeat(new_pts.shape[0], 1) + torch.pi / 2  #  + torch.pi * 5 / 6  # torch.pi / 3
        for i in range(3):
            f = open(f"/home/zt15/projects/nerfstudio/theta_{str(i)}.txt", "a")
            f.write(str(theta[0, i].item()) + " ")
        print("theta", theta[0, :])
        R = R_euler_angle(theta)
        pts_original = to_original_centered(self.aabb, new_pts[:, :3].float())
        pts = (R[0:1] @ pts_original[:, :, None].float())[:, :, 0]  #  + t[0:1]
        pts = centered_to_contract(self.aabb, pts)
        # print("A", new_pts[0, :3], pts[0], pts[0] - new_pts[0, :3], t[0])
        return (pts, (R, self.deform_direction))

    def forward_axis_theta(self, new_pts):  # axis and theta
        assert len(new_pts.shape) == 2
        N = new_pts.shape[0]
        # hidden = self.trunk(torch.ones_like(new_pts))  # 注意，这里要重写，直接生成w, |w|=theta是不合理的，因为theta不能为负，反着转会很困难。
        # w = self.w_net(hidden)
        # v = self.v_net(hidden)
        # # print("debug", w.shape, v.shape, w[0], v[0], hidden[0])
        # theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # [N, 1]
        # eps = 1e-4
        # w = w / (theta + eps)
        # v = v / (theta + eps)
        # # theta = torch.zeros_like(theta)

        # # print("debug2", w[0], v[0], theta[0])
        # # theta = theta * 0

        # w = torch.ones_like(w)
        # w[:, 0] = 0
        # w[:, 1] = 0
        # v[:] = 0
        # theta = self.theta_net(hidden) - torch.pi / 3
        theta = self.theta.repeat(N, 1) + torch.pi * 5 / 6  # torch.pi / 3
        f = open("/home/zt15/projects/nerfstudio/theta.txt", "a")
        f.write(str(theta[0, 0].item()) + " ")
        w = self.w / self.w.norm(dim=-1)
        w_all = w.repeat(N, 1)
        print("theta & w", theta[0, 0], w)
        R, t = get_transform(w_all, torch.zeros_like(w_all), theta.float())  # (N, 3, 3), (N, 3)
        print("R", R[0])
        # R[:, 0, 0] = torch.cos(theta[:, 0])
        # R[:, 0, 1] = -torch.sin(theta[:, 0])
        # R[:, 1, 0] = torch.sin(theta[:, 0])
        # R[:, 1, 1] = torch.cos(theta[:, 0])
        # print("SE3", R[0:2], t[0:2], theta[0:2])
        # R[1:] = R[0]
        # t[1:] = t[0]
        # print(R[0:1])
        # print(t[0:1])
        # R[:] = torch.eye(3).to(R.device)
        # t[:] = 0
        # print(R[:10], t)
        # print("R", R.shape, t.shape, new_pts[:, :3, None].shape) [N,3,3] [N,3,1]
        pts_original = to_original_centered(self.aabb, new_pts[:, :3].float())
        pts = (R[0:1] @ pts_original[:, :, None].float())[:, :, 0] + t[0:1]
        pts = centered_to_contract(self.aabb, pts)
        # print("A", new_pts[0, :3], pts[0], pts[0] - new_pts[0, :3], t[0])
        return (pts, (R, self.deform_direction))

    def forward_simple_SO2(self, new_pts):  # （这里面已经有很多debug信息了，注意检查）
        assert len(new_pts.shape) == 2
        theta = self.theta.repeat(new_pts.shape[0], 1) + torch.pi / 2  #  + torch.pi * 5 / 6  # torch.pi / 3
        f = open("/home/zt15/projects/nerfstudio/theta.txt", "a")
        f.write(str(theta[0, 0].item()) + " ")
        print("theta", theta[0, 0])
        R = torch.eye(3).to(theta.device).repeat(new_pts.shape[0], 1, 1)
        R[:, 0, 0] = torch.cos(theta[:, 0])
        R[:, 0, 2] = -torch.sin(theta[:, 0])
        R[:, 2, 0] = torch.sin(theta[:, 0])
        R[:, 2, 2] = torch.cos(theta[:, 0])
        pts_original = to_original_centered(self.aabb, new_pts[:, :3].float())
        pts = (R[0:1] @ pts_original[:, :, None].float())[:, :, 0]  #  + t[0:1]
        pts = centered_to_contract(self.aabb, pts)
        # print("A", new_pts[0, :3], pts[0], pts[0] - new_pts[0, :3], t[0])
        return (pts, (R, self.deform_direction))


class Bone(nn.Module):
    def __init__(self, num_bones=1):
        super(Bone, self).__init__()
        self.n = num_bones
        # TODO bones should be located uniformly on the surface. refer to generate_bones in geom_utils.py
        center = torch.zeros((num_bones, 3))
        orient = torch.Tensor([[1, 0, 0, 0]])
        orient = orient.repeat(num_bones, 1)
        scale = torch.zeros(num_bones, 3)
        bones = torch.cat([center, orient, scale], -1)
        self.bones = torch.nn.Parameter(bones)


class SE3(nn.Module):
    def __init__(self, num_bones=1):
        super(SE3, self).__init__()
        self.n = num_bones
        w = torch.zeros((num_bones, 3))
        v = torch.zeros((num_bones, 3))
        torch.nn.init.xavier_uniform_(w, gain=1e-4)
        torch.nn.init.constant_(v, 0.0)
        self.w = torch.nn.Parameter(w)
        self.v = torch.nn.Parameter(v)

    def get_R_t(self):

        # print("R t", get_R_t(torch.Tensor([1, 0, 0]), self.v, True))
        print("R_t", get_R_t(self.w, self.v))
        return get_R_t(self.w, self.v)


class DeformationGaussianBall(nn.Module):
    def __init__(self, input_ch, num_balls=1, **kwargs):
        super(DeformationGaussianBall, self).__init__()

        self.num_balls = num_balls
        self.bones_rst = Bone(num_balls)
        self.tranform_fw = SE3(num_balls)
        self.log_scale = torch.nn.Parameter(torch.tensor(0.0))  # for skinning

    def forward(self, new_pts):  # (N, 3)
        assert len(new_pts.shape) == 2
        bones_dfm = bone_transform(self.bones_rst.bones, self.tranform_fw.get_R_t(), is_vec=True)  # [B, 3], [B, 12]
        skin_backward = skinning(bones_dfm, new_pts, None, self.log_scale)
        print("bones_dfm", bones_dfm)
        print("skin_backward", skin_backward.shape, skin_backward)
        pts, bones_dfm = lbs(self.bones_rst.bones, self.tranform_fw.get_R_t(), skin_backward, new_pts)
        return pts.reshape(-1, 3)


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
    # deformation_field = Deformation(
    #     body_config={"type": DeformationMLPSE3, "input_ch": 3, "D": 6, "W": 128, "skips": [4]},
    #     embedding_config={"multires": 10, "input_dims": 3,},
    # ).cuda()
    # x = torch.rand(100, 3).cuda() - 0.5
    # new_x = deformation_field(x)
    # dx = new_x - x
    # print("dx", torch.abs(dx).max(0)[0])

    deformation_field = Deformation(
        body_config={"type": DeformationGaussianBall}, embedding_config={"multires": 0, "input_dims": 3,}
    ).cuda()
    x = torch.rand(100, 3).cuda() - 0.5
    new_x = deformation_field(x)
    dx = new_x - x
    # print("dx", dx.shape, torch.abs(dx).max(0)[0])


# main()
