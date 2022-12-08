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
        dx = self.body(self.embedding(x))
        print("dx", dx)
        return x + 0.001 * dx


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

        return self.net_final(h)


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
