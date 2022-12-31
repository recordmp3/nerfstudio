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
Implementation of Instant NGP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import nerfacc
import torch
from nerfacc import ContractionType
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors
from nerfstudio.fields.deformation_field import Deformation, DeformationMLPDeltaX, DeformationMLPSE3
from nerfacc import contract


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: NGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    max_num_samples_per_ray: int = 24
    """Number of samples in field evaluation."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    contraction_type: ContractionType = ContractionType.AABB
    """Resolution of the grid used for the field."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: float = 0.001
    """Minimum step size for rendering."""
    near_plane: float = 0.2
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    randomize_background: bool = False
    """Whether to randomize the background color."""
    deformation_status: str = "inactive"


class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: InstantNGPModelConfig
    field: TCNNInstantNGPField

    def __init__(self, config: InstantNGPModelConfig, keypoints_old=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        self.keypoints_old = None
        self.n_keypoints = 0
        if keypoints_old is not None:
            self.keypoints_old = Parameter(keypoints_old, requires_grad=False)  # [N,3]
            self.n_keypoints = self.keypoints_old.shape[0]

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.field = TCNNInstantNGPField(
            aabb=self.scene_box.aabb,
            contraction_type=self.config.contraction_type,
            use_appearance_embedding=self.config.use_appearance_embedding,
            num_images=self.num_train_data,
        )
        print("field param name in ingp.py", [(x, y.shape) for x, y in self.field.state_dict().items()])
        self.deformation_field = None
        if self.config.deformation_status != "inactive":
            self.deformation_field = Deformation(
                body_config={"type": DeformationMLPDeltaX, "D": 6, "W": 128, "skips": [4]},
                # body_config={
                #     "type": DeformationMLPSE3,
                #     "input_ch": 3,
                #     "D": 6,
                #     "W": 128,
                #     "skips": [4],
                #     "aabb": self.scene_box.aabb,
                # },
                embedding_config={"multires": 10, "input_dims": 3,},
                # contractor=lambda positions_flat: contract(
                #     x=positions_flat, roi=self.field.aabb, type=self.field.contraction_type
                # ),
            )  # type: ignore
            self.field.deformation_field = lambda x: self.deformation_field(x)

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        # Sampler
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler = VolumetricSampler(
            scene_aabb=vol_sampler_aabb, occupancy_grid=self.occupancy_grid, density_fn=self.field.density_fn,
        )

        # renderers
        background_color = "random" if self.config.randomize_background else colors.WHITE
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()
        self.keypoint_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            self.occupancy_grid.every_n_step(
                step=step, occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        if self.deformation_field is not None:
            param_groups["deformation_fields"] = list(self.deformation_field.parameters())
        return param_groups

    def get_outputs1(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, packed_info, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
            )

        return ray_bundle, ray_samples, packed_info, ray_indices

    def get_outputs2(self, ray_bundle, ray_samples, packed_info, ray_indices):
        num_rays = len(ray_bundle)
        field_outputs = self.field(ray_samples)

        # accumulation
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY],
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB], weights=weights, ray_indices=ray_indices, num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        alive_ray_mask = accumulation.squeeze(-1) > 0
        # print("depth shape", depth.shape) # [N, 1]
        pt = ray_bundle.get_positions_depth(depth)

        if ray_bundle.extra_info["keypoints_included"]:
            pt_keypoint = pt[-self.n_keypoints :]
            pt_keypoint_c = self.field.get_new_pt_from_deform(pt_keypoint)
            outputs = {
                "rgb": rgb[: -self.n_keypoints],
                "accumulation": accumulation[: -self.n_keypoints],
                "depth": depth[: -self.n_keypoints],
                "alive_ray_mask": alive_ray_mask[: -self.n_keypoints],  # the rays we kept from sampler
                "num_samples_per_ray": packed_info[: -self.n_keypoints, 1],
                "pt_keypoint_c": pt_keypoint_c,
                "alive_ray_mask_keypoint": alive_ray_mask[-self.n_keypoints :],
            }
        else:
            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
                "num_samples_per_ray": packed_info[:, 1],
            }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        mask = outputs["alive_ray_mask"]
        # print(mask.float().mean()) # around 0.2
        rgb_loss = self.rgb_loss(image[mask], outputs["rgb"][mask])
        keypoint_loss = torch.zeros((1,))
        has_keypoint = False
        if "pt_keypoint_c" in outputs.keys():
            has_keypoint = True
            mask_keypoint = outputs["alive_ray_mask_keypoint"]
            pt_keypoint_c = outputs["pt_keypoint_c"]
            keypoint_loss = self.keypoint_loss(pt_keypoint_c[mask_keypoint], self.keypoints_old[mask_keypoint].detach())
        rgb_loss_all = self.rgb_loss(image, outputs["rgb"])

        f = open("/home/zt15/projects/nerfstudio/rgb.txt", "a")
        f.write(str(keypoint_loss.item()) + " ")
        print(
            "************************** loss",
            rgb_loss.item(),
            rgb_loss_all.item(),
            keypoint_loss.item(),
            image[mask].shape,
            outputs["rgb"][mask].shape,
        )
        if has_keypoint:
            loss_dict = {
                "keypoint_loss": keypoint_loss,
            }
        else:
            loss_dict = {
                "rgb_loss": rgb_loss_all,
            }
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"],)
        alive_ray_mask = colormaps.apply_colormap(outputs["alive_ray_mask"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_alive_ray_mask = torch.cat([alive_ray_mask], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "alive_ray_mask": combined_alive_ray_mask,
        }

        return metrics_dict, images_dict
