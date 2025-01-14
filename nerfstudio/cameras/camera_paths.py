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
Code for camera paths.
"""

from typing import Any, Dict, Optional, Tuple

import torch

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_utils import get_interpolated_poses_many
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length


def get_interpolated_camera_path(cameras: Cameras, steps: int) -> Cameras:
    """Generate a camera path between two cameras.

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices().cpu().numpy()
    poses = cameras.camera_to_worlds().cpu().numpy()
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps)

    cameras = Cameras(fx=Ks[:, 0, 0], fy=Ks[:, 1, 1], cx=Ks[0, 0, 2], cy=Ks[0, 1, 2], camera_to_worlds=poses)
    return cameras


def get_spiral_path(
    camera: Cameras,
    steps: int = 30,
    radius: Optional[float] = None,
    radiuses: Optional[Tuple[float]] = None,
    rots: int = 2,
    zrate: float = 0.5,
) -> Cameras:
    """
    Returns a list of camera in a sprial trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        radius: The radius of the spiral for all xyz directions.
        radiuses: The list of radii for the spiral in xyz directions.
        rots: The number of rotations to apply to the camera.
        zrate: How much to change the z position of the camera.

    Returns:
        A spiral camera path.
    """

    assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."
    assert camera.ndim == 1, "We assume only one batch dim here"
    if radius is not None and radiuses is None:
        rad = torch.tensor([radius] * 3, device=camera.device)
    elif radiuses is not None and radius is None:
        rad = torch.tensor(radiuses, device=camera.device)
    else:
        raise ValueError("Only one of radius or radiuses must be specified.")

    up = camera.camera_to_worlds[0, :3, 2]  # scene is z up
    focal = torch.min(camera.fx[0], camera.fy[0])
    target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

    c2w = camera.camera_to_worlds[0]
    c2wh_global = pose_utils.to4x4(c2w)

    local_c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
        center = (
            torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
        )
        lookat = center - target
        c2w = camera_utils.viewmatrix(lookat, up, center)
        c2wh = pose_utils.to4x4(c2w)
        local_c2whs.append(c2wh)

    new_c2ws = []
    first_cam = torch.Tensor(
        [[-0.1865, -0.3247, 0.9273, 0.8318], [0.9823, -0.0462, 0.1814, 0.2519], [-0.0160, 0.9447, 0.3276, -0.2828]]
    ).to(camera.device)

    for local_c2wh in local_c2whs:
        c2wh = torch.matmul(c2wh_global, local_c2wh)
        new_c2ws.append(c2wh[:3, :4])
    new_c2ws = torch.stack(new_c2ws, dim=0)
    new_c2ws[0] = first_cam
    print(camera, new_c2ws)
    return Cameras(fx=camera.fx[0], fy=camera.fy[0], cx=camera.cx[0], cy=camera.cy[0], camera_to_worlds=new_c2ws,)


def skew(w):

    W = torch.zeros(w.shape[0], 3, 3).to(w.device)
    W[:, 0, 1] = -w[:, 2]
    W[:, 0, 2] = w[:, 1]
    W[:, 1, 0] = w[:, 2]
    W[:, 1, 2] = -w[:, 0]
    W[:, 2, 0] = -w[:, 1]
    W[:, 2, 1] = w[:, 0]
    return W


def exp_so3(w, theta):  # w[N,3], theta[N,1]

    W = skew(w)
    return (
        torch.eye(3).repeat(W.shape[0], 1, 1).to(W.device)
        + torch.sin(theta[:, :, None]) * W
        + (1 - torch.cos(theta[:, :, None])) * (W @ W)
    )


def get_spiral_path2(
    camera: Cameras,
    steps: int = 30,
    radius: Optional[float] = None,
    radiuses: Optional[Tuple[float]] = None,
    rots: int = 1,
    zrate: float = 0.5,
) -> Cameras:
    """
    Returns a list of camera in a sprial trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        radius: The radius of the spiral for all xyz directions.
        radiuses: The list of radii for the spiral in xyz directions.
        rots: The number of rotations to apply to the camera.
        zrate: How much to change the z position of the camera.

    Returns:
        A spiral camera path.
    """
    print(camera)
    assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."
    assert camera.ndim == 1, "We assume only one batch dim here"
    if radius is not None and radiuses is None:
        rad = torch.tensor([radius] * 3, device=camera.device)
    elif radiuses is not None and radius is None:
        rad = torch.tensor(radiuses, device=camera.device)
    else:
        raise ValueError("Only one of radius or radiuses must be specified.")

    up = camera.camera_to_worlds[0, :3, 2]  # scene is z up
    focal = torch.min(camera.fx[0], camera.fy[0])
    target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

    c2w = camera.camera_to_worlds[0]
    c2wh_global = pose_utils.to4x4(c2w)

    first_cam = torch.Tensor(
        [[-0.1865, -0.3247, 0.9273, 0.8318], [0.9823, -0.0462, 0.1814, 0.2519], [-0.0160, 0.9447, 0.3276, -0.2828]]
    ).to(camera.device)
    local_c2whs = []
    new_c2ws = []

    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
        rot = (
            exp_so3(torch.Tensor([0, 0, 1]).reshape(1, 3), torch.Tensor([theta]).reshape(1, 1))
            .reshape(3, 3)
            .to(camera.device)
        )
        temp = rot @ first_cam
        center = (
            torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
        )
        lookat = center - target
        c2w = camera_utils.viewmatrix(lookat, up, center)
        c2wh = pose_utils.to4x4(c2w)
        c2wh[:3] = temp
        local_c2whs.append(c2wh)
        new_c2ws.append(temp)
    new_c2ws = torch.stack(new_c2ws, dim=0)
    new_c2ws[0] = first_cam
    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=new_c2ws,
        width=camera.width[0],
        height=camera.height[0],
    )


def get_path_from_json(camera_path: Dict[str, Any]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # pose
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)[:3]
        c2ws.append(c2w)
        # field of view
        fov = camera["fov"]
        focal_length = three_js_perspective_camera_focal_length(fov, image_height)
        fxs.append(focal_length)
        fys.append(focal_length)

    camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    return Cameras(fx=fx, fy=fy, cx=image_width / 2, cy=image_height / 2, camera_to_worlds=camera_to_worlds,)
