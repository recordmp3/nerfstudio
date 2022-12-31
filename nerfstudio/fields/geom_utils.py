import torch
import numpy as np
from pytorch3d import transforms
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


def R_euler_angle(theta):

    n = theta.shape[0]
    R1 = torch.eye(3).repeat(n, 1, 1).to(theta.device)
    R2 = torch.eye(3).repeat(n, 1, 1).to(theta.device)
    R3 = torch.eye(3).repeat(n, 1, 1).to(theta.device)

    R1[:, 0, 0] = torch.cos(theta[:, 0])
    R1[:, 0, 1] = -torch.sin(theta[:, 0])
    R1[:, 1, 0] = torch.sin(theta[:, 0])
    R1[:, 1, 1] = torch.cos(theta[:, 0])

    R2[:, 0, 0] = torch.cos(theta[:, 1])
    R2[:, 0, 2] = -torch.sin(theta[:, 1])
    R2[:, 2, 0] = torch.sin(theta[:, 1])
    R2[:, 2, 2] = torch.cos(theta[:, 1])

    R3[:, 1, 1] = torch.cos(theta[:, 2])
    R3[:, 1, 2] = -torch.sin(theta[:, 2])
    R3[:, 2, 1] = torch.sin(theta[:, 2])
    R3[:, 2, 2] = torch.cos(theta[:, 2])

    return R1 @ R2 @ R3


def to_original(aabb, x):

    assert len(x.shape) == 2
    x = x * (aabb[1] - aabb[0]) + aabb[0]
    return x


def to_original_centered(aabb, x):

    assert len(x.shape) == 2
    x = x * (aabb[1] - aabb[0]) - (aabb[1] - aabb[0]) / 2
    return x


def centered_to_contract(aabb, x):

    assert len(x.shape) == 2
    x = (x + (aabb[1] - aabb[0]) / 2) / (aabb[1] - aabb[0])
    return x


def to_contract(aabb, x):

    assert len(x.shape) == 2
    x = (x - aabb[0]) / (aabb[1] - aabb[0])
    return x


def get_R_t(w, v, mat=False):  # [N, 3]
    w = w.reshape(-1, 3)
    v = v.reshape(-1, 3)
    R = transforms.so3_exponential_map(w)
    t = v
    if mat:
        return R, t
    else:
        return torch.cat([R.reshape(-1, 9), t], -1)


def get_transform(w, v, theta=None):

    if theta is None:
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
    W = skew(w)
    R = exp_so3(W, theta)
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


def bone_transform(bones_in, rts, is_vec=False):
    """  
    bones_in: 1,B,10  - B gaussian ellipsoids of bone coordinates
    rts: ...,B,3,4    - B transforms
    rts are applied to bone coordinate transforms (left multiply)
    is_vec:     whether rts are stored as r1...9,t1...3 vector form
    """
    B = bones_in.shape[-2]
    bones = bones_in.view(-1, B, 10).clone()
    if is_vec:
        rts = rts.view(-1, B, 12)
    else:
        rts = rts.view(-1, B, 3, 4)
    bs = rts.shape[0]

    center = bones[:, :, :3]
    orient = bones[:, :, 3:7]  # real first
    scale = bones[:, :, 7:10]
    if is_vec:
        Rmat = rts[:, :, :9].view(-1, B, 3, 3)
        Tmat = rts[:, :, 9:12].view(-1, B, 3, 1)
    else:
        Rmat = rts[:, :, :3, :3]
        Tmat = rts[:, :, :3, 3:4]
    # print("Rmat", Rmat.shape, Rmat)
    # move bone coordinates (left multiply)
    center = Rmat.matmul(center[..., None])[..., 0] + Tmat[..., 0]
    Rquat = transforms.matrix_to_quaternion(Rmat)
    # print("bone_transform", center, Rquat, orient, Rmat)
    orient = transforms.quaternion_multiply(Rquat, orient)

    if scale.shape[0] == 1:
        scale = scale.repeat(bs, 1, 1)
    bones = torch.cat([center, orient, scale], -1)
    # print("new bones", center, orient, scale)
    return bones


# def gauss_mlp_skinning(xyz, embedding_xyz, bones, pose_code, nerf_skin, skin_aux=None):
def gauss_mlp_skinning(xyz, bones, log_scale):
    """
    xyz:        N_rays, ndepth, 3
    bones:      ... nbones, 10
    pose_code:  ...,1, nchannel
    """
    skin = skinning(bones, xyz, None, log_scale)  # bs, N, B
    return skin


# def mlp_skinning(mlp, code, pts_embed):
#     """
#     code: bs, D          - N D-dimensional pose code
#     pts_embed: bs,N,x    - N point positional embeddings
#     dskin: bs,N,B        - delta skinning matrix
#     """
#     if mlp is None:
#         dskin = None
#     else:
#         dskin = evaluate_mlp(mlp, pts_embed, code=code, chunk=8 * 1024)
#     return dskin


def axis_rotate(orient, mdis):
    bs, N, B, _, _ = mdis.shape
    mdis = (orient * mdis.view(bs, N, B, 1, 3)).sum(4)[..., None]  # faster
    # mdis = orient.matmul(mdis) # bs,N,B,3,1 # slower
    return mdis


def vec_to_sim3(vec):
    """
    vec:      ...,10
    center:   ...,3
    orient:   ...,3,3
    scale:    ...,3
    """
    center = vec[..., :3]
    orient = vec[..., 3:7]  # real first
    orient = F.normalize(orient, 2, -1)
    orient = transforms.quaternion_to_matrix(orient)  # real first
    scale = vec[..., 7:10].exp()
    return center, orient, scale


def skinning_chunk(bones, pts, dskin=None, log_scale=None):
    # def skinning(bones, pts, dskin=None, skin_aux=None):
    """
    bone: bs,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
    skin: bs,N,B   - skinning matrix
    """
    device = pts.device
    bs, N, _ = pts.shape
    B = bones.shape[-2]
    if bones.dim() == 2:
        bones = bones[None].repeat(bs, 1, 1)
    bones = bones.view(-1, B, 10)
    if bones.shape[0] == 1:
        bones = bones.repeat(bs, 1, 1)
    # print("chunk", bones.shape, pts.shape)

    center, orient, scale = vec_to_sim3(bones)
    orient = orient.permute(0, 1, 3, 2)  # transpose R

    # mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
    # transform a vector to the local coordinate
    mdis = center.view(bs, 1, B, 3) - pts.view(bs, N, 1, 3)  # bs,N,B,3
    mdis = axis_rotate(orient.view(bs, 1, B, 3, 3), mdis[..., None])
    mdis = mdis[..., 0]
    mdis = scale.view(bs, 1, B, 3) * mdis.pow(2)
    # log_scale (being optimized) controls temporature of the skinning weight softmax
    # multiply 1000 to make the weights more concentrated initially
    inv_temperature = 1000 * log_scale.exp()
    mdis = -inv_temperature * mdis.sum(3)  # bs,N,B

    if dskin is not None:
        mdis = mdis + dskin

    skin = mdis.softmax(2)
    return skin


def skinning(bones, pts, dskin=None, log_scale=None):  # [B, 10], [N, 3]
    """
    bone: ...,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points
    skin: bs,N,B   - skinning matrix
    """
    chunk = 4096
    pts = pts.reshape(pts.shape[0], 1, -1)
    print("skin", bones.shape, pts.shape, log_scale)
    bs, N, _ = pts.shape
    B = bones.shape[-2]
    if bones.dim() == 2:
        bones = bones[None].repeat(bs, 1, 1)
    bones = bones.view(-1, B, 10)

    skin = []
    for i in range(0, bs, chunk):
        if dskin is None:
            dskin_chunk = None
        else:
            dskin_chunk = dskin[i : i + chunk]
        skin_chunk = skinning_chunk(bones[i : i + chunk], pts[i : i + chunk], dskin=dskin_chunk, log_scale=log_scale)
        skin.append(skin_chunk)
    skin = torch.cat(skin, 0)
    return skin[:, :, :]  # [bs, N=1, B]


def lbs(bones, rts_fw, skin, xyz_in, backward=True):
    """
    bones: bs,B,10       - B gaussian ellipsoids indicating rest bone coordinates
    rts_fw: bs,B,12       - B rigid transforms, applied to the rest bones
    xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
    """
    xyz_in = xyz_in[:, None]
    B = bones.shape[-2]
    N = xyz_in.shape[-2]
    xyz_in = xyz_in.view(-1, N, 3)
    bs = xyz_in.shape[0]
    bones = bones.view(-1, B, 10)
    if bones.shape[0] == 1:
        bones = bones.repeat(bs, 1, 1)
    rts_fw = rts_fw.view(-1, B, 12)  # B,12
    if rts_fw.shape[0] == 1:
        rts_fw = rts_fw.repeat(bs, 1, 1)
    rmat = rts_fw[:, :, :9]
    rmat = rmat.view(bs, B, 3, 3)
    tmat = rts_fw[:, :, 9:12]
    rts_fw = torch.cat([rmat, tmat[..., None]], -1)
    rts_fw = rts_fw.view(-1, B, 3, 4)
    print("lbs", bones.shape, rts_fw.shape, xyz_in.shape)
    if backward:
        bones_dfm = bone_transform(bones, rts_fw)  # bone coordinates after deform
        rts_bw = rts_invert(rts_fw)
        xyz = blend_skinning(bones_dfm, rts_bw, skin, xyz_in)
    else:
        xyz = blend_skinning(bones.repeat(bs, 1, 1), rts_fw, skin, xyz_in)
        bones_dfm = bone_transform(bones, rts_fw)  # bone coordinates after deform
    return xyz, bones_dfm


def rts_invert(rts_in):
    """
    rts: ...,3,4   - B ririd transforms
    """
    rts = rts_in.view(-1, 3, 4).clone()
    Rmat = rts[:, :3, :3]  # bs, B, 3,3
    Tmat = rts[:, :3, 3:]
    Rmat_i = Rmat.permute(0, 2, 1)
    Tmat_i = -Rmat_i.matmul(Tmat)
    rts_i = torch.cat([Rmat_i, Tmat_i], -1)
    rts_i = rts_i.view(rts_in.shape)
    return rts_i


def blend_skinning_chunk(bones, rts, skin, pts):
    # def blend_skinning(bones, rts, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    B = rts.shape[-3]
    N = pts.shape[-2]
    pts = pts.view(-1, N, 3)
    rts = rts.view(-1, B, 3, 4)
    Rmat = rts[:, :, :3, :3]  # bs, B, 3,3
    Tmat = rts[:, :, :3, 3]
    device = Tmat.device

    ## convert from bone to root transforms
    # bones = bones.view(-1,B,10)
    # bs = Rmat.shape[0]
    # center = bones[:,:,:3]
    # orient = bones[:,:,3:7] # real first
    # orient = F.normalize(orient, 2,-1)
    # orient = transforms.quaternion_to_matrix(orient) # real first
    # gmat = torch.eye(4)[None,None].repeat(bs, B, 1, 1).to(device)
    #
    ## root to bone
    # gmat_r2b = gmat.clone()
    # gmat_r2b[:,:,:3,:3] = orient.permute(0,1,3,2)
    # gmat_r2b[:,:,:3,3] = -orient.permute(0,1,3,2).matmul(center[...,None])[...,0]

    ## bone to root
    # gmat_b2r = gmat.clone()
    # gmat_b2r[:,:,:3,:3] = orient
    # gmat_b2r[:,:,:3,3] = center

    ## bone to bone
    # gmat_b = gmat.clone()
    # gmat_b[:,:,:3,:3] = Rmat
    # gmat_b[:,:,:3,3] = Tmat

    # gmat = gmat_b2r.matmul(gmat_b.matmul(gmat_r2b))
    # Rmat = gmat[:,:,:3,:3]
    # Tmat = gmat[:,:,:3,3]

    # Gi=sum(wbGb), V=RV+T
    Rmat_w = (skin[..., None, None] * Rmat[:, None]).sum(2)  # bs,N,B,3
    Tmat_w = (skin[..., None] * Tmat[:, None]).sum(2)  # bs,N,B,3
    pts = Rmat_w.matmul(pts[..., None]) + Tmat_w[..., None]
    pts = pts[..., 0]
    return pts


def blend_skinning(bones, rts, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    print("blend", bones.shape, rts.shape, pts.shape, skin.shape)
    chunk = 4096
    B = rts.shape[-3]
    N = pts.shape[-2]
    bones = bones.view(-1, B, 10)
    pts = pts.view(-1, N, 3)
    rts = rts.view(-1, B, 3, 4)
    bs = pts.shape[0]

    pts_out = []
    for i in range(0, bs, chunk):
        pts_chunk = blend_skinning_chunk(
            bones[i : i + chunk], rts[i : i + chunk], skin[i : i + chunk], pts[i : i + chunk]
        )
        pts_out.append(pts_chunk)
    pts = torch.cat(pts_out, 0)
    return pts
