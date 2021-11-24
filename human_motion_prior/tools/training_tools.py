# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.01.02
import torch
import numpy as np
import sys
import torch.distributed as dist

from human_motion_prior.body_model.smpl_model.smpl import SMPL
from human_motion_prior.body_model.smpl_model import config


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.reset(patience)

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

    def reset(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf


batch_size = 12
# smpl = SMPL(config.SMPL_MODEL_DIR,
#             batch_size=batch_size * 128,
#             create_transl=False, gender='male').eval().cuda()


def get_joint(vertices, smpl=None):
    body_vertices = vertices
    joints_45 = smpl.get_joints_ori()
    joints_17 = smpl.get_joints_h36m(body_vertices)
    s3d21 = joints_17[:, [0, 7, 8, 9, 10, 1, 2, 3, 3, 4, 5, 6, 6, 11, 12, 13, 13, 14, 15, 16, 16]]
    s3d21[:, 8] = joints_45[:, 10]
    s3d21[:, 12] = joints_45[:, 11]
    s3d21[:, 16] = joints_45[:, 22]
    s3d21[:, 20] = joints_45[:, 23]
    
    s3d21 = s3d21 - s3d21[:, [0]]
    s3d21 = s3d21[:, [0, 9, 10, 11, 5, 6, 7, 1, 2, 3, 4, 13, 14, 15, 17, 18, 19, 12, 8]]                                                                                                                               
                
    return s3d21


def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t


def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2


def mpjpe_loss(local_pose, root_orient, trans, gt_item, add_hand=True):
    device = local_pose.device
    dtype = local_pose.dtype

    total_frm_num = local_pose.shape[0]   # batch_size * frm_num

    local_pose = local_pose.reshape(total_frm_num, 21 * 3).detach()
    root_orient = root_orient.reshape(total_frm_num, 3).detach()
    trans = trans.reshape(total_frm_num, 1, 3).detach()

    gt_local_pose = gt_item['pose_aa'].reshape(total_frm_num, 21 * 3).detach()
    gt_root_orient = gt_item['root_orient_aa'].reshape(total_frm_num, 3).detach()
    gt_trans = gt_item['trans'].reshape(total_frm_num, 1, 3).detach()

    if add_hand:
        pose_pad = torch.zeros((total_frm_num, 2 * 3), device=device, dtype=dtype)
        local_pose = torch.cat([local_pose, pose_pad], dim=-1)
        gt_local_pose = torch.cat([gt_local_pose, pose_pad], dim=-1)
        del pose_pad

    smpl_out = smpl(global_orient=root_orient, body_pose=local_pose)
    p3ds = get_joint(smpl_out.vertices, smpl)[:, :-2].cpu().detach().numpy()

    gt_smpl_out = smpl(global_orient=gt_root_orient, body_pose=gt_local_pose)
    gt_p3ds = get_joint(gt_smpl_out.vertices, smpl)[:, :-2].cpu().detach().numpy()

    # print(gt_p3ds.shape)
    pa_p3ds = np.zeros_like(p3ds)
    for i in range(gt_p3ds.shape[0]):
        pa_p3ds[i] = rigid_align(p3ds[i], gt_p3ds[i])

    pa_mpjpe = np.mean(np.sqrt(np.sum(((pa_p3ds - gt_p3ds) ** 2), axis=-1)), axis=-1)
    normalize_mpjpe = np.mean(np.sqrt(np.sum(((p3ds - gt_p3ds) ** 2), axis=-1)), axis=-1)

    p3ds = p3ds + trans.cpu().detach().numpy()
    gt_p3ds = gt_p3ds + gt_trans.reshape(-1, 1, 3).cpu().detach().numpy()

    mpjpe = np.mean(np.sqrt(np.sum(((p3ds - gt_p3ds) ** 2), axis=-1)), axis=-1)

    ################### ACC ##################
    p3ds = p3ds.reshape(batch_size, 128, -1, 3) * 1000
    v = p3ds[:, 1:, ...] - p3ds[:, :-1, ...]  # (B, FRM-1, 17, 3)
    acc = v[:, 1:, ...] - v[:, :-1, ...]      # (B, FRM-2, 17, 3)
    batch_acc = np.mean(np.linalg.norm(acc, axis=-1), axis=-1)

    print(np.mean(mpjpe) * 1000, np.mean(normalize_mpjpe) * 1000, np.mean(pa_mpjpe) * 1000, np.mean(batch_acc))
    return mpjpe.sum(), normalize_mpjpe.sum(), pa_mpjpe.sum(), batch_acc.sum(), mpjpe.shape[0]

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)

    return tensor


def cal_bone_len(p3ds, kinematic=None):
    if kinematic is None:
        kinematic = MI_KINEMATIC
    batch_size = p3ds.shape[0]
    real_bonelen = torch.zeros((batch_size, len(kinematic), 1))
    for idx, bone in enumerate(kinematic):
        id1 = bone[0]
        id2 = bone[1]
        bl = torch.norm(p3ds[:, id1] - p3ds[:, id2], dim=-1, keepdim=True)
        real_bonelen[:, idx] = bl

    return real_bonelen

def adapt_3d_np(s3ds, bonelens, parents=None):
    """
    change bone lengths
    :param s3ds: B x K x 3
    :param bonelens: B x (K-1) x 3
    :return: new_s3d ()
    """
    if parents is None:
        parents = MI_PARENT
    bonelens = bonelens.reshape(1, -1, 1).repeat(s3ds.shape[0], axis=0)
    new_s3d = np.zeros_like(s3ds)
    num_joint = new_s3d.shape[1]
    for i in range(num_joint):
        parent = parents[i]
        if parent == -1:
            new_s3d[:, i] = s3ds[:, i]
        else:
            bone = s3ds[:, i] - s3ds[:, parent]
            bone_len = np.linalg.norm(bone, axis=-1, keepdims=True) + 1e-6
            new_s3d[:, i] = new_s3d[:, parent] + bone / bone_len * bonelens[:, i - 1]
    return new_s3d

def get_p3ds(pose_aa, root_orient, trans=None, device=torch.device('cpu')):
    bs, frame_num = pose_aa.shape[:2]
    pose_aa = pose_aa.reshape(bs * frame_num, -1)
    root_orient = root_orient.reshape(bs * frame_num, -1)

    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=frame_num * bs,
                create_transl=False,
                gender="MALE").eval().to(device)

    pose_pad = torch.zeros((bs * frame_num, 2 * 3), device=device, dtype=torch.float32)
    pose_aa_pad = torch.cat([pose_aa, pose_pad], dim=-1)
    smpl_out = smpl(global_orient=root_orient, body_pose=pose_aa_pad)
    p3ds = get_joint(smpl_out.vertices, smpl)[:, :]
    p3ds = p3ds.reshape(bs, frame_num, -1, 3)

    if trans is not None:
        p3ds += trans.unsqueeze(-2)

    return p3ds
