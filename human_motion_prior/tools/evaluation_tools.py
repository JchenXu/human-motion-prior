import torch
import numpy as np

c2c = lambda x : x.cpu().detach().numpy()

class KinematicConfig:
    def __init__(self):
        self.visual_mesh = False
        self.visual_p3d = True
        self.KINEMATIC = [[0, 1], [1, 2], [2, 3], [0, 4],  # 0:Hip-RHip, 1:RHip-RKnee, 2:RKnee-RAnkle, 3:Hip-LHip
                          [4, 5], [5, 6], [0, 7], [7, 8],  # 4:LHip-LKnee, 5:LKnee-LAnkle, 6:Hip-Chest, 7:Chest-Neck
                          [8, 9], [9, 10], [8, 11], [11, 12],  # 8:Neck-Nose, 9:Nose-Head, 10:Neck-LShd, 11:LShd-LElbow,
                          [12, 13], [8, 14], [14, 15], [15, 16],
                          # 12:LElbow-LWrist, 13:Neck-RShd, 14:RShd-RElbow, 15:RElbow-RWrist
                          [3, 17], [6, 18]]  # 16:RAnkle-RFoot, 17:LAnkle-LFoot
        self.PARENT = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 3, 6]
        self.SDK21_to_19 = [0, 9, 10, 11, 5, 6, 7, 1, 2, 3, 4, 13, 14, 15, 17, 18, 19, 12, 8]
        self.tar_bone_len = np.array([[0.1456, 0.4478, 0.4378,
                                       0.1449, 0.4498, 0.4336,
                                       0.2562, 0.2475, 0.1249, 0.1158,
                                       0.1712, 0.2831, 0.2393,
                                       0.1725, 0.2881, 0.2424,
                                       0.1816, 0.1762]])
        self.convert_mat = np.array([[1, 0, 0],
                                     [0, -1, 0],
                                     [0, 0, -1]])


cfg = KinematicConfig()

def get_p3ds(smpl, pose_aa, root_orient, trans=None, return_vert=False):
    bs, frame_num = pose_aa.shape[:2]

    pose_aa = pose_aa.reshape(bs * frame_num, -1)
    pose_pad = torch.zeros((bs * frame_num, 2 * 3), device=torch.device('cuda'), dtype=torch.float32)
    pose_aa_pad = torch.cat([pose_aa, pose_pad], dim=-1)

    root_orient = root_orient.reshape(bs * frame_num, -1)

    smpl_out = smpl(global_orient=root_orient, body_pose=pose_aa_pad)
    body_vertices = smpl_out.vertices

    joints_45 = smpl.get_joints_ori()
    joints_17 = smpl.get_joints_h36m(body_vertices)
    s3d21 = joints_17[:, [0, 7, 8, 9, 10, 1, 2, 3, 3, 4, 5, 6, 6, 11, 12, 13, 13, 14, 15, 16, 16]]
    s3d21[:, 8] = joints_45[:, 10]
    s3d21[:, 12] = joints_45[:, 11]
    s3d21[:, 16] = joints_45[:, 22]
    s3d21[:, 20] = joints_45[:, 23]

    s3d21 = s3d21 - s3d21[:, [0]]
    s3d21 = s3d21[:, cfg.SDK21_to_19]

    p3ds = s3d21.reshape(bs, frame_num, -1, 3)

    if trans is not None:
        p3ds += trans.unsqueeze(-2)

    if return_vert:
        return p3ds, body_vertices
    else:
        return p3ds

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


def mpjpe_loss(smpl, local_pose, root_orient, trans, gt_item, add_hand=True):
    device = local_pose.device
    dtype = local_pose.dtype

    total_frm_num = local_pose.shape[0]   # batch_size * frm_num
    batch_size = total_frm_num // 128

    local_pose = local_pose.reshape(batch_size, 128, 21 * 3).detach()
    root_orient = root_orient.reshape(batch_size, 128, 3).detach()
    trans = trans.reshape(batch_size, 128, 1, 3).detach()

    gt_local_pose = gt_item['pose_aa'].reshape(batch_size, 128, 21 * 3).detach()
    gt_root_orient = gt_item['root_orient_aa'].reshape(batch_size, 128, 3).detach()
    gt_trans = gt_item['trans'].reshape(batch_size, 128, 1, 3).detach()

    p3ds, v3ds = get_p3ds(smpl, local_pose, root_orient, return_vert=True)
    p3ds = c2c(p3ds[:, :, :-2]).reshape(total_frm_num, 17, 3)
    v3ds = c2c(v3ds).reshape(total_frm_num, -1, 3)
    gt_p3ds, gt_v3ds = get_p3ds(smpl, gt_local_pose, gt_root_orient, return_vert=True)
    gt_p3ds = c2c(gt_p3ds[:, :, :-2]).reshape(total_frm_num, 17, 3)
    gt_v3ds = c2c(gt_v3ds).reshape(total_frm_num, -1, 3)

    pa_p3ds = np.zeros_like(p3ds)
    for i in range(gt_p3ds.shape[0]):
        pa_p3ds[i] = rigid_align(p3ds[i], gt_p3ds[i])

    pa_mpjpe = np.mean(np.sqrt(np.sum(((pa_p3ds - gt_p3ds) ** 2), axis=-1)), axis=-1)
    normalize_mpjpe = np.mean(np.sqrt(np.sum(((p3ds - gt_p3ds) ** 2), axis=-1)), axis=-1)

    ################### ACC ##################
    p3ds = p3ds.reshape(batch_size, 128, -1, 3)
    v = (p3ds[:, 1:, ...] - p3ds[:, :-1, ...]) * 1000  # (B, FRM-1, 17, 3)
    acc = (v[:, 1:, ...] - v[:, :-1, ...])  # (B, FRM-2, 17, 3)

    gt_p3ds = gt_p3ds.reshape(batch_size, 128, -1, 3)
    gt_v = (gt_p3ds[:, 1:, ...] - gt_p3ds[:, :-1, ...]) * 1000  # (B, FRM-1, 17, 3)
    gt_acc = (gt_v[:, 1:, ...] - gt_v[:, :-1, ...])  # (B, FRM-2, 17, 3)

    p3ds = p3ds + c2c(trans)
    gt_p3ds = gt_p3ds + c2c(gt_trans.reshape(-1, 1, 3))

    mpjpe = np.mean(np.sqrt(np.sum(((p3ds - gt_p3ds) ** 2), axis=-1)), axis=-1).reshape(batch_size * 128, -1)
    mpvpe = np.mean(np.sqrt(np.sum(((gt_v3ds - v3ds) ** 2), axis=-1)), axis=-1).reshape(batch_size * 128, -1)

    batch_acc = np.mean(np.linalg.norm(acc - gt_acc, axis=-1), axis=-1)
    # print('{:.2f} / {:.2f} / {:.2f} / {:.2f} / {:.2f} '.format(
    #     np.mean(mpjpe) * 1000, np.mean(normalize_mpjpe) * 1000, np.mean(pa_mpjpe) * 1000, np.mean(batch_acc), mpvpe.mean() * 1000))

    return mpjpe.sum(), normalize_mpjpe.sum(), pa_mpjpe.sum(), batch_acc.sum(), mpvpe.sum(), mpjpe.shape[0]
