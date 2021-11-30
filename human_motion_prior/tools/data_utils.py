import open3d
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.fftpack import dct
from human_motion_prior.tools.evaluation_tools import mpjpe_loss, get_p3ds
from human_motion_prior.train.motion_prior import MotionPrior, ContinousRotReprDecoder

c2c = lambda x: x.detach().cpu().numpy()
rot_decoder = ContinousRotReprDecoder()


def data_prepare(dorig, smpl, vposer_pt, block_size, frequency_num, batch_size=1, frame_num=128, noise=True, trans_convert=True, recomp_p3ds=False, device=torch.device('cuda')):
    dorig = {k: dorig[k].to(device) for k in ['pose', 'trans', 'p3ds']}  # (B, FRAME_NUM, 156)
    pose_aa = dorig['pose'][:, :, 3:63 + 3]
    vpose_code = vposer_pt.encode(pose_aa.reshape(batch_size * frame_num,
                                                  1, 21, 3))
    vpose_seq = vpose_code.mean  # (B * FRAME_NUM, latentD)
    vpose_seq = vpose_seq.reshape(batch_size, frame_num, -1)  # (B, FRAME_NUM * latentD)

    ###### PREPARE ROTATION AND TRANSLATION #######
    trans = dorig['trans'][:, :, :].to(device)
    trans = trans - trans[:, [0], :]
    if trans_convert:
        convert_mat = torch.tensor([[[1, 0, 0],
                                     [0, 0, -1],
                                     [0, 1, 0]]],
                                   dtype=torch.float32,
                                   device=device).repeat([batch_size, 1, 1])
        trans = torch.bmm(trans,
                          convert_mat)  # .reshape(batch_size * frame_num, -1)    # (B * F, 3)

    root_orient_aa = dorig['pose'][:, :, :3].to(
        device)  # .reshape(batch_size * frame_num, 1, 1, -1)    # (B * F, 1, 1 3)

    ###### FRONT DIRECTION ADJUST #######
    root_orient_matrot = MotionPrior.aa2matrot(root_orient_aa).reshape(batch_size, frame_num, 3, 3)
    r = R.from_matrix(root_orient_matrot[:, 0].cpu())
    std_euler = r.as_euler('zxy', degrees=True)
    std_euler_org = std_euler.copy()

    ############ NORMALIZED ##########
    std_euler[:, 2] = 0  # remove rotation around y-axis
    std_r = R.from_euler(seq='zxy', angles=std_euler, degrees=True)
    std_rmat = torch.tensor(std_r.as_matrix(), device=device, dtype=torch.float32)
    transform_mat = torch.bmm(std_rmat,
                              torch.transpose(root_orient_matrot[:, 0], 1, 2)).unsqueeze(1).repeat(
        [1, frame_num, 1, 1]).reshape(-1, 3, 3)

    # root_orient_matrot = root_orient_matrot.reshape(-1, 3, 3)
    root_orient_matrot_norm = torch.bmm(transform_mat, root_orient_matrot.reshape(-1, 3, 3))
    root_orient_aa_norm = MotionPrior.matrot2aa(root_orient_matrot_norm).reshape(batch_size, frame_num, 3)

    ############ NOISE !!!! ###########
    bs = std_euler_org.shape[0]
    if noise:
        std_euler_org[:, 2] = np.random.rand(bs) * 2 * 180 - 180.
    ##################################
    std_r = R.from_euler(seq='zxy', angles=std_euler_org, degrees=True)
    std_rmat = torch.tensor(std_r.as_matrix(), device=device, dtype=torch.float32)
    transform_mat = torch.bmm(std_rmat,
                              torch.transpose(root_orient_matrot[:, 0], 1, 2)).unsqueeze(1).repeat(
        [1, frame_num, 1, 1]).reshape(-1, 3, 3)

    # root_orient_matrot = root_orient_matrot.reshape(-1, 3, 3)
    root_orient_matrot_noise = torch.bmm(transform_mat.reshape(-1, 3, 3), root_orient_matrot.reshape(-1, 3, 3))
    root_orient_aa_noise = MotionPrior.matrot2aa(root_orient_matrot_noise).reshape(batch_size, frame_num, 3)
    convert_trans = torch.bmm(transform_mat.reshape(-1, 3, 3), trans.reshape(-1, 3, 1)).reshape(batch_size,
                                                                                                frame_num,
                                                                                                3)
    convert_trans = convert_trans - convert_trans[:, [0], :]

    ####### RESHAPE AND VELOCITY #######
    velocity = torch.zeros_like(convert_trans, device=device, dtype=torch.float32)
    velocity[:, 1:, :] = convert_trans[:, 1:, :] - convert_trans[:, :-1, :]
    trans = convert_trans.reshape(batch_size, frame_num, -1)  # (B * F, 3)

    root_orient_matrot_noise = root_orient_matrot_noise.reshape(batch_size, frame_num, 3,
                                                      3)  # (B, FRAME, 3, 3)
    orient_matrot_t = root_orient_matrot_noise[:, 1:, :, :].reshape(-1, 3, 3)  # (B*(FRAME-1), 3, 3)
    orient_matrot_t_1 = root_orient_matrot_noise[:, :-1, :, :].reshape(-1, 3, 3)
    rvelocity = torch.eye(3, device=device, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(
        [batch_size, frame_num, 1, 1])  # (B, F, 3, 3)
    rvelocity[:, 1:, :, :] = torch.bmm(orient_matrot_t,
                                       orient_matrot_t_1.transpose(1, 2)).reshape(batch_size,
                                                                                  frame_num - 1,
                                                                                  3, 3)
    rvelocity[:, 0, :, :] = torch.bmm(root_orient_matrot_noise[:, 0, :, :], rvelocity[:, 0, :, :].transpose(1, 2))
    rvelocity_aa = MotionPrior.matrot2aa(rvelocity.reshape(-1, 3, 3)).reshape(batch_size, frame_num, 3)
    rvelocity_cont = rvelocity.reshape(-1, 3, 3)[:, :3, :2].contiguous().reshape(batch_size,
                                                                                 frame_num, 6)

    root_orient_cont_noise = MotionPrior.aa2matrot_cont(root_orient_aa_noise)
    root_orient_cont_noise = root_orient_cont_noise.reshape(batch_size, frame_num, -1)  # # (B * F, 6)

    # (B, FRAME_NUM * (latentD + 3 + 6))
    # latent_code_sample = torch.cat([latent_code_sample, root_orient_rotmat, trans], dim=-1).reshape(
    #     batch_size, -1)
    if recomp_p3ds:
        dorig['p3ds'] = get_p3ds(smpl, pose_aa, root_orient_aa_noise, trans)
        dorig['p3ds_norm'] = get_p3ds(smpl, pose_aa, root_orient_aa_norm, trans)

    p3ds = dorig['p3ds'].float()
    p3ds = p3ds[:, :, :-2, :] - p3ds[:, :, [0], :]

    p3ds_norm = dorig['p3ds_norm'].float()
    p3ds_norm = p3ds_norm[:, :, :-2, :] - p3ds_norm[:, :, [0], :]

    p3ds_rot = np.zeros_like(std_euler_org)
    p3ds_rot[:, 2] = std_euler_org[:, 2]
    p3ds_rot = R.from_euler(seq='zxy', angles=p3ds_rot, degrees=True)
    p3ds_rot = torch.tensor(p3ds_rot.as_matrix(), device=device, dtype=torch.float32)

    p3ds_rot = p3ds_rot.reshape(batch_size, 1, 1, 3, 3).repeat(
        [1, frame_num, p3ds_norm.shape[2], 1, 1])
    p3ds_rot = p3ds_rot.reshape(batch_size * frame_num * p3ds_norm.shape[2], 3, 3)
    p3ds_norm = torch.bmm(p3ds_rot, p3ds_norm.reshape(-1, 3, 1))

    basis = torch.tensor(dct(np.eye(frame_num), norm='ortho', axis=0)[:20],
                         device=device).float().unsqueeze(0).repeat([batch_size, 1, 1])
    basis_block = torch.tensor(dct(np.eye(block_size), norm='ortho', axis=0)[:20],
                               device=device).float().unsqueeze(0).repeat(
        [batch_size * frame_num, 1, 1])

    p3ds_flatten = p3ds.reshape(batch_size, frame_num, -1)
    frequency_dct = torch.bmm(basis, p3ds_flatten)
    frequency_dct = frequency_dct.reshape(batch_size, -1)

    p3ds_pad = torch.cat(
        [p3ds_flatten[:, [0], :].repeat([1, 2, 1]), p3ds_flatten, p3ds_flatten[:, [-1], :].repeat([1, 2, 1])], dim=1)
    p3ds_block = torch.zeros((batch_size, frame_num, block_size, 51), dtype=p3ds_pad.dtype,
                             device=p3ds_pad.device)
    for i in range(frame_num):
        p3ds_block[:, i, :, :] = p3ds_pad[:, i:i + block_size, :]
    p3ds_block = p3ds_block.reshape(batch_size * frame_num, block_size, -1)
    frequency_block = torch.bmm(basis_block, p3ds_block)
    frequency_block = frequency_block.reshape(batch_size, frame_num * block_size * 51)

    item = {'pose_aa': pose_aa, 'vpose_seq': vpose_seq,
            'root_orient_aa': root_orient_aa_noise, 'root_orient_aa_norm': root_orient_aa_norm,
            'root_orient_cont': root_orient_cont_noise, 'trans': trans,
            'velocity': velocity, 'rvelocity_cont': rvelocity_cont, 'rvelocity_aa': rvelocity_aa,
            'p3ds': p3ds.reshape(batch_size, frame_num, -1), 'p3ds_norm': p3ds_norm.reshape(batch_size, frame_num, -1), 'frequency': frequency_dct,
            'frequency_block': frequency_block}
    return item


