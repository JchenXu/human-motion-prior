import os

import open3d
import torch
import numpy as np
import tqdm
import pickle

from configer import Configer
from scipy.spatial.transform import Rotation as R
from scipy.fftpack import dct
from human_motion_prior.tools.model_loader import load_vposer
from human_motion_prior.tools.evaluation_tools import mpjpe_loss, get_p3ds
from human_motion_prior.tools.data_utils import data_prepare
from human_motion_prior.train.motion_prior import MotionPrior, ContinousRotReprDecoder

c2c = lambda x: x.detach().cpu().numpy()
rot_decoder = ContinousRotReprDecoder()


def loader_3dpw(data_dir = '/data/3dpw/sequenceFiles', split='test'):
    dtype = torch.float32
    device = torch.device('cuda')
    item_list = []
    SKIP = 128

    for test_file in os.listdir(os.path.join(data_dir, split)):
        if 'pkl' not in test_file:
            continue

        pkl_file = os.path.join(data_dir, split, test_file)
        motion_data = pickle.load(open(pkl_file, 'rb'), encoding='latin1')
        for PERSON_ID in range(len(motion_data['poses'])):
            for i in range(motion_data['poses'][PERSON_ID].shape[0] // 128 + 1):
                if SKIP * i +128 > motion_data['poses'][PERSON_ID].shape[0]:
                    start_idx = -128
                    end_idx = motion_data['poses'][PERSON_ID].shape[0]
                else:
                    start_idx = SKIP * i
                    end_idx = SKIP * i + 128

                theta = motion_data['poses'][PERSON_ID][start_idx:end_idx]
                root_orient = torch.tensor(theta[:, :3], dtype=dtype, device=device)
                pose_aa = torch.tensor(theta[:, 3:63+3], dtype=dtype, device=device)      # remove hand rot
                trans = torch.tensor(motion_data['trans'][PERSON_ID][start_idx:end_idx], dtype=dtype, device=device)
                # trans = trans - trans[[0]]
                cam_poses = torch.tensor(motion_data['cam_poses'][start_idx:end_idx], dtype=dtype, device=device)
                valid_flag = torch.tensor(motion_data['campose_valid'][PERSON_ID][start_idx:end_idx], dtype=dtype, device=device)

                convert_mat = torch.tensor([[[1, 0, 0],
                                             [0, -1, 0],
                                             [0, 0, -1]]], dtype=dtype, device=device).repeat([128, 1, 1])
                new_root_orient = torch.bmm(cam_poses[:, :3, :3].cpu(), MotionPrior.aa2matrot(root_orient[None, ...].cpu()).reshape(-1, 3, 3))
                new_root_orient = torch.bmm(convert_mat.cpu(), new_root_orient)
                new_root_orient = MotionPrior.matrot2aa(new_root_orient).reshape(-1, 3).cuda()
                new_trans = torch.bmm(cam_poses[:, :3, :3].cpu(), trans[..., None].cpu()).reshape(-1, 3).cuda()
                new_trans += cam_poses[:, :3, -1]
                new_trans = torch.bmm(convert_mat.cpu(), new_trans[..., None].cpu()).reshape(-1, 3).cuda()
                new_trans = new_trans - new_trans[[0]]


                item = {'pose': torch.cat([new_root_orient, pose_aa], dim=1)[None, ...],
                        'trans': new_trans[None, ...],
                        'beta': torch.tensor(motion_data['betas'][PERSON_ID][:10])[None, None, ...].repeat([1, 128, 1]).float(),
                        'p3ds': torch.zeros((1, 128, 19, 3)),
                        'valid_flag': valid_flag[None, ...].int(),
                        'seq_name': test_file+'_'+str(PERSON_ID)}
                item_list.append(item)

    return item_list


if __name__ == '__main__':
    device = torch.device('cuda')
    FRAME_NUM = 128
    batch_size = 1

    from human_motion_prior.body_model.smpl_model.smpl import SMPL 
    from human_motion_prior.body_model.smpl_model import config

    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=128 * 1,
                create_transl=False).eval().cuda()

    block_size = 5
    frequency_num = 20

    ####################### LOAD VPOSER #######################
    expr_dir = '../models/pre_trained/vposer_v1_0'
    vposer_pt, _ = load_vposer(expr_dir, vp_model='snapshot')
    vposer_pt.to(device)

    ####################### LOAD PRIOR #######################
    motion_prior_path = None # path_to_ckpt
    ini_path = None # path_to_config file .ini
    ps = Configer(default_ps_fname=ini_path)  # This is the default configuration
    motion_prior = MotionPrior(num_neurons=ps.num_neurons, latentD=ps.latentD, latentD_t=ps.latentD_t,
                               dense_freq=ps.dense_freq, block_size=block_size, frequency_num=frequency_num, 
                               frame_num=FRAME_NUM, use_cont_repr=ps.use_cont_repr)
    state_dict = torch.load(motion_prior_path, map_location='cpu')
    new_state_dict = {k[7:]: v for k, v in state_dict.items()}
    motion_prior.load_state_dict(new_state_dict)
    motion_prior.eval().to(device)

    loss_list = 0
    pa_loss_list = 0
    acc_loss_list = 0
    mpvpe_list = 0
    total_num = 0

    org_items = loader_3dpw()
    for org_item in tqdm.tqdm(org_items):
        item = data_prepare(org_item, smpl=smpl, vposer_pt=vposer_pt, block_size=block_size, frequency_num=frequency_num, batch_size=org_item['trans'].shape[0], noise=False, trans_convert=True, recomp_p3ds=True)     # for 3dpw
        input_trans = item['trans'].reshape(-1, 3)

        ######################## ENCODE #############################
        Pin = torch.cat([item['pose_aa'], item['vpose_seq'], item['root_orient_aa'], item['trans'], item['velocity'], item['rvelocity_aa'], item['p3ds']], dim=-1)
        init_z = motion_prior.encode(Pin, item['frequency'], item['frequency_block'])
        init_z = init_z.mean


        ######################## DECODE #############################
        mpose = motion_prior.decode(init_z)

        ######################## DISENTANGLE ##########################
        pose_body = vposer_pt.decode(mpose[:, :, :32].reshape(batch_size * FRAME_NUM, -1),
                                     output_type='aa').reshape(batch_size * FRAME_NUM, -1)
        root_orient = mpose[:, :, 32:32 + 3].reshape(batch_size * FRAME_NUM, -1).contiguous()

        ######################## MPJPE LOSS ###########################
        loss, _, pa_loss, acc, mpvpe, num = mpjpe_loss(smpl=smpl, local_pose=pose_body, root_orient=root_orient, trans=input_trans, gt_item=item, add_hand=True)

        loss_list += loss
        pa_loss_list += pa_loss
        acc_loss_list += acc
        mpvpe_list += mpvpe
        total_num += num

    print('\n')
    print('##' * 20)
    print('mpjpe: {:.2f} / pa-mpjpe: {:.2f} / mpvpe: {:.2f} / acc_err: {:.2f}'.format(loss_list / total_num * 1000, pa_loss_list / total_num * 1000, mpvpe_list / total_num * 1000, acc_loss_list / total_num))
    print('##' * 20)
