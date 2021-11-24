# This code is developed based on VPoser <https://github.com/nghorbani/human_body_prior>

import os, shutil
from datetime import datetime

import numpy as np

import torch
from torch import distributed as dist
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm

import torchgeometry as tgm

from configer import Configer

from human_motion_prior.tools.omni_tools import copy2cpu as c2c
from human_motion_prior.tools.omni_tools import log2file, makepath
from human_motion_prior.tools.training_tools import all_reduce_tensor, get_p3ds
from human_motion_prior.body_model.body_model import BodyModel
from human_motion_prior.tools.model_loader import load_vposer
from scipy.spatial.transform import Rotation as R
from scipy.fftpack import dct


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class MotionPrior(nn.Module):
    def __init__(self, num_neurons, latentD, latentD_t, frame_num, dense_freq, block_size, frequency_num, use_cont_repr=True):
        super(MotionPrior, self).__init__()
        self.latentD_t = latentD_t 
        self.latentD = latentD
        self.use_cont_repr = use_cont_repr
        self.dense_freq = dense_freq

        self.frame_num = frame_num
        self.joint_num = 17
        self.input_channel = 158

        if self.dense_freq:
            self.dct_att = nn.Linear(frame_num * block_size * self.joint_num * 3, frame_num)
        else:
            self.dct_att = nn.Linear(frame_num * self.joint_num * 3, frame_num // block_size)

        ####### STAGE 0 ########
        self.bodyprior_enc_fc0_1 = weight_norm(nn.Conv1d(self.input_channel, 64, kernel_size=3, stride=2, padding=1))
        self.bodyprior_enc_bn0_1 = nn.BatchNorm1d(64)
        
        self.bodyprior_enc_fc0_2 = weight_norm(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
        self.bodyprior_enc_bn0_2 = nn.BatchNorm1d(64)

        self.bodyprior_enc_fc0_3 = weight_norm(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
        self.bodyprior_enc_bn0_3 = nn.BatchNorm1d(64)

        ####### STAGE 1 ########
        self.pool1_1 = weight_norm(nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1))
        self.bodyprior_enc_fc1_1 = weight_norm(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
        self.bodyprior_enc_bn1_1 = nn.BatchNorm1d(64)

        self.bodyprior_enc_fc1_2 = weight_norm(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
        self.bodyprior_enc_bn1_2 = nn.BatchNorm1d(64)

        ####### STAGE 2 ########
        self.pool2_1 = weight_norm(nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1))
        self.bodyprior_enc_fc2_1 = weight_norm(nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1))
        self.bodyprior_enc_bn2_1 = nn.BatchNorm1d(128)

        self.bodyprior_enc_fc2_2 = weight_norm(nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1))
        self.bodyprior_enc_bn2_2 = nn.BatchNorm1d(128)

        last_layer_dim = 128 * 16

        self.bodyprior_enc_dct = weight_norm(nn.Linear(frequency_num * self.joint_num * 3, 512))

        self.bodyprior_enc_mu_local = nn.Linear(last_layer_dim * 4 + 512, self.latentD)
        self.bodyprior_enc_logvar_local = nn.Linear(last_layer_dim * 4 + 512, self.latentD)
        self.bodyprior_enc_mu_global = nn.Linear(last_layer_dim * 4, self.latentD_t)
        self.bodyprior_enc_logvar_global = nn.Linear(last_layer_dim * 4, self.latentD_t)
        self.dropout = nn.Dropout(p=.1, inplace=False)

        self.bodyprior_dec_fc0 = nn.Linear((latentD), num_neurons)
        self.bodyprior_dec_fc1 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)

        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()

        self.bodyprior_dec_pose = nn.Linear(num_neurons + latentD, frame_num * 32)
        self.bodyprior_dec_orient = nn.Linear(num_neurons + latentD, frame_num * 1 * 6)
        self.bodyprior_dec_trans = nn.Linear(num_neurons + latentD_t, frame_num * 3)

    def swish(self, Xin):
        return Xin / (1 + torch.exp(-Xin))

    def encode(self, Pin, frequency=None, frequency_block=None):
        Pin = Pin.view(Pin.size(0), self.frame_num, -1)      # (B, frame_num, 32+6+3)
        Pin = torch.transpose(Pin, 1, 2)    # (B, 32+6+3, frame_num, 1)

        ######## STAGE 0 ########
        if self.dense_freq:
            att_map = self.dct_att(frequency_block)
        else:
            att_map = self.dct_att(frequency_block).unsqueeze(-1).repeat([1, 1, 8]).reshape(-1, 128)
        Pin = Pin * att_map.unsqueeze(1)
        Xout = self.swish(self.bodyprior_enc_fc0_1(Pin))
        # att_map = self.dct_att(frequency_block)
        # Xout = Xout * att_map.unsqueeze(1)
        Xout_res = self.swish(self.bodyprior_enc_fc0_2(Xout))
        Xout_res = self.swish(self.bodyprior_enc_fc0_3(Xout_res))
        Xout = Xout + Xout_res
        Xout_local = Xout.view(Pin.size(0), -1)

        ######## STAGE 1 ########
        Xout = self.swish(self.pool1_1(Xout))
        Xout_res = self.swish(self.bodyprior_enc_fc1_1(Xout))
        Xout_res = self.swish(self.bodyprior_enc_fc1_2(Xout_res))
        Xout = Xout + Xout_res
        Xout_mid = Xout.view(Pin.size(0), -1)

        ######## STAGE 2 ########
        Xout = self.swish(self.pool2_1(Xout))
        Xout_res = self.swish(self.bodyprior_enc_fc2_1(Xout))
        Xout_res = self.swish(self.bodyprior_enc_fc2_2(Xout_res))
        Xout = Xout + Xout_res

        Xout = Xout.view(Pin.size(0), -1)
        Xout = torch.cat([Xout, Xout_local, Xout_mid], dim=-1)

        frequency = self.swish(self.bodyprior_enc_dct(frequency))
        Xout = torch.cat([Xout, frequency], dim=-1)

        Xout = self.dropout(Xout)

        dis_local = torch.distributions.normal.Normal(self.bodyprior_enc_mu_local(Xout),
                                                      F.softplus(self.bodyprior_enc_logvar_local(Xout)))

        return dis_local

    def decode(self, Zin_local, output_type='matrot'):
        assert output_type in ['matrot', 'aa']
        Zin = Zin_local
        batch_size = Zin.shape[0]

        Xout = self.swish(self.bodyprior_dec_fc0(Zin))
        Xout_d = self.swish(self.bodyprior_dec_fc1(Xout))
        Xout = Xout + Xout_d

        Xout_d = self.swish(self.bodyprior_dec_fc2(Xout))
        Xout = Xout + Xout_d

        Xout = self.dropout(Xout)
        Xout_pose = self.bodyprior_dec_pose(torch.cat([Xout, Zin_local], dim=-1)).view([batch_size, self.frame_num, 32])
        Xout_orient = self.rot_decoder(self.bodyprior_dec_orient(torch.cat([Xout, Zin_local], dim=-1)).view([-1, 6]))

        Xout = torch.cat([Xout_pose,
                          MotionPrior.matrot2aa(Xout_orient).view([batch_size, self.frame_num, 1 * 3])], dim=-1)

        return Xout

    def forward(self, item, input_type='matrot', output_type='matrot'):
        assert output_type in ['matrot', 'aa']
        Pin = torch.cat([item['pose_aa'], item['vpose_seq'], item['root_orient_aa'], item['trans'], item['velocity'], item['rvelocity_aa'], item['p3ds']], dim=-1)
        q_z = self.encode(Pin, item['frequency'], item['frequency_block'])
        q_z_sample = q_z.rsample()
        Prec = self.decode(q_z_sample)

        results = {'mean':q_z.mean, 'std':q_z.scale}
        if output_type == 'aa': results['pose'] = Prec
        else: results['pose_matrot'] = Prec
        return results

    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        batch_size = pose_matrot.size(0)
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
        return pose_body_matrot

    @staticmethod
    def aa2matrot_cont(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :2].contiguous().view(
            batch_size, 1, -1, 6)
        return pose_body_matrot

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

class MotionPriorTrainer:

    def __init__(self, work_dir, ps):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_args()

        self.world_size = int(os.environ['WORLD_SIZE'])
        self.local_rank = args.local_rank

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method="tcp://localhost:19308", rank=self.local_rank,
                                             world_size=self.world_size)
        if self.local_rank == 0:
            ps.work_dir = makepath(work_dir, isfile=False)

        synchronize()

        from tensorboardX import SummaryWriter
        from human_motion_prior.data.seq_dataloader import AMASSSeqDataset

        self.pt_dtype = torch.float64 if ps.fp_precision == '64' else torch.float32

        torch.manual_seed(ps.seed)

        logger = log2file(os.path.join(work_dir, '%s.log' % ps.expr_code))

        summary_logdir = os.path.join(work_dir, 'summaries')
        if self.local_rank == 0:
            self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)

        shutil.copy2(os.path.realpath(__file__), work_dir)

        use_cuda = torch.cuda.is_available()
        if use_cuda: torch.cuda.empty_cache()
        self.comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger('%d CUDAs available!' % torch.cuda.device_count())

        gpu_brand= torch.cuda.get_device_name(ps.cuda_id) if use_cuda else None
        logger('Training with %s [%s]' % (self.comp_device,gpu_brand)  if use_cuda else 'Training on CPU!!!')
        logger('Base dataset_dir is %s'%ps.dataset_dir)

        self.bs_per_gpu = ps.batch_size // self.world_size
        ps.batch_size = self.bs_per_gpu
        kwargs = {'num_workers': ps.n_workers}

        ############ SETTING FOR FREQUENCY GUIDANCE #############
        self.block_size = 5 if ps.dense_freq else 8
        self.frequency_num = 20
        self.basis = torch.tensor(dct(np.eye(ps.frame_num), norm='ortho', axis=0)[:self.frequency_num],
                                  device=self.comp_device).float().unsqueeze(0).repeat([ps.batch_size, 1, 1])
        if ps.dense_freq:
            self.basis_block = torch.tensor(dct(np.eye(self.block_size), norm='ortho', axis=0)[:self.frequency_num],
                                            device=self.comp_device).float().unsqueeze(0).repeat(
                [ps.batch_size * ps.frame_num, 1, 1])
        else:
            self.basis_block = torch.tensor(dct(np.eye(self.block_size), norm='ortho', axis=0)[:self.frequency_num],
                                            device=self.comp_device).float().unsqueeze(0).repeat(
                [ps.batch_size * ps.frame_num // self.block_size, 1, 1])

        ds_train = AMASSSeqDataset(data_dir=ps.dataset_dir, is_train=True)
        self.sampler_train = torch.utils.data.distributed.DistributedSampler(ds_train)
        self.ds_train = DataLoader(ds_train, batch_size=self.bs_per_gpu, shuffle=False,
                                   drop_last=True, pin_memory=True, sampler=self.sampler_train, **kwargs)

        ds_val = AMASSSeqDataset(data_dir=ps.dataset_dir, is_train=False)
        self.sampler_val = torch.utils.data.distributed.DistributedSampler(ds_val)
        self.ds_val = DataLoader(ds_val, batch_size=self.bs_per_gpu, shuffle=False,
                                 drop_last=True, pin_memory=True, sampler=self.sampler_val, **kwargs)

        self.motion_prior = MotionPrior(num_neurons=ps.num_neurons, latentD=ps.latentD, latentD_t=ps.latentD_t,
                                        dense_freq=ps.dense_freq, block_size=self.block_size, frequency_num=self.frequency_num,
                                        frame_num=ps.frame_num, use_cont_repr=ps.use_cont_repr)
        self.motion_prior.to(self.comp_device)

        if ps.use_multigpu :
            # torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.motion_prior = nn.parallel.DistributedDataParallel(self.motion_prior,
                                                                    find_unused_parameters=True,
                                                                    device_ids=[args.local_rank],
                                                                    output_device=args.local_rank)

        varlist = [var[1] for var in self.motion_prior.named_parameters()]

        params_count = sum(p.numel() for p in varlist if p.requires_grad)
        logger('Total Trainable Parameters Count is %2.2f M.' % ((params_count) * 1e-6))

        self.optimizer = optim.Adam(varlist, lr=ps.base_lr, weight_decay=ps.reg_coef)

        self.logger = logger
        self.best_loss_total = np.inf
        self.try_num = ps.try_num
        self.epochs_completed = 0
        self.ps = ps

        if ps.best_model_fname is not None:
            if isinstance(self.motion_prior, torch.nn.DataParallel):
                self.motion_prior.module.load_state_dict(
                    torch.load(ps.best_model_fname, map_location=self.comp_device))
            else:
                self.motion_prior.load_state_dict(torch.load(ps.best_model_fname, map_location=self.comp_device))

            logger('Restored model from %s' % ps.best_model_fname)

        self.bm = BodyModel(self.ps.bm_path, batch_size=self.ps.batch_size * self.ps.frame_num, use_posedirs=True).to(self.comp_device)

        self.pretrained_vposer, _ = load_vposer(ps.pretrained_vposer, vp_model='snapshot')
        self.pretrained_vposer.to(self.comp_device)
        self.pretrained_vposer.eval()

        self.convert_mat = torch.tensor([[[1, 0, 0],
                                          [0, 0, -1],
                                          [0, 1, 0]]],
                                        dtype=torch.float32,
                                        device=self.comp_device).repeat([self.ps.batch_size, 1, 1])

    def data_prepare(self, dorig):
        dorig = {k: dorig[k].to(self.comp_device) for k in ['pose', 'trans', 'p3ds']}  # (B, FRAME_NUM, 156)
        pose_aa = dorig['pose'][:, :, 3:63 + 3]
        vpose_code = self.pretrained_vposer.encode(pose_aa.reshape(self.ps.batch_size * self.ps.frame_num,
                                                                   1, 21, 3))
        vpose_seq = vpose_code.mean  # (B * FRAME_NUM, latentD)
        vpose_seq = vpose_seq.reshape(self.ps.batch_size, self.ps.frame_num, -1)     #   (B, FRAME_NUM * latentD)

        ###### PREPARE ROTATION AND TRANSLATION #######
        trans = dorig['trans'][:, :, :].to(self.comp_device)
        trans = trans - trans[:, [0], :]
        trans = torch.bmm(trans, self.convert_mat)  # .reshape(self.ps.batch_size * self.ps.frame_num, -1)    # (B * F, 3)

        root_orient_aa = dorig['pose'][:, :, :3].to(
            self.comp_device)  # .reshape(self.ps.batch_size * self.ps.frame_num, 1, 1, -1)    # (B * F, 1, 1 3)

        ###### FRONT DIRECTION ADJUST #######
        root_orient_matrot = MotionPrior.aa2matrot(root_orient_aa).reshape(self.ps.batch_size, self.ps.frame_num, 3, 3)
        r = R.from_matrix(root_orient_matrot[:, 0].cpu())
        std_euler = r.as_euler('zxy', degrees=True)
        std_euler_org = std_euler.copy()

        ############ NORMALIZED ##########
        std_euler[:, 2] = 0     # remove rotation around y-axis
        std_r = R.from_euler(seq='zxy', angles=std_euler, degrees=True)
        std_rmat = torch.tensor(std_r.as_matrix(), device=self.comp_device, dtype=torch.float32)
        transform_mat = torch.bmm(std_rmat,
                                  torch.transpose(root_orient_matrot[:, 0], 1, 2)).unsqueeze(1).repeat(
            [1, self.ps.frame_num, 1, 1]).reshape(-1, 3, 3)

        # root_orient_matrot = root_orient_matrot.reshape(-1, 3, 3)
        root_orient_matrot_norm = torch.bmm(transform_mat, root_orient_matrot.reshape(-1, 3, 3))
        root_orient_aa_norm = MotionPrior.matrot2aa(root_orient_matrot_norm).reshape(self.ps.batch_size, self.ps.frame_num, 3)

        ############ NOISE !!!! ###########
        bs = std_euler_org.shape[0]
        std_euler_org[:, 2] = np.random.rand(bs) * 360 - 180
        ##################################
        std_r = R.from_euler(seq='zxy', angles=std_euler_org, degrees=True)
        std_rmat = torch.tensor(std_r.as_matrix(), device=self.comp_device, dtype=torch.float32)
        transform_mat = torch.bmm(std_rmat,
                                  torch.transpose(root_orient_matrot[:, 0], 1, 2)).unsqueeze(1).repeat(
            [1, self.ps.frame_num, 1, 1])

        # root_orient_matrot = root_orient_matrot.reshape(-1, 3, 3)
        root_orient_matrot_ = torch.bmm(transform_mat.reshape(-1, 3, 3), root_orient_matrot.reshape(-1, 3, 3))
        root_orient_aa_ = MotionPrior.matrot2aa(root_orient_matrot_).reshape(self.ps.batch_size, self.ps.frame_num, 3)
        convert_trans = torch.bmm(transform_mat.reshape(-1, 3, 3), trans.reshape(-1, 3, 1)).reshape(self.ps.batch_size,
                                                                                  self.ps.frame_num,
                                                                                  3)
        convert_trans = convert_trans - convert_trans[:, [0], :]

        ####### RESHAPE AND VELOCITY #######
        velocity = torch.zeros_like(convert_trans, device=self.comp_device, dtype=torch.float32)
        velocity[:, 1:, :] = convert_trans[:, 1:, :] - convert_trans[:, :-1, :]
        trans = convert_trans.reshape(self.ps.batch_size, self.ps.frame_num, -1)  # (B * F, 3)

        root_orient_matrot_ = root_orient_matrot_.reshape(self.ps.batch_size, self.ps.frame_num, 3, 3)  # (B, FRAME, 3, 3)
        orient_matrot_t = root_orient_matrot_[:, 1:, :, :].reshape(-1, 3, 3)  # (B*(FRAME-1), 3, 3)
        orient_matrot_t_1 = root_orient_matrot_[:, :-1, :, :].reshape(-1, 3, 3)
        rvelocity = torch.eye(3, device=self.comp_device, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(
            [self.ps.batch_size, self.ps.frame_num, 1, 1])  # (B, F, 3, 3)
        rvelocity[:, 1:, :, :] = torch.bmm(orient_matrot_t,
                                           orient_matrot_t_1.transpose(1, 2)).reshape(self.ps.batch_size,
                                                                                      self.ps.frame_num - 1,
                                                                                      3, 3)
        rvelocity[:, 0, :, :] = torch.bmm(root_orient_matrot_[:, 0, :, :], rvelocity[:, 0, :, :].transpose(1, 2))
        rvelocity_aa = MotionPrior.matrot2aa(rvelocity.reshape(-1, 3, 3)).reshape(self.ps.batch_size, self.ps.frame_num, 3)
        rvelocity_cont = rvelocity.reshape(-1, 3, 3)[:, :3, :2].contiguous().reshape(self.ps.batch_size,
                                                                                     self.ps.frame_num, 6)

        root_orient_cont_ = MotionPrior.aa2matrot_cont(root_orient_aa_)
        root_orient_cont_ = root_orient_cont_.reshape(self.ps.batch_size, self.ps.frame_num, -1)  # # (B * F, 6)

        # dorig['p3ds'] = get_p3ds(pose_aa, root_orient_aa_, trans, device=self.comp_device)
        # dorig['p3ds_norm'] = get_p3ds(pose_aa, root_orient_aa_norm, trans, device=self.comp_device)

        p3ds = dorig['p3ds'].float()
        p3ds = p3ds[:, :, :-2, :] - p3ds[:, :, [0], :]

        p3ds_rot = np.zeros_like(std_euler_org)
        p3ds_rot[:, 2] = std_euler_org[:, 2]
        p3ds_rot = R.from_euler(seq='zxy', angles=p3ds_rot, degrees=True)
        p3ds_rot = torch.tensor(p3ds_rot.as_matrix(), device=self.comp_device, dtype=torch.float32)
        
        p3ds_rot = p3ds_rot.reshape(self.ps.batch_size, 1, 1, 3, 3).repeat([1, self.ps.frame_num, p3ds.shape[2], 1, 1])
        p3ds_rot = p3ds_rot.reshape(self.ps.batch_size * self.ps.frame_num * p3ds.shape[2], 3, 3)
        p3ds = torch.bmm(p3ds_rot, p3ds.reshape(-1, 3, 1))

        p3ds_flatten = p3ds.reshape(self.ps.batch_size, self.ps.frame_num, -1)
        frequency_dct = torch.bmm(self.basis, p3ds_flatten)
        frequency_dct = frequency_dct.reshape(self.ps.batch_size, -1)

        if self.ps.dense_freq:
            ################  DENSE SEG-FREQUENCY GUIDANCE #################
            p3ds_pad = torch.cat([p3ds_flatten[:, [0], :].repeat([1, 2, 1]), p3ds_flatten, p3ds_flatten[:, [-1], :].repeat([1, 2, 1])], dim=1)
            p3ds_block = torch.zeros((self.ps.batch_size, self.ps.frame_num, self.block_size, 51), dtype=p3ds_pad.dtype, device=p3ds_pad.device)
            for i in range(self.ps.frame_num):
                p3ds_block[:, i, :, :] = p3ds_pad[:, i:i+self.block_size, :]
            p3ds_block = p3ds_block.reshape(self.ps.batch_size * self.ps.frame_num, self.block_size, -1)
            frequency_block = torch.bmm(self.basis_block, p3ds_block)
            frequency_block = frequency_block.reshape(self.ps.batch_size, self.ps.frame_num * self.block_size * 51)
        else:
            ################  SEG-FREQUENCY GUIDANCE NO OVERLAP #################
            p3ds_pad = torch.cat([p3ds_flatten], dim=1)
            p3ds_block = p3ds_pad.reshape(self.ps.batch_size * self.ps.frame_num // self.block_size, self.block_size, -1)
            frequency_block = torch.bmm(self.basis_block, p3ds_block)
            frequency_block = frequency_block.reshape(self.ps.batch_size, self.ps.frame_num * 51)

        item = {'pose_aa': pose_aa, 'vpose_seq': vpose_seq,
                'root_orient_aa': root_orient_aa_, 'root_orient_aa_norm': root_orient_aa_norm, 'root_orient_cont': root_orient_cont_, 'trans': trans,
                'velocity': velocity, 'rvelocity_cont': rvelocity_cont, 'rvelocity_aa': rvelocity_aa, 
                'p3ds': p3ds.reshape(self.ps.batch_size, self.ps.frame_num, -1), 'frequency': frequency_dct, 'frequency_block': frequency_block}
        return item

    def train(self):
        self.motion_prior.train()
        save_every_it = len(self.ds_train) / self.ps.log_every_epoch
        train_loss_dict = {}
        for it, dorig in enumerate(self.ds_train):
            item = self.data_prepare(dorig)
            self.optimizer.zero_grad()
            drec = self.motion_prior(item, output_type='aa')
            loss_total, cur_loss_dict = self.compute_loss(item, drec)
            loss_total.backward()
            self.optimizer.step()

            for k, v in cur_loss_dict.items():
                cur_loss_dict[k] = all_reduce_tensor(v, world_size=self.world_size)

            if self.local_rank == 0:
                train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}
                if it % (save_every_it + 1) == 0:
                    cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                    train_msg = MotionPriorTrainer.creat_loss_message(cur_train_loss_dict, expr_code=self.ps.expr_code,
                                                                 epoch_num=self.epochs_completed, it=it,
                                                                 try_num=self.try_num, mode='train')

                    self.logger(train_msg)
                    self.swriter.add_histogram('q_z_sample', c2c(drec['mean']), it)

        if self.local_rank == 0:
            train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
            return train_loss_dict
        else:
            return None

    def evaluate(self):
        self.motion_prior.eval()
        eval_loss_dict = {}
        data = self.ds_val
        with torch.no_grad():
            for dorig in data:
                dorig = {k: dorig[k].to(self.comp_device) for k in dorig.keys()}
                item = self.data_prepare(dorig)
                drec = self.motion_prior(item, output_type='aa')
                _, cur_loss_dict = self.compute_loss(item, drec)

                for k, v in cur_loss_dict.items():
                    cur_loss_dict[k] = all_reduce_tensor(v, world_size=self.world_size)

                if self.local_rank == 0:
                    eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}

        if self.local_rank == 0:
            eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
            return eval_loss_dict
        else:
            return None

    def compute_loss(self, item, drec):
        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])

        pose_orig = item['pose_aa'].view(self.ps.batch_size * self.ps.frame_num, -1)
        root_orig = item['root_orient_aa_norm'].view(self.ps.batch_size * self.ps.frame_num, -1)

        loss_vpose = (drec['pose'][:, :, :32].reshape(self.ps.batch_size * self.ps.frame_num, -1) ** 2).mean() * self.ps.vpose_coef 

        pose_rec = self.pretrained_vposer.decode(drec['pose'][:, :, :32].reshape(self.ps.batch_size * self.ps.frame_num, -1),
                                                 output_type='aa').reshape(self.ps.batch_size * self.ps.frame_num, -1)
        root_rec = drec['pose'][:, :, 32:32+3].reshape(self.ps.batch_size * self.ps.frame_num, -1)

        device = pose_orig.device
        dtype = pose_orig.dtype

        MESH_SCALER = 1000

        # Reconstruction loss - L1 on the output mesh
        mesh_orig = self.bm(pose_body=pose_orig, root_orient=root_orig).v*MESH_SCALER
        mesh_rec = self.bm(pose_body=pose_rec, root_orient=root_rec).v*MESH_SCALER
        loss_mesh_rec = (1. - self.ps.kl_coef) * torch.mean(torch.abs(mesh_orig - mesh_rec))
        
        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.ps.batch_size, (self.ps.latentD)]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([self.ps.batch_size, (self.ps.latentD)]), requires_grad=False).to(device).type(dtype))

        loss_kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
        loss_kl = self.ps.kl_coef * loss_kl

        loss_dict = {'loss_kl': loss_kl,
                     'loss_mesh_rec': loss_mesh_rec,
                     'loss_vpose': loss_vpose,
                     }

        loss_dict_viz = {'loss_kl': loss_kl,
                         'loss_mesh_rec': loss_mesh_rec,
                         'loss_vpose': loss_vpose,
                        }

        loss_total = torch.stack(list(loss_dict.values())).sum() # + loss_vpose_rec
        loss_dict['loss_total'] = loss_total
        loss_dict['loss_total_viz'] = torch.stack(list(loss_dict_viz.values())).sum()

        return loss_total, loss_dict

    def perform_training(self, num_epochs=None, message=None):
        starttime = datetime.now().replace(microsecond=0)
        if num_epochs is None: num_epochs = self.ps.num_epochs

        self.logger(
            'Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), num_epochs))

        vis_bm =  BodyModel(self.ps.bm_path, num_betas=16).to(self.comp_device)
        prev_lr = np.inf
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.5)
        for epoch_num in range(self.epochs_completed, num_epochs + 1):
            scheduler.step()
            cur_lr = self.optimizer.param_groups[0]['lr']
            if cur_lr != prev_lr:
                self.logger('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr
            self.epochs_completed += 1
            self.sampler_train.set_epoch(epoch_num-1)
            train_loss_dict = self.train()
            self.sampler_val.set_epoch(0)
            eval_loss_dict = self.evaluate()

            if self.local_rank != 0:
                continue

            with torch.no_grad():
                eval_msg = MotionPriorTrainer.creat_loss_message(eval_loss_dict, expr_code=self.ps.expr_code,
                                                                 epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                                 try_num=self.try_num, mode='evald')
                if eval_loss_dict['loss_total'] < self.best_loss_total:
                    self.ps.best_model_fname = makepath(os.path.join(self.ps.work_dir, 'snapshots', 'TR%02d_E%03d.pt' % (
                    self.try_num, self.epochs_completed)), isfile=True)
                    self.logger(eval_msg + ' ** ')
                    self.best_loss_total = eval_loss_dict['loss_total']
                    torch.save(self.motion_prior.module.state_dict() if isinstance(self.motion_prior, torch.nn.DataParallel)
                               else self.motion_prior.state_dict(), self.ps.best_model_fname)
                else:
                    self.logger(eval_msg)

                self.swriter.add_scalars('total_loss/scalars', {'train_loss_total': train_loss_dict['loss_total_viz'],
                                                                'evald_loss_total': eval_loss_dict['loss_total_viz'], },
                                         self.epochs_completed)

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss_total))
        self.logger('Best model path: %s\n' % self.ps.best_model_fname)

    @staticmethod
    def creat_loss_message(loss_dict, expr_code='XX', epoch_num=0, it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s: [T:%.2e] - [%s]' % (
        expr_code, try_num, epoch_num, it, mode, loss_dict['loss_total'], ext_msg)

def run_motion_prior_trainer(ps):
    if not isinstance(ps, Configer):
        ps = Configer(default_ps_fname=ps)
    ps.work_dir = ps.work_dir + '/' + ps.expr_code 
    vp_trainer = MotionPriorTrainer(ps.work_dir, ps)

    ps.dump_settings(os.path.join(ps.work_dir, 'TR%02d_%s.ini' % (ps.try_num, ps.expr_code)))

    vp_trainer.logger(ps.expr_message)
    vp_trainer.perform_training()
    ps.dump_settings(os.path.join(ps.work_dir, 'TR%02d_%s.ini' % (ps.try_num, ps.expr_code)))

    vp_trainer.logger(ps.expr_message)

    test_loss_dict = vp_trainer.evaluate(split_name='test')
    vp_trainer.logger('Final loss on test set is %s' % (' | '.join(['%s = %.2e' % (k, v) for k, v in test_loss_dict.items()])))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a motion prior given settings',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_path', dest="config_path", type=str, help='path to ini file for Configer.')
    args = parser.parse_args()

    run_motion_prior_trainer(args.config_path)
