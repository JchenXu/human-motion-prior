import os

import torch
import numpy as np
import tqdm

from torch.utils.data import DataLoader
from configer import Configer
from human_motion_prior.action_recognition.motion_prior_ik import MotionPriorIK

c2c = lambda x: x.detach().cpu().numpy()

if __name__ == '__main__':
    device = torch.device('cuda')
    FRAME_NUM = 128
    batch_size = 10

    block_size = 5
    frequency_num = 20

    ####################### LOAD MPOSER #######################
    mposer_path = './logs/motion_prior_ik/snapshots/TR00_E007.pt'
    ini_path = './logs/motion_prior_ik/TR00_motion_prior_ik.ini'
    ps = Configer(default_ps_fname=ini_path)  # This is the default configuration
    ik = MotionPriorIK(num_neurons=ps.num_neurons, latentD=ps.latentD, latentD_t=ps.latentD_t,
                       dense_freq=ps.dense_freq, block_size=block_size, frequency_num=frequency_num,
                       frame_num=ps.frame_num, use_cont_repr=ps.use_cont_repr)
    state_dict = torch.load(mposer_path, map_location='cpu')
    new_state_dict = {k[7:]: v for k, v in state_dict.items()}
    ik.load_state_dict(new_state_dict)
    ik.eval().to(device)
    
    from human_motion_prior.action_recognition.data.babel_dataset import BabelData
    data_split = 'test'
    babel_split = 60
    babel_dataset = BabelData(data_split=data_split, babel_split=babel_split, frame_num=128, frequency_num=frequency_num, block_size=block_size)
    babel_loader = DataLoader(babel_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    embedding = np.zeros((0, 256))
    embedding_std = np.zeros((0, 256))
    labels = np.zeros((0,))

    for i, data in enumerate(tqdm.tqdm(babel_loader)):
        joints_sdk19_norm, norm_trans, velocity, frequency_dct, frequency_block, _, label = data

        item = {'trans': norm_trans, 'p3ds': joints_sdk19_norm, 'velocity': velocity,
                'frequency': frequency_dct, 'frequency_block': frequency_block}

        for k, v in item.items():
            item[k] = v.reshape(-1, *v.shape[2:]).to(device).float()


        batch_size = item['p3ds'].shape[0]

        ############### get the latent representation ###############
        Pin = torch.cat([item['trans'], item['velocity'], item['p3ds']], dim=-1)
        q_z = ik.encode(Pin, item['frequency'], item['frequency_block'])
        z = q_z.mean
        std = q_z.scale
        
        embedding = np.concatenate([embedding, c2c(z)], axis=0)
        embedding_std = np.concatenate([embedding_std, c2c(std)], axis=0)
        labels = np.concatenate([labels, c2c(label).reshape(batch_size, )])

    if not os.path.exists('ik_embedding'):
        os.makedirs('ik_embedding')

    suffix = 'noise'
    if data_split == 'train':
        np.savez('ik_embedding/ik_embedding_babel{}_train_{}.npz'.format(babel_split, suffix), embedding=embedding, labels=labels, embedding_std=embedding_std)
    elif data_split == 'val':
        np.savez('ik_embedding/ik_embedding_babel{}_val_{}.npz'.format(babel_split, suffix), embedding=embedding, labels=labels, embedding_std=embedding_std)
    elif data_split == 'test':
        np.savez('ik_embedding/ik_embedding_babel{}_test_{}.npz'.format(babel_split, suffix), embedding=embedding, labels=labels, embedding_std=embedding_std)

