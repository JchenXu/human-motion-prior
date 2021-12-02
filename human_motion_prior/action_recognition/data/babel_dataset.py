import torch
import numpy as np
import pickle

from human_motion_prior.tools.training_tools import adapt_3d_np
from scipy.spatial.transform import Rotation as R
from scipy.fftpack import dct
from torch.utils.data import Dataset

class BabelData(Dataset):
    def __init__(self, data_split='train', babel_split=60, frame_num=300, block_size=5, frequency_num=20):

        assert data_split in ['train', 'test', 'val']

        if data_split == 'train':
            self.joints = np.load('/data/BABEL/for_ar/release/train_ntu_sk_{}.npy'.format(babel_split), mmap_mode='r')
            self.sample_name, self.labels = pickle.load(open('/data/BABEL/for_ar/release/train_label_{}.pkl'.format(babel_split), 'rb'),
                                          encoding='latin')
            print(self.joints.shape)  # seq_num, 3(xyz), max_frame, num_joint, body_num
        elif data_split == 'val':
            self.joints = np.load('/data/BABEL/for_ar/release/val_ntu_sk_{}.npy'.format(babel_split), mmap_mode='r')
            self.sample_name, self.labels = np.array(pickle.load(open('/data/BABEL/for_ar/release/val_label_{}.pkl'.format(babel_split), 'rb'),
                                          encoding='latin'))

            print(self.joints.shape)  # seq_num, 3(xyz), max_frame, num_joint, body_num
        elif data_split == 'test':
            self.joints = np.load('/data/BABEL/for_ar/release/test_ntu_sk_{}.npy'.format(babel_split), mmap_mode='r')
            print(self.joints.shape)


        TO_SDK19 = [1, 13, 14, 15, 17, 18, 19, 2, 21, 3, 4, 9, 10, 12, 5, 6, 8, 16, 20]
        self.TO_SDK19 = [ii - 1 for ii in TO_SDK19]

        self.data_split = data_split
        self.block_size = block_size
        self.frequency_num = frequency_num
        self.frame_num = frame_num
        self.basis = dct(np.eye(frame_num), norm='ortho', axis=0)[:self.frequency_num][None]
        self.basis_block = dct(np.eye(self.block_size), norm='ortho', axis=0)[:self.frequency_num][None]

        self.bonelens = np.array([[0.1449], [0.4546], [0.4392], [0.1441], [0.4564], [0.4332], [0.2647], [0.2516], [0.1045], [0.1149], [0.1661], [0.2887],
                                  [0.2679], [0.1655], [0.2961], [0.2643]])
        self.parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]  # 3, 6 --> SDK17+2


    def __len__(self):
        return self.joints.shape[0]

    def _rot_norm(self, joint_sdk19, org_trans):
        # joint_sdk19: frame_num, joint_num, 3
        frame_num, joint_num = joint_sdk19.shape[:2]

        joint_sdk19 = joint_sdk19 - joint_sdk19[:, :1, :]       # minus root
        first_hip = joint_sdk19[:1, 1] - joint_sdk19[:1, 4]
        first_spin =  joint_sdk19[:1, 8] - joint_sdk19[:1, 0]

        y_3d = np.array([[0., 1.0, 0.]])
        z_2d = np.array([[0., 1.0]])

        forward = np.cross(y_3d, first_hip)
        forward = forward / (np.linalg.norm(forward, axis=1, keepdims=True) + 1e-6)
        forward_2d = forward[:, [0, 2]]

        cos = forward_2d[:, 0] * z_2d[:, 0] + forward_2d[:, 1] * z_2d[:, 1]
        rotvec = np.array([[0.0,
                            np.arccos(cos) * -np.sign(forward_2d[:, 0]),
                            0.0]])
        r = R.from_rotvec(rotvec)
        convert_rot = r.as_matrix().repeat(frame_num * joint_num, axis=0)
        convert_trans_rot = r.as_matrix().repeat(frame_num, axis=0)

        joint_sdk19 = joint_sdk19.reshape(frame_num * joint_num, 3, 1)
        joint_sdk19 = np.matmul(convert_rot, joint_sdk19)
        joint_sdk19 = joint_sdk19.reshape(frame_num, joint_num, 3)

        first_hip = joint_sdk19[:1, 1] - joint_sdk19[:1, 4]
        first_spin = joint_sdk19[:1, 8] - joint_sdk19[:1, 0]

        y_3d = np.array([[0., 1.0, 0.]])
        z_2d = np.array([[0., 1.0]])

        forward = np.cross(y_3d, first_hip)
        forward = forward / (np.linalg.norm(forward, axis=1, keepdims=True) + 1e-6)
        forward_2d = forward[:, [0, 2]]
        # print(forward_2d)

        org_trans = org_trans.reshape(frame_num, 3, 1)
        norm_trans = np.matmul(convert_trans_rot, org_trans)
        norm_trans = norm_trans.reshape(frame_num, 3)

        # forward = np.cross(first_spin, first_hip)
        # forward[:, 0] = 0
        # forward_norm = (np.linalg.norm(forward, axis=1, keepdims=True) + 1e-6)
        # forward_tar = forward[:, [0, 2]]
        # forward_tar_norm = (np.linalg.norm(forward_tar, axis=1, keepdims=True) + 1e-6)
        #
        # rotvec = np.cross(forward, forward_tar)
        # rotvec = rotvec / (np.linalg.norm(rotvec, axis=1, keepdims=True) + 1e-6)
        #
        # degree = np.arccos(forward_tar_norm, forward_norm)
        # rotvec = rotvec * degree
        #
        # print(rotvec.shape)
        #
        # r = R.from_rotvec(rotvec)
        # convert_rot = r.as_matrix().repeat(frame_num * joint_num, axis=0)
        # convert_trans_rot = r.as_matrix().repeat(frame_num, axis=0)
        #
        # joint_sdk19 = joint_sdk19.reshape(frame_num * joint_num, 3, 1)
        # joint_sdk19 = np.matmul(convert_rot, joint_sdk19)
        # joint_sdk19 = joint_sdk19.reshape(frame_num, joint_num, 3)
        #
        # org_trans = org_trans.reshape(frame_num, 3, 1)
        # norm_trans = np.matmul(convert_trans_rot, org_trans)
        # norm_trans = norm_trans.reshape(frame_num, 3)

        return joint_sdk19, norm_trans

    def _frequency(self, p3ds):
        block_size = self.block_size
        joint_num = 17

        p3ds_flatten = p3ds.reshape(self.body_num, self.frame_num, -1)
        frequency_dct = np.matmul(self.basis, p3ds_flatten)
        frequency_dct = frequency_dct.reshape(self.body_num, -1)

        p3ds_pad = np.concatenate([p3ds_flatten[:, [0], :].repeat(2, axis=1),
                                   p3ds_flatten,
                                   p3ds_flatten[:, [-1], :].repeat(2, axis=1)], axis=1)

        p3ds_block = np.zeros((self.body_num, self.frame_num, block_size, joint_num * 3))
        for i in range(self.frame_num):
            p3ds_block[:, i, :, :] = p3ds_pad[:, i:i + block_size, :]
        p3ds_block = p3ds_block.reshape(self.body_num * self.frame_num, block_size, -1)

        frequency_block = np.matmul(self.basis_block, p3ds_block)
        frequency_block = frequency_block.reshape(self.body_num, self.frame_num * block_size * joint_num * 3)

        return frequency_dct, frequency_block


    def __getitem__(self, idx):
        joints = self.joints[idx].transpose(3, 1, 2, 0)     # to body_num, max_frame, num_joint, 3(xyz)

        if self.data_split == 'test':
            labels = np.array([-1]).reshape(-1)
        else:
            labels = np.array(self.labels[0][idx]).reshape(-1)

        self.body_num = joints.shape[0]

        joints_sdk19 = joints[:, :self.frame_num, self.TO_SDK19][:, :, :-2]   # remove foot SDK17
        # adjust joints
        joints_sdk19[:, :, 0] = (joints_sdk19[:, :, 1] + joints_sdk19[:, :, 4]) / 2
        joints_sdk19[:, :, 8] = (joints_sdk19[:, :, 11] + joints_sdk19[:, :, 14]) / 2
        joints_sdk19[:, :, 7] = (joints_sdk19[:, :, 8] + joints_sdk19[:, :, 0]) / 2
        joints_sdk19 = adapt_3d_np(joints_sdk19.reshape(-1, 17, 3), bonelens=self.bonelens,
                                   parents=self.parent).reshape(self.body_num, self.frame_num, 17, 3)

        # dummy rotation
        t = joints_sdk19[..., -1].copy()
        joints_sdk19[..., -1] = joints_sdk19[..., -2].copy()
        joints_sdk19[..., -2] = t

        ##################### ADD skip sample sequence for data augmentation #################
        joints_sdk19_subsample = joints_sdk19[:, ::2]
        joints_sdk19_subsample = np.concatenate([joints_sdk19_subsample,
                                                 joints_sdk19_subsample[:, -1:].repeat(128 - joints_sdk19_subsample.shape[1], axis=1)], axis=1)
        # if self.data_split == 'train':
        joints_sdk19 = np.concatenate([joints_sdk19, joints_sdk19_subsample], axis=0)
        labels = np.concatenate([labels, labels], axis=0)

        self.body_num = joints_sdk19.shape[0]
        org_trans = joints_sdk19[:, :, 0] - joints_sdk19[:, :1, 0]

        joints_sdk19_norm = np.zeros_like(joints_sdk19)
        norm_trans = np.zeros_like(org_trans)
        velocity = np.zeros_like(norm_trans)

        # if train ik module with denoising training this is not necessary
        for body in range(joints_sdk19.shape[0]):
            joints_sdk19_norm[body], norm_trans[body] = self._rot_norm(joints_sdk19[body], org_trans[body])

        velocity[:, 1:, :] = norm_trans[:, 1:, :] - norm_trans[:, :-1, :]

        frequency_dct, frequency_block = self._frequency(joints_sdk19_norm)

        return joints_sdk19_norm.reshape(*joints_sdk19.shape[:-2], -1), norm_trans, velocity, frequency_dct, frequency_block, \
               joints_sdk19.reshape(*joints_sdk19.shape[:-2], -1), labels





