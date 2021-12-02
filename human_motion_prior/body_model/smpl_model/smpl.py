import torch
import numpy as np
import smplx
from smplx import SMPL as _SMPL
from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints

from human_motion_prior.body_model.smpl_model import config, constants


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        J_regressor_h36m = np.load(config.JOINT_REGRESSOR_H36M)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        self.joints = None

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        self.joints = smpl_output.joints
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

    def get_joints_h36m(self, vertices):
        joints = vertices2joints(self.J_regressor_h36m, vertices)
        return joints

    def get_joints_ori(self):
        return self.joints
