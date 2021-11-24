from abc import ABCMeta, abstractmethod
import numpy as np
import math
from scipy.spatial.transform.rotation import Rotation as R


class DCTSmoother(object):
    def __init__(self, nbasis=10, win_len=50, blend_len=10):
        self.nbasis = nbasis
        self.win_len = win_len
        self.blend_len = blend_len
        self.basis = gen_dct_basis(win_len, nbasis)

    def smooth(self, data):
        """

        :param data: T x D
        :return:
        """
        data_s = np.zeros_like(data)
        for i in range(self.win_len, data.shape[0], self.win_len - self.blend_len):
            # each time fill [i-win_len,i-1], blend range [i-win_len, i-win_len+blend_len]
            data_window = data[i - self.win_len:i]
            data_window_s, _ = dctsmooth(data_window, self.basis)
            ## blend pose
            if i > self.win_len:
                for k in range(self.blend_len):
                    weight = 1.0 / self.blend_len * (k + 1)
                    data_s[i - self.win_len + k] = data_window_s[k] * weight + data_s[i - self.win_len + k] * (1 - weight)
                    data_s[i - self.win_len + self.blend_len: i] = data_window_s[self.blend_len:self.win_len]
            else:
                data_s[i - self.win_len:i] = data_window_s
        # blend last sequence
        blend_len_last = self.win_len - (data.shape[0] - self.win_len) % (self.win_len - self.blend_len)
        data_window_last = data[-self.win_len:]
        data_window_last_s, _ = dctsmooth(data_window_last, self.basis)
        for k in range(blend_len_last):
            weight = 1.0 / blend_len_last * (k + 1)
            data_s[-self.win_len + k] = data_window_last_s[k] * weight + data_s[-self.win_len + k] * (1 - weight)
            data_s[-self.win_len + blend_len_last:] = data_window_last_s[blend_len_last:self.win_len]
        return data_s


class AASmoother(object):
    '''
    Given an sequence of angle axis, smooth it with DCT
    '''
    def __init__(self, num_joints, nbasis_list, win_len=50, blend_len=10):
        assert num_joints == len(nbasis_list)
        self.num_joints = num_joints
        self.nbasis_list = nbasis_list
        self.win_len = win_len
        self.blend_len = blend_len
        self.basis = []
        for nbasis in self.nbasis_list:
            basis = gen_dct_basis(win_len, nbasis)
            self.basis.append(basis.reshape(1, win_len, nbasis))

    def smooth(self, aa):
        """
        :param aa: T x Nj x 3
        :return: aa_s: T x Nj x 3
        """
        # aa = make_aa_continuous(aa)
        aa_s = np.zeros_like(aa)
        for i in range(self.win_len, aa.shape[0], self.win_len - self.blend_len):
            for j in range(aa.shape[1]):
                aa_window = aa[i - self.win_len:i, j].reshape(-1, 1, 3)
                aa_window_s = dctsmooth_batch(aa_window, self.basis[j])
                ## blend pose
                if i > self.win_len:
                    for k in range(self.blend_len):
                        weight = 1.0 / self.blend_len * (k + 1)
                        aa_s[i -self.win_len + k, j] = aa_window_s[k] * weight + aa_s[i - self.win_len + k, j] * (1 - weight)
                    aa_s[i - self.win_len + self.blend_len: i, None, j] = aa_window_s[self.blend_len:self.win_len]
                else:
                    aa_s[i - self.win_len:i, None, j] = aa_window_s
        # blend last sequence
        blend_len_last = (self.win_len - (aa.shape[0] - self.win_len) % (self.win_len - self.blend_len)) % self.win_len
        for j in range(aa.shape[1]):
            aa_window_last = aa[-self.win_len:, j].reshape(-1, 1, 3)
            aa_window_last_s = dctsmooth_batch(aa_window_last, self.basis[j])
            for k in range(blend_len_last):
                weight = 1.0 / blend_len_last * (k + 1)
                aa_s[-self.win_len + k, j] = aa_window_last_s[k] * weight + aa_s[-self.win_len + k, j] * (1 - weight)
                aa_s[-self.win_len + blend_len_last:, None, j] = aa_window_last_s[blend_len_last:self.win_len]
        return aa_s

def make_aa_continuous(aa):
    for i in range(1, aa.shape[0]):
        for j in range(aa.shape[1]):
            aa[i, j] = check_aa_invert(aa[i - 1, j], aa[i, j])
    return aa


def check_aa_invert(prev_aa, cur_aa):
    """
    Consider two aa (-3.14, 0, 0) and (3.14, 0, 0)
    They almost the same, but the arithmetical mean of them are (0,0,0)
    which may lead wrong interpolation
    We modify the latter one by invert the angle of aa.
    :param prev_aa:
    :param cur_aa:
    :return:
    """
    flag = False
    avg_aa = R.from_rotvec(0.5 * prev_aa + 0.5 * cur_aa)
    meanR = R.from_rotvec([prev_aa, cur_aa]).mean()
    diff = avg_aa.inv() * meanR
    if diff.magnitude() > math.pi * 0.5:
        flag = True
    if flag:
        cur_theta = np.linalg.norm(cur_aa)
        if cur_theta > 1e-6:
            cur_aa = cur_aa / cur_theta * (cur_theta - math.pi * 2)
    return cur_aa


class EKF(object):
    '''
    A abstrat class for the Extended Kalman Filter, based on the tutorial in
    http://home.wlu.edu/~levys/kalman_tutorial.
    Code from here
    https://github.com/simondlevy/TinyEKF/blob/master/extras/python/tinyekf/__init__.py
    '''
    __metaclass__ = ABCMeta

    def __init__(self, n, m, pval=0.1, qval=1e-4, rval=0.1):
        '''
        Creates a KF object with n states, m observables, and specified values for
        prediction noise covariance pval, process noise covariance qval, and
        measurement noise covariance rval.
        '''

        # No previous prediction noise covariance
        self.P_pre = None

        # Current state is zero, with diagonal noise covariance matrix
        self.x = np.zeros(n)
        self.P_post = np.eye(n) * pval

        # Set up covariance matrices for process noise and measurement noise
        self.Q = np.eye(n) * qval
        self.R = np.eye(m) * rval

        # Identity matrix will be usefel later
        self.I = np.eye(n)

    def step(self, z):
        '''
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        '''

        # Predict ----------------------------------------------------

        # $\hat{x}_k = f(\hat{x}_{k-1})$
        self.x, F = self.f(self.x)

        # $P_k = F_{k-1} P_{k-1} F^T_{k-1} + Q_{k-1}$
        self.P_pre = F * self.P_post * F.T + self.Q

        # Update -----------------------------------------------------

        h, H = self.h(self.x)

        # $G_k = P_k H^T_k (H_k P_k H^T_k + R)^{-1}$
        G = np.dot(self.P_pre.dot(H.T), np.linalg.inv(H.dot(self.P_pre).dot(H.T) + self.R))

        # $\hat{x}_k = \hat{x_k} + G_k(z_k - h(\hat{x}_k))$
        self.x += np.dot(G, (np.array(z) - h.T).T)

        # $P_k = (I - G_k H_k) P_k$
        self.P_post = np.dot(self.I - np.dot(G, H), self.P_pre)

        # return self.x.asarray()
        return self.x

    @abstractmethod
    def f(self, x):
        '''
        Your implementing class should define this method for the state-transition function f(x).
        Your state-transition fucntion should return a NumPy array of n elements representing the
        new state, and a nXn NumPy array of elements representing the the Jacobian of the function
        with respect to the new state.  Typically this is just the identity
        function np.copy(x), so the Jacobian is just np.eye(len(x)).  '''
        raise NotImplementedError()

    @abstractmethod
    def h(self, x):
        '''
        Your implementing class should define this method for the observation function h(x), returning
        a NumPy array of m elements, and a NumPy array of m x n elements representing the Jacobian matrix
        H of the observation function with respect to the observation. For
        example, your function might include a component that turns barometric
        pressure into altitude in meters.
        '''
        raise NotImplementedError()


class DepthEKF(EKF):
    def __init__(self, pval=50, qval=1, rval=10):
        EKF.__init__(self, 1, 1, pval, qval, rval)
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def reset(self, x):
        self.x = np.array(x).reshape(-1, 1)

    def f(self, x):
        # State-transition function is identity
        return np.copy(x)

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        return np.eye(1)

    def h(self, x):
        # Observation function is identity
        return x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return np.eye(1)

    def step(self, z):
        '''
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        '''
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.P_pre = self.F * self.P_post * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = np.dot(self.P_pre * self.H.T, np.linalg.inv(self.H * self.P_pre * self.H.T + self.R))
        self.x += np.dot(G, (np.array(z) - self.h(self.x).T).T)
        self.P_post = np.dot(self.I - np.dot(G, self.H), self.P_pre)

        # return self.x.asarray()
        return self.x


class KeyPointEKF(EKF):
    def __init__(self, in_dimension, pval=50, qval=1, rval=10):
        EKF.__init__(self, in_dimension, in_dimension, pval, qval, rval)
        self.in_dimension = in_dimension

    def reset(self, x):
        self.x = np.array(x)

    def f(self, x):
        # State-transition function is identity
        return np.copy(x), np.eye(self.in_dimension)

    def h(self, x):
        # Observation function is identity
        return x, np.eye(self.in_dimension)


class SmootherKeyPointEKF(object):
    def __init__(self, num_in_points, num_in_features, default_ratio=None):
        """
        Extended Kalman Filter with num_frames * num_in_points * num_in_features
        :param num_in_points: i.e. 17
        :param num_in_features: i.e. 2 for 2d, 3 for 3d
        :param default_ratio: i.e. (5, 3.5) for (limbs, torso)
        """
        self.filters = []
        smooth_ratios = num_in_points * [5 if default_ratio is None else default_ratio[0]]
        if num_in_points == 28:
            torso_ext = [2, 3, 8, 9, 27]
        elif num_in_points == 17:
            torso_ext = [0, 1, 4, 7, 8, 11, 14]
        else:
            torso_ext = [2, 3, 8, 9]
        for keypoint in torso_ext:
            smooth_ratios[keypoint] = 3.5 if default_ratio is None else default_ratio[1]
        for i in range(num_in_points):
            self.filters.append(KeyPointEKF(in_dimension=num_in_features, qval=smooth_ratios[i]))
        self.num = num_in_points
        self.in_features = num_in_features
        self.is_First = True

    def step(self, data_input):
        if self.is_First:
            self.is_First = False
            for i in range(self.num):
                self.filters[i].reset([data_input[i][j] for j in range(self.in_features)])
        data_rtn = []
        for i in range(self.num):
            data_rtn.append(self.filters[i].step([data_input[i][j] for j in range(self.in_features)]))
        return np.array(data_rtn).reshape(self.num, self.in_features)


class CommonEKF(object):
    def __init__(self):
        self.filters = {}

    def step(self, key, data, qval=5, rval=10):
        num_dim = data.shape[-1]
        if key in self.filters.keys():
            return self.filters[key].step(data)
        else:
            self.filters[key] = KeyPointEKF(in_dimension=num_dim, qval=qval, rval=rval)
            self.filters[key].x = data
            return self.filters[key].step(data)


class MoveEKF(object):
    def __init__(self, n, pval=10, qval=1, rval=10):
        """

        :param in_dimension:
        :param pval: initial convariance
        :param qval: for state transfer
        :param rval: for observation
        """
        self.n = n
        # No previous prediction noise covariance
        self.P_pre = None

        # Current state is zero, with diagonal noise covariance matrix
        self.x = np.zeros(n)
        self.P_post = np.eye(n) * pval

        # Set up covariance matrices for process noise and measurement noise
        self.Q = np.eye(n) * qval
        self.R = np.eye(n) * rval

        # Identity matrix will be useful later
        self.I = np.eye(n)

    def reset(self, x):
        self.x = np.array(x)

    def f(self, x, u, v):
        """
        :param x:
        :param u: the speed of the 2D keypoint, dim(x)=dim(u)
        :param v: the norm of the speed on pixels
        :param w: uncertaincy of the predicted 2D keypoint
        :return:
        """
        # State-transition function is identity
        self.Q = np.eye(self.n) * v
        return np.copy(x + u), np.eye(self.n)

    def h(self, x):
        # Observation function is identity
        return x, np.eye(self.n)

    def step(self, z, u, v, w=10):
        '''
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        :param z: Predicted key-point, observation
        :param u: speed vector on 2D
        :param v: the scalar indicate the speed on 2D
        :param w: uncertaincy of the predicted 2D keypoint
        '''

        # Predict ----------------------------------------------------

        # $\hat{x}_k = f(\hat{x}_{k-1})$
        self.x, F = self.f(self.x, u, v)
        self.R = np.eye(self.n) * (w + 1)

        # $P_k = F_{k-1} P_{k-1} F^T_{k-1} + Q_{k-1}$
        self.P_pre = F * self.P_post * F.T + self.Q

        # Update -----------------------------------------------------

        h, H = self.h(self.x)

        # $G_k = P_k H^T_k (H_k P_k H^T_k + R)^{-1}$
        G = np.dot(self.P_pre.dot(H.T), np.linalg.inv(H.dot(self.P_pre).dot(H.T) + self.R))

        # $\hat{x}_k = \hat{x_k} + G_k(z_k - h(\hat{x}_k))$
        self.x += np.dot(G, (np.array(z) - h.T).T)

        # $P_k = (I - G_k H_k) P_k$
        self.P_post = np.dot(self.I - np.dot(G, H), self.P_pre)

        # return self.x.asarray()
        return self.x


class CommonMoveEKF(object):
    def __init__(self, num_dim, pval, qval, rval):
        self.filters = {}
        self.num_dim = num_dim
        self.pval = pval
        self.qval = qval
        self.rval = rval

    def step(self, key, point, speed, velocity, uncertainty=10):
        assert point.shape[-1] == self.num_dim
        assert speed.shape[-1] == self.num_dim
        if key in self.filters.keys():
            return self.filters[key].step(point, speed, velocity, uncertainty)
        else:
            self.filters[key] = MoveEKF(self.num_dim, pval=self.pval, qval=self.qval, rval=self.rval)
            self.filters[key].reset(point)
            return self.filters[key].step(point, speed, velocity, uncertainty)


def GaussMul(z1, v1, z2, v2):
    dim = z1.shape[0]
    assert z2.shape[0] == dim
    z = z1 + (v1 * v1) / (v1 * v1 + v2 * v2) * (z2 - z1)
    v = math.sqrt((v1 * v1 * v2 * v2) / (v1 * v1 + v2 * v2))
    return z, v


def GaussMulwithConf(z1, v1, c1, z2, v2, c2):
    dim = z1.shape[0]
    assert z2.shape[0] == dim
    K = 1 / math.sqrt(2 * math.pi)
    t1 = c1 * v1 / K
    t2 = c2 * v2 / K
    z = z1 + (v1 * v1) / (v1 * v1 + v2 * v2) * (z2 - z1)
    v = math.sqrt((v1 * v1 * v2 * v2) / (v1 * v1 + v2 * v2))
    c = min(max(K / v * math.sqrt(t1 * t2), 1e-2), 1-1e-2)
    return z, v, c


"""
This script only aims to use DCT to smooth the shaking skeletons.
Visualization can be realized by other tools. (PoseViz is suggested.)
"""

import numpy as np
from scipy.fftpack import dct
from scipy.interpolate import interp1d


def gen_dct_basis(L, n):
    """
    produce dct basis.
    L: length of skeleton sequence
    n: wanted basis number
    return: dct basis
    """
    D = dct(np.eye(L), norm='ortho', axis=0)
    basis = D[:n].T
    return basis


def interpolate_basis_bytools(basis, target_video_frames, axis=0):
    """
    axis: which axis is the temporal dimension
    basis: dct basis, shape: (L, n)
    interpolatomg video from n to target_video_frames
    target_video_frames: target frame number
    """
    x = np.array(range(basis.shape[axis]))
    f1 = interp1d(x, basis, kind='quadratic', axis=axis)

    x_int = np.linspace(x.min(), x.max(), target_video_frames)
    y_int = f1(x_int)
    return y_int


def run_dct(basis, c):
    """
    y = basis @ c
    """
    y = basis @ c
    return y


def dctsmooth(x, basis, target_video_frames=None):
    """
    B @ c - x = 0
    solution is: c = (B^H @ B) ^-1 @ B^H @ x
    """
    print((np.linalg.inv(basis.T @ basis) @ basis.T).shape)
    c = np.linalg.inv(basis.T @ basis) @ basis.T @ x
    if target_video_frames is not None:
        basis = interpolate_basis_bytools(basis, target_video_frames)
    y = run_dct(basis, c)
    return y, c


def dctsmooth_batch(x, basis, target_video_frames=None):
    """
    X: (T, nJ, 3)
    basis: (nJ, T, n), n is the num of dct basis
    target_video_frames: target video frame number
    """
    b_T = np.transpose(basis, (0, 2, 1))
    c = np.matmul(np.matmul(np.linalg.inv(np.matmul(b_T, basis)), b_T), np.transpose(x, (1, 0, 2)))
    if target_video_frames is not None:
        basis = interpolate_basis_bytools(basis, target_video_frames, axis=1)
    y = np.matmul(basis, c)
    return y.transpose(1, 0, 2)

def dctsmooth_batch_ratio(x, ratio, target_video_frames=None):
    """
    X: (T, nJ, 3)
    basis: (nJ, T, n), n is the num of dct basis
    target_video_frames: target video frame number
    """
    tlen = x.shape[0]
    nbasis = int(tlen * ratio)
    basis = np.tile(gen_dct_basis(tlen, nbasis), (x.shape[1], 1, 1))

    b_T = np.transpose(basis, (0, 2, 1))
    c = np.matmul(np.matmul(np.linalg.inv(np.matmul(b_T, basis)), b_T), np.transpose(x, (1, 0, 2)))
    if target_video_frames is not None:
        basis = interpolate_basis_bytools(basis, target_video_frames, axis=1)
    y = np.matmul(basis, c)
    return y.transpose(1, 0, 2)

def smooth_root(aa, ratio=0.2, order=[0, 2, 1]):
    """
    :param aa:  T X 3 np.ndarray
    :return:
    """
    root = np.tile(np.eye(3), (aa.shape[0], 1, 1))
    rotmat = R.from_rotvec(aa).as_matrix()
    coord = np.matmul(root, rotmat.transpose(0, 2, 1))
    coord_smooth = dctsmooth_batch_ratio(coord, ratio)
    coord_smooth[:, 0] = np.cross(coord_smooth[:, 1], coord_smooth[:, 2])
    coord_smooth[:, 0] /= np.linalg.norm(coord_smooth[:, 0], axis=-1, keepdims=True)
    coord_smooth[:, 2] = np.cross(coord_smooth[:, 0], coord_smooth[:, 1])
    coord_smooth[:, 2] /= np.linalg.norm(coord_smooth[:, 0], axis=-1, keepdims=True)
    coord_smooth[:, 1] /= np.linalg.norm(coord_smooth[:, 1], axis=-1, keepdims=True)
    coord_smooth = coord_smooth.transpose(0, 2, 1)
    r_smooth = R.from_matrix(coord_smooth)
    return r_smooth.as_rotvec()

def lpf_smooth(s3d, a, b):
    """

    :param s3d: K x J x 3
    :param a: [5,]
    :param b: [5,]
    :return:
    """
    lpf_num = len(a)
    s3d_y_ = [s3d[0].copy() for _ in range(lpf_num - 1)]
    s3d_x_ = np.pad(s3d, ((lpf_num - 1, 0), (0, 0), (0, 0)), 'edge')
    for i, s3d1f in enumerate(s3d_x_):
        if i < lpf_num - 1:
            continue
        # y(i) = [b(0) * x(i) + b(1) * x(i - 1) + ....b(4) * x(i - 4) - a(1) * y(i - 1) - ... - a(4) * y(i - 4)] / a(0)
        part_ = b[0] * s3d1f + sum([b[j] * s3d_x_[i - j] for j in range(1, lpf_num)]) - sum(
            [a[j] * s3d_y_[i - j] for j in range(1, lpf_num)])
        part_ /= a[0]
        s3d_y_.append(part_)
    s3d_y_ = s3d_y_[lpf_num - 1:]
    return np.array(s3d_y_)


def lpf_smooth_shift(s3d, a, b, k=2):
    """
    wrapper of lpf_smooth, with k-frame-shift added
    :param s3d:
    :param a:
    :param b:
    :param k:
    :return:
    """
    res = lpf_smooth(s3d, a, b)
    return np.concatenate([res[k:], res[-1:].repeat(repeats=k, axis=0)], axis=0)


