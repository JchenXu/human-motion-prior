import tqdm
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

c2c = lambda x: x.detach().cpu().numpy()

class AMASSSeqDataset(Dataset):
    def __init__(self, data_dir, ratio=0.85, is_train=True):
        self.is_train = is_train
        self.data = np.load(data_dir)['arr_0']

        if self.is_train:
            self.length = int(self.data.shape[0] * ratio)
            self.data = self.data[:self.length]
            self.p3ds = np.load('../data/p3ds_train.npy')
            assert self.data.shape[0] == self.p3ds.shape[0], 'training data length should be same to the p3ds'

        else:
            self.length = self.data.shape[0] - int(self.data.shape[0] * ratio)
            self.data = self.data[-self.length:]
            self.p3ds = np.load('../data/p3ds_eval.npy')
            assert self.data.shape[0] == self.p3ds.shape[0], 'evaluation data length should be same to the p3ds'


    def __getitem__(self, idx):
        seq = self.data[idx]
        seq_pose = seq[:, :156]
        seq_beta = seq[:, 156: 156 + 16]
        seq_trans = seq[:, 156 + 16: 156 + 16 + 3]
        seq_flag = seq[:, -1:]
        p3ds = self.p3ds[idx]

        item = {}
        item["pose"] = torch.from_numpy(seq_pose)
        item["beta"] = torch.from_numpy(seq_beta)
        item["trans"] = torch.from_numpy(seq_trans)
        item["flag"] = torch.from_numpy(seq_flag)
        item["p3ds"] = torch.from_numpy(p3ds)

        return item

    def __len__(self):
        return self.length
