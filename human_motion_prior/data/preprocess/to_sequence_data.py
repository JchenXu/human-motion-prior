import open3d as o3d
import torch
import numpy as np
import cv2
import h5py

from torch.utils.data import DataLoader

frame_num = 128
batch_size = 192
np2pt = lambda x: torch.from_numpy(x)

class AMASSDataset(torch.utils.data.Dataset):
    def __init__(self, options, dataset_path, is_train=True, all=False):
        self.is_train = is_train
        self.data = h5py.File(dataset_path, "r")
        self.poses = self.data["poses"]
        self.betas = self.data["betas"]

        print('size = ', self.poses.shape, flush=True)

        if all:
            self.length = int(self.poses.shape[0])
            self.poses = self.data["poses"]
            self.betas = self.data["betas"]
            self.flags = self.data["flags"]
            self.trans = self.data["trans"]
        else:
            if self.is_train:
                self.length = int(self.poses.shape[0] * 0.85)
                self.poses = self.data["poses"][:self.length]
                self.betas = self.data["betas"][:self.length]
                self.flags = self.data["flags"][:self.length]
                self.trans = self.data["trans"][:self.length]
            else:
                self.length = self.poses.shape[0] - int(self.poses.shape[0] * 0.85)
                self.poses = self.data["poses"][-self.length:]
                self.betas = self.data["betas"][-self.length:]
                self.flags = self.data["flags"][-self.length:]
                self.trans = self.data["trans"][-self.length:]

        self.R_np = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

        self.mapping = [0, 7, 8, 9, 10, 1, 2, 3, 3, 4, 5, 6, 6, 11, 12, 13, 13, 14, 15, 16, 16]
        self.rotation = torch.eye(3)
        self.rotation[1, 1] = -1
        self.rotation[2, 2] = -1
        self.rotation = self.rotation.unsqueeze(0)
        self.translation = torch.tensor([[0, 0, 3.0]], dtype=torch.float32)
        self.camera_center = torch.tensor([500, 500], dtype=torch.float32)

    def visualize(self, vertice, name):
        pcd_list = []

        for ver in vertice:
            pcd_a = o3d.geometry.PointCloud()
            pcd_a.points = o3d.utility.Vector3dVector(ver)
            pcd_list.append(pcd_a)

        o3d.visualization.draw_geometries(pcd_list, name)

    def __getitem__(self, idx):
        pose = torch.tensor(self.poses[idx][:], dtype=torch.float32).unsqueeze(0)
        assert pose.shape[-1] == 156
        beta = torch.tensor(self.betas[idx][:], dtype=torch.float32).unsqueeze(0)
        assert beta.shape[-1] == 16
        flag = self.flags[idx]
        trans = self.trans[idx]

        global_orient = pose[:, :3]
        rotate_mat = cv2.Rodrigues(global_orient.numpy())[0]
        rotate_mat = np.dot(self.R_np, rotate_mat)
        rotate_vec = cv2.Rodrigues(rotate_mat)[0]
        pose[:, :3] = torch.from_numpy(rotate_vec).view(1, 3)
        body_pose = pose[:, 3:]

        item = {}
        item["pose"] = pose.squeeze(0)
        item["betas"] = beta.squeeze(0)
        item["flag"] = flag
        item["trans"] = trans

        return item
        # return body_vertices.squeeze().numpy(), flag

    def __len__(self):
        return self.length


def visualize(vertice, name):
    pcd_list = []

    for ver in vertice:
        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(ver)
        pcd_list.append(pcd_a)

    o3d.visualization.draw_geometries(pcd_list, name)


def to_sequence_npz():
    visual = False

    dataset = AMASSDataset(None, './preprocessed_data/amass_smpl_30fps_complete.h5', all=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    interval = 32
    sequence_data = []
    for idx, batch in enumerate(loader):
        print("{} / {}".format(idx, len(loader)), flush=True)
        batch_poses = batch["pose"]     # (frame_num, 72)
        batch_betas = batch["betas"]    # (frame_num, 10)
        batch_flags = batch["flag"].float()     # (frame_num, 1)
        batch_trans = batch["trans"].float()

        if batch_poses.shape[0] < batch_size:
            print('drop')
            continue

        for i in range((batch_size - frame_num) // interval + 1):
            poses = batch_poses[i*interval: i*interval + frame_num]
            betas = batch_betas[i*interval: i*interval + frame_num]
            flags = batch_flags[i*interval: i*interval + frame_num]
            trans = batch_trans[i*interval: i*interval + frame_num]

            if len(torch.unique(flags)) > 1:
                print(torch.unique(flags))
                continue

            seq_data = torch.cat([poses, betas, trans, flags], -1)
            sequence_data.append(seq_data.numpy()[None, ...])

    out_data = np.concatenate(sequence_data, axis=0)
    print(out_data.shape)
    np.savez('./amass_smpl_30fps_128frame.npz', out_data)


if __name__ == '__main__':
    to_sequence_npz()
