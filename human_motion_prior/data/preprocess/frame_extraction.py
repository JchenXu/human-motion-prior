import os
import pickle
import torch
import numpy as np
import h5py

frame_num = 30
fps = 30

def load_smpl_parameters(poses, betas, trans, step):
    np.set_printoptions(suppress=True)
    # smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    R = torch.tensor(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T, dtype=torch.float32)
    mapping = [0, 7, 8, 9, 10, 4, 5, 6, 6, 1, 2, 3, 3, 14, 15, 16, 16, 11, 12, 13, 13]
    pose_size = poses.shape[0]
    # K = np.array(constants.K)
    # batch_s3ds = []
    batch_poses = []
    batch_betas = []
    batch_trans = []
    for idx in range(0, pose_size, step):
        batch_betas.append(betas[:])
        batch_poses.append(poses[idx][:])
        batch_trans.append(trans[idx])

    # batch_s3ds = np.array(batch_s3ds).reshape(-1, 21, 3)
    batch_poses = np.array(batch_poses).reshape(-1, 156) # 156 for pose
    batch_betas = np.array(batch_betas).reshape(-1, 16) # 16 params for shape!!
    batch_trans = np.array(batch_trans).reshape(-1, 3)
    return batch_poses, batch_betas, batch_trans


def np_concat(data, new_batch):
    if data is None:
        data = new_batch
    else:
        data = np.concatenate((data, new_batch), axis=0)
    return data


def process_amass(amass_dir):
    poses = None  # [n, 72]
    betas = None  # [n, 10]
    flags = None
    trans = None
    cnt = 0
    for root, dirs, files in os.walk(amass_dir):
        for f in files:
            if f[-4:] == '.npz':
                if f[-9:] == "shape.npz":
                    continue
                print("amass process... [{} / {}] {:.2f}%".format(cnt, 100230, cnt / 10230.0), flush=True)
                filename = os.path.join(root, f)
                npz = np.load(filename)
                data_poses = npz["poses"]
                data_betas = npz["betas"]
                data_trans = npz["trans"]
                source_fps = npz["mocap_framerate"]
                step = source_fps // fps
                print(filename, source_fps, flush=True)
                batch_poses, batch_betas, batch_trans = load_smpl_parameters(data_poses, data_betas, data_trans, int(step))
                batch_flags = np.ones((batch_poses.shape[0], 1)) * cnt
                poses = np_concat(poses, batch_poses)
                betas = np_concat(betas, batch_betas)
                flags = np_concat(flags, batch_flags)
                trans = np_concat(trans, batch_trans)
                cnt += 1

    print("cnt = ", cnt, flush=True)
    print(poses.shape, betas.shape, flags.shape, trans.shape, flush=True)
    return poses, betas, flags, trans


def main():
    datadir = "./raw_data"
    output_file = "./preprocessed_data/amass_smpl_30fps_complete.h5"
    poses, betas, flags, trans = process_amass(datadir)

    h5 = h5py.File(output_file, 'w')
    h5.create_dataset("poses", data=poses)
    h5.create_dataset("betas", data=betas)
    h5.create_dataset("flags", data=flags)
    h5.create_dataset("trans", data=trans)
    h5.close()



if __name__ == '__main__':
    print('start ...', flush=True)
    main()
