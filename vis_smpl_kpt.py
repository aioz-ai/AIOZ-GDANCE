import os
from smplx import SMPL
import torch
import numpy as np
import pickle as pkl
import vedo
import trimesh
import time

COLOR = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
    ]
).astype(np.float32).reshape(-1, 3) 
COLOR = (255*COLOR).astype(int)

def vis(args):
    data_dir = args.data_dir
    seq_name = args.sequence_name
    smpl_path = args.smpl_path

    smpl = SMPL(smpl_path, batch_size=1)

    data = pkl.load(open(os.path.join(data_dir, seq_name),"rb"))

    smpl_poses = data['smpl_poses']
    smpl_trans = data['root_trans']

    

    N, T = smpl_poses.shape[:2]
    J = smpl.NUM_JOINTS #23


    global_orient = smpl_poses[:,:,:3].reshape(-1, 3)
    body_pose = smpl_poses[:,:,3:].reshape(-1, J*3)
    smpl_trans = smpl_trans.reshape(-1,3)
    

    all_joints3d = smpl.forward(
        global_orient=torch.from_numpy(global_orient).float(),
        body_pose=torch.from_numpy(body_pose).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy().reshape(N,T,-1,3)


    for i in range(T):
        joints3d = all_joints3d[:,i]  # first frame
        pts = []
        for p, joint in enumerate(joints3d):
            pts.append(vedo.Points(joint, r=20, c = COLOR[p]))

        plotter = vedo.show(pts,  interactive=False)
        plotter.clear()
        time.sleep(1/30)





if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='motions_smpl', help='path to motions_smpl folder containing .pkl files')
    parser.add_argument('--sequence_name', type=str, default=None, help='')
    parser.add_argument('--smpl_path', type=str, default="smpl/SMPL_MALE.pkl", help='')
    args = parser.parse_args()

    vis(args)

