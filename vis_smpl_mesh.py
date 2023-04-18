import os
from smplx import SMPL
import torch
import numpy
import pickle as pkl
import vedo
import trimesh
import time

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
    

    vertices = smpl.forward(
        global_orient=torch.from_numpy(global_orient).float(),
        body_pose=torch.from_numpy(body_pose).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).vertices.detach().numpy().reshape(N,T,-1,3)

    for i in range(T):
        verts = vertices[:,i]  # first frame
        faces = smpl.faces
        meshes = []
        for vert in verts:
            mesh = trimesh.Trimesh(vert, faces)
            mesh.visual.face_colors = [200, 200, 250, 100]
            meshes.append(mesh)

        plotter = vedo.show(meshes,  interactive=False)
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

