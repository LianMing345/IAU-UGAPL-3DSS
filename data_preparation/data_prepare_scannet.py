# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import sys
import numpy as np
import os
import pickle
from plyfile import PlyData
from sklearn.neighbors import KDTree
sys.path.append("..")

from helper_tool import DataProcessing as DP

# When you enable the code with comment "For train/val set", you need disable the code with comment "For test set"
# When you enable the code with comment "For test set", you need disable the code with comment "For train/val set"

# Data path (Need to modify)
STANFORD_3D_IN_PATH = '/userHOME/yb/data/scannet/scans/'  # For train/val set
STANFORD_3D_OUT_PATH = '/userHOME/yb/data/HPASSL/scannet/'

split_path_root = 'split/scannet/'  # The official split of train/val/test


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class ScannetDatasetConverter:

    @classmethod
    def convert_to_npy(cls, root_path, out_path):
        """Convert ScanNet to NPY format.
        Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
        """

        out_path_train = os.path.join(out_path, "train")
        out_path_val = os.path.join(out_path, "val")
        out_path_test = os.path.join(out_path, "test")

        for out_path_x in [out_path_train, out_path_val, out_path_test]:
            os.makedirs(os.path.join(out_path_x, "coords"), exist_ok=True)
            os.makedirs(os.path.join(out_path_x, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(out_path_x, "labels"), exist_ok=True)
            # os.makedirs(os.path.join(out_path_x, "proj"), exist_ok=True)  # Used when downsampling point clouds are requied

        file_dirs = os.listdir(root_path)

        # read data split
        split_path_train = os.path.join(split_path_root, 'scannetv2_train.txt')
        split_path_val = os.path.join(split_path_root, 'scannetv2_val.txt')
        split_path_test = os.path.join(split_path_root, 'scannetv2_test.txt')
        with open(split_path_train, 'r') as f_t:
            train_pc = [line[:-1] for line in f_t.readlines()]
        with open(split_path_val, 'r') as f_v:
            val_pc = [line[:-1] for line in f_v.readlines()]
        with open(split_path_test, 'r') as f_test:
            test_pc = [line[:-1] for line in f_test.readlines()]

        for file in file_dirs:
            print(file)

            path = glob.glob(os.path.join(root_path, file, '*.labels.ply'))[0]  # For train/val set
            # path = glob.glob(os.path.join(root_path, file, '*.ply'))[0]  # For test set

            data = PlyData.read(path).elements[0].data
            xyz = np.vstack((data['x'], data['y'], data['z'])).T
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            label = data['label']  # For train/val set

            xyz = xyz.astype(np.float32)
            rgb = rgb.astype(np.uint8)
            label = label.astype(np.uint8)  # For train/val set

            file_sp = os.path.normpath(path).split('/')

            if file in train_pc:
                out_coords = os.path.join(out_path_train, "coords", file_sp[-1].split('.')[0][:12] + '.npy')
                out_rgb = os.path.join(out_path_train, "rgb", file_sp[-1].split('.')[0][:12] + '.npy')
                out_labels = os.path.join(out_path_train, "labels", file_sp[-1].split('.')[0][:12] + '.npy')
                out_proj = os.path.join(out_path_train, "proj", file_sp[-1].split('.')[0][:12] + '_proj.pkl')
            elif file in val_pc:
                out_coords = os.path.join(out_path_val, "coords", file_sp[-1].split('.')[0][:12] + '.npy')
                out_rgb = os.path.join(out_path_val, "rgb", file_sp[-1].split('.')[0][:12] + '.npy')
                out_labels = os.path.join(out_path_val, "labels", file_sp[-1].split('.')[0][:12] + '.npy')
                out_proj = os.path.join(out_path_val, "proj", file_sp[-1].split('.')[0][:12] + '_proj.pkl')
            elif file in test_pc:
                out_coords = os.path.join(out_path_test, "coords", file_sp[-1].split('.')[0][:12] + '.npy')
                out_rgb = os.path.join(out_path_test, "rgb", file_sp[-1].split('.')[0][:12] + '.npy')
                out_proj = os.path.join(out_path_test, "proj", file_sp[-1].split('.')[0][:12] + '_proj.pkl')

            np.save(out_coords, xyz)
            np.save(out_rgb, rgb)
            np.save(out_labels, label)  # For train/val set


            # For scannet, we do not use downsample by default because the voxel size of MinkowskiNet's optimum input is small(0.02)
            ##########################downsampling point clouds################################################
            # coords_min = np.amin(xyz, axis=0)
            # xyz -= coords_min
            #
            # sub_grid_size = 0.02
            #
            # # for train/val set
            # sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, rgb, label,
            #                                                        sub_grid_size)
            #
            # # for test
            # # label = np.zeros(shape=(len(xyz),)).astype(np.uint8)
            # # sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, rgb, label, sub_grid_size)
            #
            # search_tree = KDTree(sub_xyz)
            #
            # proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
            # proj_idx = proj_idx.astype(np.int32)
            #
            # with open(out_proj, 'wb') as f:
            #     pickle.dump([proj_idx, label], f)
            #
            # np.save(out_coords, sub_xyz)
            # np.save(out_rgb, sub_colors)
            #
            # # for train/val set
            # np.save(out_labels, sub_labels)
            ##########################downsampling point clouds################################################



if __name__ == '__main__':
    ScannetDatasetConverter.convert_to_npy(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH)

