import glob
import json
import os
from os.path import join

import numpy as np
from tqdm import tqdm


class S3DIS:
    def __init__(self, test_area_idx, log_file, cfg):
        self.log_file = log_file
        self.name = 'S3DIS'
        self.path = cfg.data_path
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = []

        for area in ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']:
            cur_dir = os.path.join(self.path, area, 'coords')
            files = glob.glob(join(cur_dir, '*.npy'))
            self.all_files += files

        # print(self.all_files)#

        f_l = open(cfg.init_labeled_data, 'r')
        self.labeled_points = json.load(f_l)
        f_l.close()

        self.input_xyz = {'train': [], 'validation': []}
        self.input_colors = {'train': [], 'validation': []}
        self.input_labels = {'train': [], 'validation': []}
        self.input_names = {'train': [], 'validation': []}

        self.load_sub_sampled_clouds()

    def load_sub_sampled_clouds(self):
        for i, file_path in enumerate(self.all_files):
            xyz_path = file_path
            colors_path = file_path.replace('coords', 'rgb')
            label_path = file_path.replace('coords', 'labels')

            area = file_path.split('/')[-3]
            cloud_name = file_path.split('/')[-1][:-4]
            sp_key_name = area + '#' + cloud_name

            if sp_key_name in self.labeled_points:
                cloud_split = 'train'
                # print('train:' + sp_key_name)#
            else:
                cloud_split = 'validation'
                # print('val:' + sp_key_name)#

            self.input_xyz[cloud_split] += [xyz_path]
            self.input_colors[cloud_split] += [colors_path]
            self.input_labels[cloud_split] += [np.load(label_path)]
            self.input_names[cloud_split] += [sp_key_name]


class Scannet:
    def __init__(self, log_file, cfg):
        self.log_file = log_file
        self.name = 'Scannet'
        self.path = cfg.data_path
        self.label_to_names = {1: 'wall',
                               2: 'floor',
                               3: 'cabinet',
                               4: 'bed',
                               5: 'chair',
                               6: 'sofa',
                               7: 'table',
                               8: 'door',
                               9: 'window',
                               10: 'bookshelf',
                               11: 'picture',
                               12: 'counter',
                               14: 'desk',
                               16: 'curtain',
                               24: 'refrigerator',
                               28: 'shower curtain',
                               33: 'toilet',
                               34: 'sink',
                               36: 'bathtub',
                               39: 'otherfurniture'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = np.ones(41) * -100
        self.label_to_idx = self.label_to_idx.astype(np.int32)
        for i, k in enumerate(self.label_to_names):
            self.label_to_idx[k] = i

        self.all_files = []

        for area in ['train', 'val', 'test']:
            cur_dir = os.path.join(self.path, area, 'coords')
            files = glob.glob(join(cur_dir, '*.npy'))
            self.all_files += files

        # print(self.all_files)#

        f_l = open(cfg.init_labeled_data, 'r')
        self.labeled_points = json.load(f_l)
        f_l.close()

        self.input_xyz = {'train': [], 'validation': [], 'test': []}
        self.input_colors = {'train': [], 'validation': [], 'test': []}
        self.input_labels = {'train': [], 'validation': []}
        self.input_names = {'train': [], 'validation': [], 'test': []}
        self.pseudoconfidence = {'train': []}

        self.val_proj = {'validation': [], 'test': []}

        self.load_sub_sampled_clouds()

    def load_sub_sampled_clouds(self):
        split_path_val = 'data_preparation/split/scannet/scannetv2_val.txt'
        with open(split_path_val, 'r') as f_val:
            val_pc = [line[:-1] for line in f_val.readlines()]
        f_val.close()

        for i, file_path in enumerate(self.all_files):
            xyz_path = file_path
            colors_path = file_path.replace('coords', 'rgb')
            label_path = file_path.replace('coords', 'labels')
            proj_path = file_path.replace('coords', 'proj')
            proj_path = proj_path.replace('.npy', '_proj.pkl')

            cloud_name = file_path.split('/')[-1][:-4]

            xyz = np.load(xyz_path)

            if cloud_name in self.labeled_points:
                cloud_split = 'train'#
                # print('train:' + xyz_path + ':' + str(len(xyz)))
                self.pseudoconfidence[cloud_split] += [np.zeros_like(np.load(label_path))]
                self.input_labels[cloud_split] += [label_path]
            elif cloud_name in val_pc:
                cloud_split = 'validation'#
                # print('val:' + xyz_path + ':' + str(len(xyz)))
                self.input_labels[cloud_split] += [label_path]
                self.val_proj[cloud_split] += [proj_path]
            else:
                cloud_split = 'test'#
                # print('test:' + xyz_path + ':' + str(len(xyz)))
                self.val_proj[cloud_split] += [proj_path]

            self.input_xyz[cloud_split] += [xyz_path]
            self.input_colors[cloud_split] += [colors_path]
            self.input_names[cloud_split] += [cloud_name]

class SemanticKITTI:
    """
        定义一个semanticKITTI类，用于加载和处理semanticKITTI数据集的子样本点云数据
        参数：
        - test_area_idx:测试区域的索引，semanticKITTI验证集是8
        - log_file:日志文件路径，用于记录处理过程中的信息
        - cfg:配置对象，包含数据路径和其他设置
    """
    def __init__(self, val_seq_idx, log_file, cfg):
        #初始化类变量
        self.log_file = log_file
        self.name = "SemanticKITTI"
        self.path = cfg.data_path #/data/xzy/jy/HPAL-main/data/semantic_kitti/sequences
        self.num_classes = cfg.num_classes
        # 根据测试区域索引确定验证集的划分。
        self.val_split = str(val_seq_idx)
        self.all_files = []
        #收集0-22区域的数据文件
        for sequence in tqdm(['{:02d}'.format(i) for i in range(22)]):
            cur_dir = os.path.join(self.path,sequence,'velodyne')
            files = glob.glob(join(cur_dir,'*.bin'))
            self.all_files += files
        # print(self.all_files)
        # 加载已标注数据点
        f_l = open(cfg.init_labeled_data,'r')
        self.labeled_points = json.load(f_l)
        f_l.close()
        # 初始化存储数据的字典
        self.input_pc = {'train':[],'validation':[],'test':[]}
        self.input_xyz= {'train':[],'validation':[],'test':[]}
        self.input_labels = {'train': [], 'validation': [],'test':[]}
        self.input_names = {'train': [], 'validation': [],'test':[]}
        self.load_clouds()

    def load_clouds(self):
        for i, file_path in enumerate(tqdm(self.all_files, desc="Processing files")):
            pc_path = file_path
            label_path = file_path.replace('velodyne', 'labels')
            sequence = file_path.split('/')[-3]
            cloud_name = file_path.split('/')[-1][:-4]
            sp_key_name = sequence + '#' + cloud_name

            # 硬编码 sequence 划分
            seq_num = int(sequence)
            if 0 <= seq_num <= 7 or 9 <= seq_num <= 10:
                cloud_split = 'train'
            elif seq_num == 8:
                cloud_split = 'validation'
            elif 11 <= seq_num <= 21:
                cloud_split = 'test'
            else:
                # 可以根据实际情况处理未知的 sequence
                cloud_split = 'validation'  # 或者 'train' / 'test'

            # 将当前点云的路径、标签和点云名称添加到相应训练、验证、测试集中，以便后续处理和访问
            self.input_pc[cloud_split] += [pc_path]
            self.input_xyz[cloud_split] += [pc_path]
            self.input_labels[cloud_split] += [label_path]
            self.input_names[cloud_split] += [sp_key_name]


class SemanticPoss:
    """
         定义一个semanticPOSS类，用于加载和处理semanticPOSS数据集的子样本点云数据
         参数：
         - test_area_idx:测试区域的索引，semanticPOSS验证集是：4、5
         - log_file:日志文件路径，用于记录处理过程中的信息
         - cfg:配置对象，包含数据路径和其他设置
     """

    def __init__(self, test_seq_idx, log_file, cfg):
        # 初始化类变量
        self.log_file = log_file
        self.name = "SemanticPOSS"
        self.path = cfg.data_path  # /data/xzy/jy/HPAL-main/data/semantic_kitti/sequences
        self.num_classes = 13
        # 根据测试区域索引确定验证集的划分。
        self.val_split = str(test_seq_idx)
        self.all_files = []
        # 收集0-5区域的数据文件
        for sequence in tqdm(['{:02d}'.format(i) for i in range(6)]):
            cur_dir = os.path.join(self.path, sequence, 'velodyne')
            files = glob.glob(join(cur_dir, '*.bin'))
            self.all_files += files
        # print(self.all_files)
        # 加载已标注数据点
        f_l = open(cfg.init_labeled_data, 'r')
        self.labeled_points = json.load(f_l)
        f_l.close()
        # 初始化存储数据的字典
        self.input_pc = {'train': [], 'validation': []}
        self.input_xyz = {'train': [], 'validation': []}
        self.input_labels = {'train': [], 'validation': []}
        self.input_names = {'train': [], 'validation': []}
        self.load_clouds()

    def load_clouds(self):
        for i, file_path in enumerate(tqdm(self.all_files, desc="Processing files")):
            # for i,file_path in enumerate(self.all_files):#['/data/xzy/jy/HPAL-main/data/semantic_kitti/sequences/xx/velodyne/xxxxxx.bin',...]
            pc_path = file_path
            label_path = file_path.replace('velodyne', 'labels')
            sequence = file_path.split('/')[-3]
            cloud_name = file_path.split('/')[-1][:-4]
            sp_key_name = sequence + '#' + cloud_name
            # 还要加
            if sp_key_name in self.labeled_points:
                cloud_split = 'train'
                # print('train:'+sp_key_name)
            else:
                cloud_split = 'validation'
                # print('val:'+sp_key_name)
            # 将当前点云的路径、标签和点云名称添加到相应训练、验证、测试集中，以便后续处理和访问
            # print(pc_path)
            self.input_pc[cloud_split] += [pc_path]
            self.input_xyz[cloud_split] += [pc_path]
            self.input_labels[cloud_split] += [label_path]
            self.input_names[cloud_split] += [sp_key_name]
