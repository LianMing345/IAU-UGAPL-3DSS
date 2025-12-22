import glob
import numpy as np
import os
import pickle
from helper_tool import DataProcessing as DP
from sklearn.neighbors import KDTree

# 🎈Data path (Need to modify)
semKITTI_3D_IN_PATH = '/data2/jy/semantic_kitti'
semKITTI_3D_OUT_PATH = '/data2/jy/semantick_kitti_downed0.05_v2'
sub_grid_size = 0.05  # 🎈对于SemanticKITTI，ReDAL是0.05m
print(sub_grid_size)

# 定义validation序列

# 确保目录存在，如果不存在则创建
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class SemKITTIDatasetConverter:
    @staticmethod#
    def read_bin(binfile):
        points = np.fromfile(binfile, dtype=np.float32).reshape(-1, 4)
        return points[:, :3], points[:, 3]

    @staticmethod
    def read_label(labelfile):
        labels = np.fromfile(labelfile, dtype=np.int32)#
        labels = labels & 0xFFFF  # 只保留标签的低16位
        return labels

    @staticmethod
    def write_bin(binfile, points, intensity):
        data = np.hstack((points, intensity.reshape(-1, 1))).astype(np.float32)
        data.tofile(binfile)

    @staticmethod
    def write_label(labelfile, labels):
        labels = labels.astype(np.int32)
        labels.tofile(labelfile)

    @classmethod
    def convert_to_bin_and_label(cls, data_path, out_path):
        # 确保输出目录存在
        ensure_dir(out_path)

        # 获取所有序列
        sequences = [f'sequences/{i:02d}' for i in range(11)]  # 仅处理00-10序列
        for sequence in sequences:

            # 获取该序列中的所有点云文件
            bin_files = sorted(glob.glob(os.path.join(data_path, sequence, 'velodyne/*.bin')))

            # 处理每隔10帧的点云文件
            for idx, bin_file in enumerate(bin_files):

                print(f"Processing {sequence}, frame {idx + 1} / {len(bin_files)}", flush=True)
                frame = os.path.splitext(os.path.basename(bin_file))[0]

                label_file = os.path.join(data_path, sequence, 'labels', frame + '.label')
                if not os.path.exists(label_file):
                    print(f"Label file {label_file} does not exist.")
                    continue

                # 读取点云和标签
                xyz, intensity = cls.read_bin(bin_file)
                labels = cls.read_label(label_file)

                # 合并点云数据和强度
                feats = intensity.reshape(-1, 1).astype(np.float32)

                # 确保输出目录结构
                seq_out_path = os.path.join(out_path, sequence, 'velodyne')
                ensure_dir(seq_out_path)
                seq_out_label_path = os.path.join(out_path, sequence, 'labels')
                ensure_dir(seq_out_label_path)

                out_bin = os.path.join(seq_out_path, frame + '.bin')
                out_label = os.path.join(seq_out_label_path, frame + '.label')

                # 确保 proj 文件夹存在
                seq_out_proj_path = os.path.join(out_path, sequence, 'proj')
                ensure_dir(seq_out_proj_path)
                proj_file = os.path.join(seq_out_proj_path, frame + '_proj.pkl')

                # 对点云进行降采样
                coords_min = np.amin(xyz, axis=0)
                xyz -= coords_min

                sub_xyz, sub_feats, sub_labels = DP.grid_sub_sampling(xyz, feats, labels, sub_grid_size)
                search_tree = KDTree(sub_xyz)
                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

                # 写入降采样后的 .bin 和 .label 文件
                cls.write_bin(out_bin, sub_xyz, sub_feats[:, 0])
                cls.write_label(out_label, sub_labels)

if __name__ == '__main__':
    SemKITTIDatasetConverter.convert_to_bin_and_label(semKITTI_3D_IN_PATH, semKITTI_3D_OUT_PATH)
