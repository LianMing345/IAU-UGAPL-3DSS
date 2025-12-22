import json
import os
import numpy as np

from Mink.dataloader.dataset import ScannetDataset
from data_base import Scannet
import config
from Mink.base_agent import BaseTrainer as minkNet
#TODO 暂时不考虑在TS上运行！
np.random.seed(0)

label_to_names = {
    1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair',
    6: 'sofa', 7: 'table', 8: 'door', 9: 'window', 10: 'bookshelf',
    11: 'picture', 12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refrigerator',
    28: 'shower curtain', 33: 'toilet', 34: 'sink', 36: 'bathtub', 39: 'otherfurniture'
}
label_to_idx = np.ones(shape=(41,)) * -100
for i, k in enumerate(label_to_names):
    label_to_idx[k] = i

root = '/data1/jy/data/scannet'
save_path = '/data1/yb/code/HPAL-ICME-50/HPAL/data_preparation/init/scannet/random_seed_v0_0.5percent.json'

initial_labeled_dic = {}
init_points = 0
all_points = 0
totoal_points = 0

# 用于存储每个场景中每个标签的点数
scene_label_count = {}

cur_dir = os.path.join(root, 'train', 'coords')
file_names = os.listdir(cur_dir)
print(file_names)

# 加载S3DIS模型从而AL
cfg =  config.ConfigScannet
Log_file = cfg.saving_path
dataset = Scannet(Log_file, cfg)
model = minkNet(cfg, Log_file, dataset)

train_dataset = ScannetDataset(dataset.input_xyz['train'], dataset.input_colors['train'],
                               dataset.label_to_idx,
                               dataset.input_names['train'], labels=dataset.input_labels['train'],
                               labeled_points=dataset.labeled_points, voxel_size=0.02)
val_dataset = ScannetDataset(dataset.input_xyz['validation'], dataset.input_colors['validation'],
                             dataset.label_to_idx,
                             dataset.input_names['validation'], labels=dataset.input_labels['validation'],
                             voxel_size=0.02)

model_path = '/data1/yb/model/HPAL-ICME-50/HPAL+TS-inconsistency-trainweightv0061/random_seed_v0_0.05percent/mink_pth_s/checkpoint1.tar'
model.load_checkpoint(model_path, local_rank=0)
# Active learning
# score_final = generate_score(cfg, model_student, dataset, train_dataset, Log_file)
score_final = generate_score(cfg, model_teacher, model_student, dataset, train_dataset, Log_file)
log_out('scoring finish', Log_file)

active_chose(cfg, score_final, dataset, log_file=Log_file)
log_out('choosing finish', Log_file)

# Scannet dataset needs to exclude 20 classes which are not used
for name in file_names:
    cur_path_xyz = os.path.join(cur_dir, name)
    xyz = np.load(cur_path_xyz)

    cur_path_label = cur_path_xyz.replace('coords', 'labels')
    labels = np.load(cur_path_label)
    labels = label_to_idx[labels]

    valid_totoal_Num = np.sum(labels != -100)

    # 初始化每个标签的计数
    scene_label_count[name] = {label: 0 for label in label_to_names.values()}

    # 统计每个标签的点数
    for label in label_to_names.values():
        scene_label_count[name][label] = np.sum(
            labels == list(label_to_names.keys())[list(label_to_names.values()).index(label)])

    # labeled_num_cur_pc = 3#🎈
    labeled_num_cur_pc = round(valid_totoal_Num * 0.005)

    # random initial
    init_pts = np.argsort(np.random.rand(xyz.shape[0]))

    out = [False] * len(init_pts)
    idx = 0
    while labeled_num_cur_pc > 0:
        i = init_pts[idx]
        if labels[i] != -100:
            out[i] = True
            init_points += 1
            labeled_num_cur_pc -= 1
        idx += 1

    all_points += valid_totoal_Num
    totoal_points += len(xyz)

    pc_name = name[:-4]

    initial_labeled_dic[pc_name] = out

# 保存初始标记的数据
with open(save_path, 'w') as f1:
    json.dump(initial_labeled_dic, f1)

# 打印统计信息
print(f'Initial Points: {init_points}')
print(f'All Points: {all_points}')
print(f'Total Points: {totoal_points}')

# 打印每个场景中的标签点数
# for scene, counts in scene_label_count.items():
#     print(f'Scene: {scene}')
    # for label, count in counts.items():
    #     print(f'  {label}: {count}')
