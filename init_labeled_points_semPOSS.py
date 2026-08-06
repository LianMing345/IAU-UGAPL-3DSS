"""Generate initial labeled-point JSON for SemanticPOSS train sequences (RC2 split)."""

import json
import os

import numpy as np
from tqdm import tqdm

from config import ConfigSemanticPoss as cfg

np.random.seed(0)

root = cfg.data_path
save_path = cfg.init_labeled_data
init_ratio = 0.00025  # 0.025% initial labels
train_seqs = ['00', '01', '02', '04', '05']

os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

initial_labeled_dic = {}
init_points = 0
all_points = 0

for sequence in tqdm(train_seqs):
    cur_dir = os.path.join(root, sequence, 'velodyne')
    if not os.path.isdir(cur_dir):
        print(f'Skip missing sequence dir: {cur_dir}')
        continue
    for file_name in os.listdir(cur_dir):
        if not file_name.endswith('.bin'):
            continue
        file_path = os.path.join(cur_dir, file_name)
        try:
            points = np.fromfile(file_path, dtype=np.float32)
            if points.size % 4 != 0:
                continue
            points = points.reshape(-1, 4)
        except Exception as e:
            print(f'Error reading file {file_path}: {e}')
            continue

        points_num = points.shape[0]
        all_points += points_num
        labeled_num_cur_pc = max(1, round(points_num * init_ratio))
        init_pts = np.argsort(np.random.rand(points_num))[:labeled_num_cur_pc]
        out = [False] * points_num
        for i in init_pts:
            out[i] = True
            init_points += 1
        pc_name = sequence + '#' + file_name[:-4]
        initial_labeled_dic[pc_name] = out

with open(save_path, 'w') as f1:
    json.dump(initial_labeled_dic, f1)

print(f'Saved: {save_path}')
print(f'Initial Points: {init_points}')
print(f'All Points: {all_points}')
