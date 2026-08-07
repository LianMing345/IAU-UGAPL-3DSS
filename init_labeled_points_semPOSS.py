"""Generate initial labeled-point JSON for SemanticPOSS train sequences (RC2 split)."""

import json
import os

import numpy as np
from tqdm import tqdm

from config import ConfigSemanticPoss as cfg
from Mink.label_maps import get_learning_map

np.random.seed(0)

root = cfg.data_path
save_path = cfg.init_labeled_data
init_ratio = 0.00025  # 0.025% initial labels
train_seqs = ['00', '01', '02', '04', '05']
learning_map = get_learning_map('semanticposs', cfg.num_classes)

if cfg.TRANSFER not in ('syn2poss', 'nus2poss'):
    raise ValueError(
        f'init_labeled_points_semPOSS.py requires a poss transfer, got {cfg.TRANSFER}'
    )

os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

initial_labeled_dic = {}
init_points = 0
all_points = 0
valid_points = 0

print(f'TRANSFER: {cfg.TRANSFER}; target classes: {cfg.num_classes}')

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
        sequence_dir = os.path.dirname(os.path.dirname(file_path))
        label_name = os.path.splitext(os.path.basename(file_path))[0] + '.label'
        label_path = os.path.join(sequence_dir, 'labels', label_name)
        if not os.path.isfile(label_path):
            print(f'Skip missing label file: {label_path}')
            continue

        try:
            raw_labels = np.fromfile(label_path, dtype=np.int32).reshape(-1)
            if raw_labels.size != points_num:
                print(
                    f'Skip mismatched labels: {file_path} '
                    f'({points_num} points vs {raw_labels.size} labels)'
                )
                continue
            raw_labels &= 0xFFFF
            mapped_labels = np.fromiter(
                (learning_map.get(int(label), 0) for label in raw_labels),
                dtype=np.int32,
                count=raw_labels.size,
            )
        except Exception as e:
            print(f'Error reading labels {label_path}: {e}')
            continue

        valid_indices = np.flatnonzero(mapped_labels > 0)
        if valid_indices.size == 0:
            print(f'Skip frame without valid mapped labels: {file_path}')
            continue

        all_points += points_num
        valid_points += valid_indices.size
        labeled_num_cur_pc = max(1, round(points_num * init_ratio))
        labeled_num_cur_pc = min(labeled_num_cur_pc, valid_indices.size)
        init_pts = np.random.choice(valid_indices, labeled_num_cur_pc, replace=False)
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
print(f'Valid Mapped Points: {valid_points}')
