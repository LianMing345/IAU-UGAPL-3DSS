import json
import os
import numpy as np
from tqdm import tqdm

np.random.seed(0)
# 这里自定义
root = '/data2/jy/1outof10_semantick_kitti_downed0.05/sequences'#🎈
save_path = 'init/semKITTI/1outof10_seed0_random_1%labelpts.json' #🎈
init_ratio = 0.05 #初始点标记比率

# 初始化用于存储各点云区域的初始标注信息的字典
initial_labeled_dic = {}
init_points = 0#
all_points = 0
# 默认8号序列不需要标注
for sequence in tqdm(['{:02d}'.format(i) for i in range(11) if i not in [8]]):
    # 拼接路径
    cur_dir = os.path.join(root, sequence, 'velodyne')
    file_names = os.listdir(cur_dir)
    for file_name in file_names:
        # 只处理 .bin 结尾的文件
        if not file_name.endswith('.bin'):
            continue

        file_path = os.path.join(cur_dir, file_name)

        try:
            # 读取点云文件
            points = np.fromfile(file_path, dtype=np.float32)

            if points.size % 4 != 0:
                continue

            points = points.reshape(-1, 4)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # 获取点云文件中的点数
        points_num = points.shape[0]
        # 将点云文件中的点数累加
        all_points += points_num
        # 随机选择点云文件中的点数(🎈这可以改)
        labeled_num_cur_pc = round(points_num * init_ratio)
        # 随机选择初始标注的点
        init_pts = np.argsort(np.random.rand(points_num))[:labeled_num_cur_pc]
        # 生成对应点的标注信息列表
        out = [False] * points_num
        for i in init_pts:
            out[i] = True
            init_points += 1
        # 构造点云名称
        pc_name = sequence + '#' + file_name[:-4]
        # 将当前点云的标注信息添加到字典中
        initial_labeled_dic[pc_name] = out

# 将标注信息字典保存到JSON文件中
with open(save_path, 'w') as f1:
    json.dump(initial_labeled_dic, f1)

print(f"Initial Points: {init_points}")
print(f"All Points: {all_points}")
