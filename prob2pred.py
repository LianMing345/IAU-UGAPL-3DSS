import numpy as np
import os
from tqdm import tqdm

# 定义输入和输出路径
probs_save_path = '/data1/jy/Result/HPAL-ICME-50/vis_scannet_HPAL/probs_save_path'
preds_save_path = '/data1/jy/Result/HPAL-ICME-50/vis_scannet_HPAL/prebs_save_path'

# 确保输出路径存在
os.makedirs(preds_save_path, exist_ok=True)

# 获取所有 .npy 文件的列表
npy_files = [filename for filename in os.listdir(probs_save_path) if filename.endswith('.npy')]

# 遍历输入路径下的所有 .npy 文件，并显示进度条
for filename in tqdm(npy_files, desc="Processing .npy files"):
    # 构建完整的文件路径
    prob_file_path = os.path.join(probs_save_path, filename)
    pred_file_path = os.path.join(preds_save_path, filename)

    # 读取概率值
    probabilities = np.load(prob_file_path)

    # 将概率值转换为预测值（假设是取最大概率对应的索引）
    predictions = np.argmax(probabilities, axis=1)+1

    # 保存预测值
    np.save(pred_file_path, predictions)
