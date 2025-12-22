import os
import numpy as np

def count_points_in_bin(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points.shape[0]

def count_total_points(base_path):###
    total_points = 0
    for seq in range(11):  # 假设序列编号从00到10
        seq_path = os.path.join(base_path, f"{seq:02d}", "velodyne")
        if not os.path.exists(seq_path):
            print(f"Warning: {seq_path} does not exist.")
            continue
        for bin_file in os.listdir(seq_path):
            if bin_file.endswith('.bin'):
                file_path = os.path.join(seq_path, bin_file)
                point_count = count_points_in_bin(file_path)
                total_points += point_count
                print(f"File: {file_path}, Points: {point_count}")
    return total_points

if __name__ == "__main__":
    # base_path = "/data2/jy/1outof10_semantick_kitti_downed0.05/sequences"
    base_path = "/data2/jy/semantic_kitti/sequences"
    total_points = count_total_points(base_path)
    print(f"Total points in all sequences: {total_points}")
