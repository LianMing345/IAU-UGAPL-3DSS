import os
import numpy as np

def count_points_in_npy(file_path):
    points = np.load(file_path).astype(np.float32)
    return points.size

def count_total_points(base_path):
    total_points = 0
    for area in range(1, 7):  # 假设区域编号从Area_1到Area_6
        area_path = os.path.join(base_path, f"Area_{area}", "coords")
        if not os.path.exists(area_path):
            print(f"Warning: {area_path} does not exist.")
            continue
        for npy_file in os.listdir(area_path):
            if npy_file.endswith('.npy'):
                file_path = os.path.join(area_path, npy_file)
                point_count = count_points_in_npy(file_path)
                total_points += point_count
                print(f"File: {file_path}, Points: {point_count}")
    return total_points

if __name__ == "__main__":
    base_path = "/data1/jy/data/s3dis"
    total_points = count_total_points(base_path)
    print(f"Total points in all areas: {total_points}")
