import os
import numpy as np
import argparse
import natsort
import cv2
from tqdm import tqdm
import open3d as o3d
import math

import sys
sys.path.append('./')

from common.utils import save_image
from common.utils_loader import read_calib_file_orfd, pose_read_orfd, bin_read

def planeFit(points):
    """平面拟合函数"""
    points = np.transpose(points)
    points = np.reshape(points, (points.shape[0], -1))
    assert points.shape[0] <= points.shape[1], "Insufficient points for plane fitting"
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    return ctr, np.linalg.svd(M)[0][:, -1]


def compute_roughness(pc, pcd_tree, radius=0.5, k=50):
    """带统计的粗糙度计算"""
    roughness = np.full(len(pc), np.nan)
    valid_data = []  # 用于存储有效的粗糙度数据
    valid_count = 0

    for pt_idx in range(len(pc)):
        [k_found, idx, _] = pcd_tree.search_hybrid_vector_3d(pc[pt_idx], radius, k)
        if k_found < 3:
            continue

        try:
            pt, normal = planeFit(pc[idx])
            normal = normal / np.linalg.norm(normal)
            plane_to_point = pc[pt_idx] - pt
            dist = abs(np.dot(normal, plane_to_point))
            roughness[pt_idx] = dist
            valid_data.append(dist)
            valid_count += 1
        except:
            pass

    return roughness, valid_data, valid_count


# 3️⃣ 计算投影 ###
def project_points_to_image(points, P_velo2im):
    """
    将 LiDAR 点投影到相机图像
    :param points: (N, 3) LiDAR 点
    :param P_velo2im: (3,4) LiDAR -> 相机投影矩阵
    :return: 2D 投影坐标 (N, experiment) 和 3D 点
    """
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points[:, :3], ones))  # (N, 4)

    proj_points = (P_velo2im @ points_homogeneous.T).T  
    proj_points[:, :2] /= proj_points[:, 2, np.newaxis]  

    valid_mask = proj_points[:, 2] > 0  # 仅保留 Z > 0 的点
    pts_2d = proj_points[valid_mask, :2]
    points_filtered = points[valid_mask]

    return pts_2d, points_filtered


def print_stats(roughness, valid_data):
    """输出粗糙度统计信息"""
    stats = {
        'Total Points': len(roughness),  # 总点数
        'Valid Points': len(valid_data),  # 有效点数
        'Invalid Points': np.isnan(roughness).sum(),  # 无法计算的点数
        'Mean (mm)': np.mean(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',  # 平均粗糙度
        'Std Dev (mm)': np.std(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',  # 标准差
        'Max (mm)': np.max(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',  # 最大粗糙度
        'Min (mm)': np.min(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',  # 最小粗糙度
        'Median (mm)': np.median(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',  # 中位数
        '95th Percentile (mm)': np.percentile(valid_data, 95) * 1000 if len(valid_data) > 0 else 'N/A'  # 95%的数据点粗糙度低于此值
    }

    print("\nRoughness Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()


if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Project LiDAR Points to Camera View")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--roughness_threshold", type=float, default=0.01, help="粗糙度阈值")

    args = parser.parse_args()

    print("*********************************************")
    print("[i] Start LiDAR Projection")
    print("[i] Data root: ", args.data_root)
    print("[i] Mode: ", args.mode)
    print("[i] Sequence: ", args.seq)

    file_name = "x0613_1627"
    # 设置存储路径
    save_root = os.path.join(args.data_root, "ORFD-custom", args.mode, "roughness", args.seq)
    os.makedirs(save_root, exist_ok=True)
    
    data_path = os.path.join(args.data_root, "Final_Dataset", args.mode, file_name, "image_data")
    file_list = natsort.natsorted(os.listdir(data_path))

    seq_dict = {}
    for fname in file_list:
        k = fname[:8]  
        if k not in seq_dict.keys():
            seq_dict[k] = []
        seq_dict[k].append(fname[:-4])  

    for seq_name in seq_dict.keys():
        if seq_name != args.seq:
            continue

        description = f"[i] Processing Sequence: {seq_name}"
        for i, str_frame in enumerate(tqdm(seq_dict[seq_name], desc=description)):
            
            # 读取相机标定参数
            calib_file = os.path.join(args.data_root, "Final_Dataset", args.mode, file_name, "calib", str_frame + '.txt')
            P_velo2im = read_calib_file_orfd(calib_file)  

            # 读取相机图像
            image_path = os.path.join(args.data_root, "Final_Dataset", args.mode, file_name, "image_data", str_frame + '.png')
            image = cv2.imread(image_path)

            # 读取 LiDAR 点云数据
            lidar_file = os.path.join(args.data_root, "Final_Dataset", args.mode, file_name, "lidar_data", str_frame + '.bin')
            lidar_data = bin_read(lidar_file)

            # 计算粗糙度
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_data)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            roughness, valid_data, valid_count = compute_roughness(lidar_data, pcd_tree, radius=0.5, k=50)

            # 输出统计信息
            print_stats(roughness, valid_data)

            # 投影点到相机
            pts_2d, lidar_points_filtered = project_points_to_image(lidar_data, P_velo2im)

            image_project = image.copy()
            for j, point_2d in enumerate(pts_2d):
                x, y = int(point_2d[0]), int(point_2d[1])
                rough_value = roughness[j]  # 获取该点的粗糙度

                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    color = (0, 255, 0) if rough_value <= args.roughness_threshold else (0, 0, 255)  # 绿色 = ≤阈值，红色 = >阈值
                    cv2.circle(image_project, (x, y), 2, color, -1)  # 绘制圆圈标记

            save_path = os.path.join(save_root, str_frame + ".png")
            cv2.imwrite(save_path, image_project)

    print("[i] Finished LiDAR Projection")
    print("*********************************************")
