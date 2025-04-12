import os
import numpy as np
import argparse
import natsort
import cv2
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

import sys
sys.path.append('./')

from common.utils import save_image
from common.utils_loader import read_calib_file_orfd, pose_read_orfd, bin_read

import open3d as o3d

def voxelize_lidar(lidar_points, voxel_size=0.25):
    """
    对 LiDAR 点云进行体素化（分辨率 0.25m），同时计算每个体素的粗糙度的平均值
    :param lidar_points: (N, 3) 点云数据
    :param voxel_size: 体素大小 (默认 0.25m)
    :return: 体素中心点 & 体素的平均高度
    """
    # 计算每个点的体素索引
    voxel_indices = np.floor(lidar_points[:, :3] / voxel_size).astype(int)

    # 用字典存储体素的最大 Z 值和所有粗糙度数据
    voxel_dict = {}
    for idx, voxel in enumerate(voxel_indices):
        key = tuple(voxel)  # 体素索引作为键
        if key in voxel_dict:
            voxel_dict[key]['points'].append(lidar_points[idx])
        else:
            voxel_dict[key] = {'points': [lidar_points[idx]]}

    # 计算每个体素的中心点和每个体素的平均高度
    voxel_centers = []
    voxel_heights_avg = []

    for key in voxel_dict.keys():
        # 计算体素中心点
        voxel_center = np.mean(voxel_dict[key]['points'], axis=0)
        # 计算体素内点的高度的平均值
        avg_height = np.mean([point[2] for point in voxel_dict[key]['points']])

        voxel_centers.append(voxel_center)
        voxel_heights_avg.append(avg_height)

    voxel_centers = np.array(voxel_centers)
    voxel_heights_avg = np.array(voxel_heights_avg)

    return voxel_centers, voxel_heights_avg

# 2️⃣ 计算地面高度（5% 分位数法） ###
def estimate_ground_height_percentile(voxel_heights, percentile=5):
    """
    计算体素地面高度，使用最低 5% 体素高度的均值
    :param voxel_heights: (N,) 体素高度
    :param percentile: 分位数阈值，默认 5%
    :return: 估计的地面高度
    """
    ground_height = np.percentile(voxel_heights, percentile)
    return ground_height


# 3️⃣ 计算物体高度 ###
def compute_voxel_heights(voxel_heights, ground_height):
    """
    计算每个体素的物体高度（相对于地面）
    :param voxel_heights: (N,) 体素的最大 Z 值
    :param ground_height: 估计的地面高度
    :return: 体素的物体高度
    """
    heights = voxel_heights - ground_height
    return heights


# 3️⃣ 计算投影 ###
def project_voxels_to_image(voxel_centers, P_velo2im):
    """
    将体素中心点投影到相机图像
    :param voxel_centers: (N, 3) 体素中心点
    :param P_velo2im: (3,4) LiDAR -> 相机投影矩阵
    :return: 2D 投影坐标 (N, experiment) 和 3D 体素中心点
    """
    ones = np.ones((voxel_centers.shape[0], 1))
    voxel_homogeneous = np.hstack((voxel_centers[:, :3], ones))  # (N, 4)

    proj_points = (P_velo2im @ voxel_homogeneous.T).T  
    proj_points[:, :2] /= proj_points[:, 2, np.newaxis]  

    valid_mask = proj_points[:, 2] > 0  # 仅保留 Z > 0 的点
    pts_2d = proj_points[valid_mask, :2]
    voxels_filtered = voxel_centers[valid_mask]

    return pts_2d, voxels_filtered


if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Project LiDAR Voxels to Camera View")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)

    args = parser.parse_args()

    print("*********************************************")
    print("[i] Start LiDAR Voxel Projection")
    print("[i] Data root: ", args.data_root)
    print("[i] Mode: ", args.mode)
    print("[i] Sequence: ", args.seq)

    file_name = "x2021_0223_1310"
    # 设置存储路径
    save_root = os.path.join(args.data_root, "ORFD-custom", args.mode, "visual_voxels", args.seq)
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
            image = io.imread(image_path)

            # 读取 LiDAR 点云数据
            lidar_file = os.path.join(args.data_root, "Final_Dataset", args.mode, file_name, "lidar_data", str_frame + '.bin')
            lidar_data = bin_read(lidar_file)

            # 体素化 LiDAR 数据 (0.25m)
            voxel_centers, voxel_heights = voxelize_lidar(lidar_data, voxel_size=0.25)

            # 计算地面高度
            ground_height = estimate_ground_height_percentile(voxel_heights, percentile=20)

            # 计算物体高度
            object_heights = compute_voxel_heights(voxel_heights, ground_height)
            # print("[i] Object Heights: ", object_heights)
            # print("[i] Ground Height: ", ground_height)
            # 投影体素到相机
            pts_2d, voxels_filtered = project_voxels_to_image(voxel_centers, P_velo2im)
            coordinates_and_slope = []  # 用于存储 (x, y, rough_value)
            image_project = image.copy()
            
            for j, point_2d in enumerate(pts_2d):
                x, y = int(point_2d[0]), int(point_2d[1])
                slope = object_heights[j]  # 获取该点的坡度
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    coordinates_and_slope.append([x, y, slope])
                    color = (0, 255, 0) if slope  <= 1.5 else (0, 0, 255)  # 绿色 = ≤阈值，红色 = >阈值
                    cv2.circle(image_project, (x, y), 2, color, -1)  # 绘制圆圈标记
            np.save(os.path.join(save_root, "height_array" ,str_frame + ".npy"), coordinates_and_slope)
            save_path = os.path.join(save_root, "result" ,str_frame + ".png")
            save_image(image_project, save_path)

        
    print("[i] Finished LiDAR Voxel Projection")
    print("*********************************************")
