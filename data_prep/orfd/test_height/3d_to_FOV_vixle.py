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


def project_lidar_to_image(lidar_data, P_velo2im):
    """
    将 LiDAR 点云投影到相机图像
    :param lidar_data: (N, 3) LiDAR 3D 点云
    :param P_velo2im: (3,4) LiDAR -> 相机投影矩阵
    :return: 2D 投影坐标 (N, experiment)
    """
    # 添加齐次坐标
    ones = np.ones((lidar_data.shape[0], 1))
    lidar_homogeneous = np.hstack((lidar_data[:, :3], ones))  # (N, 4)

    # 计算投影坐标
    proj_points = (P_velo2im @ lidar_homogeneous.T).T  # (N, 3)
    proj_points[:, :2] /= proj_points[:, 2, np.newaxis]  # 归一化

    # 过滤 Z <= 0 的点（防止反向投影）
    valid_mask = proj_points[:, 2] > 0
    pts_2d = proj_points[valid_mask, :2]

    return pts_2d


if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Project LiDAR to Camera View")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--mode", type=str, help="ORFD mode: training/validation", required=True)
    parser.add_argument("--seq", type=str, help="Sequence name", required=True) # 16203310

    args = parser.parse_args()

    print("*********************************************")
    print("[i] Start LiDAR Projection")
    print("[i] Data root: ", args.data_root)
    print("[i] Mode: ", args.mode)
    print("[i] Sequence: ", args.seq)

    # 设置存储路径
    save_root = os.path.join(args.data_root, "ORFD-custom", args.mode, "visual", args.seq)
    os.makedirs(save_root, exist_ok=True)

    # 读取数据路径
    data_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "c2021_0228_1819", "image_data")
    file_list = natsort.natsorted(os.listdir(data_path))

    seq_dict = {}
    for fname in file_list:
        k = fname[:8]  # 提取前缀 (时间戳)
        if k not in seq_dict.keys():
            seq_dict[k] = []
        seq_dict[k].append(fname[:-4])  # 去掉 ".png"

    # 处理序列
    for seq_name in seq_dict.keys():
        if seq_name != args.seq:
            continue

        description = f"[i] Processing Sequence: {seq_name}"
        for i, str_frame in enumerate(tqdm(seq_dict[seq_name], desc=description)):

            # 读取相机标定参数
            calib_file = os.path.join(args.data_root, "Final_Dataset", args.mode, "c2021_0228_1819", "calib", str_frame + '.txt')
            P_velo2im = read_calib_file_orfd(calib_file)  # 直接获取 LiDAR -> 相机投影矩阵 (3x4)

            # 读取相机图像
            image_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "c2021_0228_1819", "image_data", str_frame + '.png')
            image = io.imread(image_path)

            # 读取 LiDAR 点云数据
            lidar_file = os.path.join(args.data_root, "Final_Dataset", args.mode, "c2021_0228_1819", "lidar_data", str_frame + '.bin')
            lidar_data = bin_read(lidar_file)

            # 投影 LiDAR 点到相机
            pts_2d = project_lidar_to_image(lidar_data, P_velo2im)

            # 画点
            image_project = image.copy()
            for j, point_2d in enumerate(pts_2d):
                x, y = int(point_2d[0]), int(point_2d[1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image_project, (x, y), 1, (0, 255, 0), -1)  # 绿色小点

            # 保存投影结果
            save_path = os.path.join(save_root, str_frame + ".png")
            save_image(image_project, save_path)

    print("[i] Finished LiDAR Projection")
    print("*********************************************")