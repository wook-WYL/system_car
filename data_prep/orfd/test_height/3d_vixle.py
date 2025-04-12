# import os
# import numpy as np
# import argparse
# import natsort
# import open3d as o3d
# from tqdm import tqdm
#
# import sys
# sys.path.append('./')
#
# from common.utils_loader import read_calib_file_orfd, bin_read
#
#
# def visualize_voxel_grid(lidar_data, voxel_size=0.25):
#     """
#     可视化 LiDAR 点云的体素网格
#     :param lidar_data: (N, 3) LiDAR 3D 点云
#     :param voxel_size: 体素大小
#     """
#     # 创建 Open3D 点云对象
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])
#
#     # 体素化点云
#     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
#
#     # 可视化体素网格
#     o3d.visualization.draw_geometries([voxel_grid])
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Visualize LiDAR Voxel Grid")
#     parser.add_argument("--data_root", type=str, help="Data root", required=True)
#     parser.add_argument("--mode", type=str, help="ORFD mode: training/validation", required=True)
#     args = parser.parse_args()
#
#     print("*********************************************")
#     print("[i] Start LiDAR Voxel Visualization")
#     print("[i] Data root: ", args.data_root)
#     print("[i] Mode: ", args.mode)
#
#     # 读取 LiDAR 数据路径
#     lidar_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "c2021_0228_1819", "lidar_data")
#     file_list = natsort.natsorted(os.listdir(lidar_path))
#
#     seq_dict = {}
#     for fname in file_list:
#         k = fname[:8]  # 提取前缀 (时间戳)
#         if k not in seq_dict.keys():
#             seq_dict[k] = []
#         seq_dict[k].append(fname[:-4])  # 去掉 ".bin"
#
#     # 遍历所有序列
#     for seq_name in seq_dict.keys():
#         for i, str_frame in enumerate(tqdm(seq_dict[seq_name], desc=f"[i] Processing Sequence: {seq_name}")):
#
#             # 读取 LiDAR 点云数据
#             lidar_file = os.path.join(lidar_path, str_frame + '.bin')
#             lidar_data = bin_read(lidar_file)
#
#             # 可视化体素网格
#             visualize_voxel_grid(lidar_data)
#
#     print("[i] Finished LiDAR Voxel Visualization")
#     print("*********************************************")
import os
import numpy as np
import argparse
import natsort
import open3d as o3d
from tqdm import tqdm
import cv2  # 用于读取RGB图像

import sys

sys.path.append('./')

from common.utils_loader import read_calib_file_orfd, bin_read


def visualize_voxel_grid_with_color(lidar_data, rgb_image, voxel_size=0.25):
    """
    可视化 LiDAR 点云的体素网格，并为每个体素点赋予颜色
    :param lidar_data: (N, 3) LiDAR 3D 点云
    :param rgb_image: 对应的RGB图像
    :param voxel_size: 体素大小
    """
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])

    # 计算每个点的颜色 (RGB)
    # 假设LiDAR点云和RGB图像的尺寸匹配
    colors = []
    for point in lidar_data[:, :3]:
        # 将3D LiDAR坐标投影到2D图像平面 (假设已经通过校准获得投影公式)
        # 这里只是简单地用X和Y坐标作为示例，可以根据实际需求进行计算
        u = int(np.clip(point[0], 0, rgb_image.shape[1] - 1))
        v = int(np.clip(point[1], 0, rgb_image.shape[0] - 1))

        # 获取RGB图像中的像素值
        color = rgb_image[v, u] / 255.0  # 归一化到[0, 1]
        colors.append(color)

    # 转换颜色列表为Open3D的颜色格式
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # 体素化点云
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    # 可视化体素网格
    o3d.visualization.draw_geometries([voxel_grid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LiDAR Voxel Grid with RGB")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--mode", type=str, help="ORFD mode: training/validation", required=True)
    args = parser.parse_args()

    print("*********************************************")
    print("[i] Start LiDAR Voxel Visualization with RGB")
    print("[i] Data root: ", args.data_root)
    print("[i] Mode: ", args.mode)

    # 读取 LiDAR 数据路径
    lidar_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "c2021_0228_1819", "lidar_data")
    file_list = natsort.natsorted(os.listdir(lidar_path))

    seq_dict = {}
    for fname in file_list:
        k = fname[:8]  # 提取前缀 (时间戳)
        if k not in seq_dict.keys():
            seq_dict[k] = []
        seq_dict[k].append(fname[:-4])  # 去掉 ".bin"

    # 遍历所有序列
    for seq_name in seq_dict.keys():
        for i, str_frame in enumerate(tqdm(seq_dict[seq_name], desc=f"[i] Processing Sequence: {seq_name}")):
            # 读取 LiDAR 点云数据
            lidar_file = os.path.join(lidar_path, str_frame + '.bin')
            lidar_data = bin_read(lidar_file)

            # 假设RGB图像与LiDAR数据具有相同的时间戳
            # 这里的文件路径和命名方式需要根据实际情况调整
            rgb_image_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "c2021_0228_1819", "image_data",
                                          str_frame + '.png')
            rgb_image = cv2.imread(rgb_image_path)

            # 可视化体素网格，带有RGB颜色
            visualize_voxel_grid_with_color(lidar_data, rgb_image)

    print("[i] Finished LiDAR Voxel Visualization with RGB")
    print("*********************************************")
