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

# 计算评分函数
# def compute_score(slope, height, roughness, s_crit=10, h_crit=1, r_crit=0.01, omega1=1.experiment, omega2=0.5, omega3=1):
# def compute_score(slope, height, roughness, s_crit=8, h_crit=1.5, r_crit=0.005, omega1=1, omega2=0.5, omega3=1):
#     """
#     根据坡度、物体高度和粗糙度计算评分
#     :param slope: 坡度值
#     :param height: 物体高度值
#     :param roughness: 粗糙度值
#     :param s_crit: 坡度阈值
#     :param h_crit: 物体高度阈值
#     :param r_crit: 粗糙度阈值
#     :param omega1, omega2, omega3: 权重参数
#     :return: 计算得到的评分
#     """
#     # 调整权重，注重坡度和粗糙度
#     score = omega1 * (1 / (1 + np.exp(-50*(slope - s_crit)))) + \
#            omega2 * (1 / (1 + np.exp(-1 * (height - h_crit)))) + \
#            omega3 * (1 / (1 + np.exp(-50 * (roughness - r_crit))))
    
#     return score
def compute_score(slope, height, roughness, s_crit=10, h_crit=1.5, r_crit=0.005, omega1=1, omega2=0.5, omega3=0.8):
    """
    根据坡度、物体高度和粗糙度计算评分
    :param slope: 坡度值
    :param height: 物体高度值
    :param roughness: 粗糙度值
    :param s_crit: 坡度阈值
    :param h_crit: 物体高度阈值
    :param r_crit: 粗糙度阈值
    :param omega1, omega2, omega3: 权重参数
    :return: 计算得到的评分
    """
    # 调整权重，注重坡度和粗糙度
    score = omega1 * (1 / (1 + np.exp(-50*(slope - s_crit)))) + \
           omega2 * (1 / (1 + np.exp(-1 * (height - h_crit)))) + \
           omega3 * (1 / (1 + np.exp(-50 * (roughness - r_crit))))
    
    return score

def compute_slope(normals):
    """计算坡度，基于法线向量的z分量"""
    slopes = []
    for normal in normals:
        # 法线向量的 z 分量
        z_normal = normal[2]
        
        # 防止 z_normal 超出 [-1, 1] 范围
        z_normal = max(-1.0, min(1.0, z_normal))
        
        # 计算坡度，使用反余弦计算角度，弧度转角度
        slope = math.acos(z_normal) * 180.0 / math.pi  # 转换为角度
        slopes.append(slope)
    return np.array(slopes)

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


import numpy as np
# 1️⃣ 体素化 LiDAR 点云（0.25m 分辨率）
def voxelize_lidar(lidar_points, roughness, voxel_size=0.25):
    """
    对 LiDAR 点云进行体素化（分辨率 0.25m），同时计算每个体素的粗糙度的平均值
    :param lidar_points: (N, 3) 点云数据
    :param roughness: (N,) 对应每个点的粗糙度
    :param voxel_size: 体素大小 (默认 0.25m)
    :return: 体素中心点 & 每个体素的平均粗糙度
    """
    # 计算每个点的体素索引
    voxel_indices = np.floor(lidar_points[:, :3] / voxel_size).astype(int)

    # 用字典存储体素的最大 Z 值和所有粗糙度数据
    voxel_dict = {}
    for idx, voxel in enumerate(voxel_indices):
        key = tuple(voxel)  # 体素索引作为键
        if key in voxel_dict:
            voxel_dict[key]['points'].append(lidar_points[idx])
            voxel_dict[key]['roughness'].append(roughness[idx])
        else:
            voxel_dict[key] = {'points': [lidar_points[idx]], 'roughness': [roughness[idx]]}

    # 计算每个体素的中心点和平均粗糙度
    voxel_centers = []
    voxel_roughness_avg = []

    for key in voxel_dict.keys():
        # 计算体素中心点
        voxel_center = np.mean(voxel_dict[key]['points'], axis=0)
        # 计算体素内点的粗糙度的平均值
        avg_roughness = np.mean(voxel_dict[key]['roughness'])

        voxel_centers.append(voxel_center)
        voxel_roughness_avg.append(avg_roughness)

    voxel_centers = np.array(voxel_centers)
    voxel_roughness_avg = np.array(voxel_roughness_avg)

    return voxel_centers, voxel_roughness_avg

def voxelize_lidar2(lidar_points, voxel_size=0.25):
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

# 计算评分函数
def compute_score(slope, height, roughness, s_crit=10, h_crit=1.5, r_crit=0.005, omega1=1, omega2=0.5, omega3=0.8):
    """
    根据坡度、物体高度和粗糙度计算评分
    :param slope: 坡度值
    :param height: 物体高度值
    :param roughness: 粗糙度值
    :param s_crit: 坡度阈值
    :param h_crit: 物体高度阈值
    :param r_crit: 粗糙度阈值
    :param omega1, omega2, omega3: 权重参数
    :return: 计算得到的评分
    """
    # 调整权重，注重坡度和粗糙度
    score = omega1 * (1 / (1 + np.exp(-50*(slope - s_crit)))) + \
           omega2 * (1 / (1 + np.exp(-1 * (height - h_crit)))) + \
           omega3 * (1 / (1 + np.exp(-50 * (roughness - r_crit))))

    return score

# 检查文件是否存在并且完整
def check_file_integrity(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Project LiDAR Points to Camera View")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--score_threshold", type=float, default=1.0, help="Score")
    parser.add_argument("--file_name", type=str, default="x0613_1627", help="Score")

    args = parser.parse_args()

    print("*********************************************")
    print("[i] Start LiDAR Projection")
    print("[i] Data root: ", args.data_root)
    print("[i] Mode: ", args.mode)
    print("[i] Sequence: ", args.seq)

    # 设置存储路径
    save_root = os.path.join(args.data_root, "ORFD-custom", args.mode, "score", args.seq)
    os.makedirs(save_root, exist_ok=True)
    
    data_path = os.path.join(args.data_root, "Final_Dataset", args.mode, args.file_name, "image_data")
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
            
            # 生成保存路径
            score_file_path = os.path.join(save_root, "score_array", str_frame + ".npy")
            image_file_path = os.path.join(save_root, "result", str_frame + ".png")
            image_file_path1 = os.path.join(save_root, "result1", str_frame + ".png")
            image_file_path2 = os.path.join(save_root, "result2", str_frame + ".png")
            image_file_path3 = os.path.join(save_root, "result3", str_frame + ".png")

            # 如果评分文件已经存在并且完整，跳过处理
            if check_file_integrity(score_file_path) and \
               check_file_integrity(image_file_path) and \
               check_file_integrity(image_file_path1) and \
               check_file_integrity(image_file_path2) and \
               check_file_integrity(image_file_path3):
                print(f"[i] Skipping {str_frame}, files already processed.")
                continue

            # 读取相机标定参数
            calib_file = os.path.join(args.data_root, "Final_Dataset", args.mode, args.file_name, "calib", str_frame + '.txt')
            P_velo2im = read_calib_file_orfd(calib_file)  

            # 读取相机图像
            image_path = os.path.join(args.data_root, "Final_Dataset", args.mode, args.file_name, "image_data", str_frame + '.png')
            image = cv2.imread(image_path)

            # 读取 LiDAR 点云数据
            lidar_file = os.path.join(args.data_root, "Final_Dataset", args.mode, args.file_name, "lidar_data", str_frame + '.bin')
            lidar_data = bin_read(lidar_file)

            # 计算粗糙度
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_data)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            roughness, valid_data, valid_count = compute_roughness(lidar_data, pcd_tree, radius=0.5, k=50)
            
            print_stats(roughness, valid_data)

            # 体素化 LiDAR 点云并计算每个体素的粗糙度平均值
            voxel_centers, voxel_roughness_avg = voxelize_lidar(lidar_data, roughness, voxel_size=0.25)
            
            voxel_centers1, voxel_heights = voxelize_lidar2(lidar_data, voxel_size=0.25)

             # 计算地面高度
            ground_height = estimate_ground_height_percentile(voxel_heights, percentile=20)

            # 计算物体高度
            object_heights = compute_voxel_heights(voxel_heights, ground_height)
            print("[i] Object Heights: ", object_heights)
            print("[i] Ground Height: ", ground_height)
            
            # 投影点到相机
            pts_2d, _ = project_points_to_image(voxel_centers, P_velo2im)

            # 计算法线并估算坡度
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(voxel_centers1)
            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=50))
            normals_ = np.array(source_pcd.normals)
            slopes = compute_slope(normals_)

            # 创建保存目录
            os.makedirs(os.path.join(save_root, "score_array"), exist_ok=True)
            os.makedirs(os.path.join(save_root, "result"), exist_ok=True)
            # os.makedirs(os.path.join(save_root, "result1"), exist_ok=True)
            # os.makedirs(os.path.join(save_root, "result2"), exist_ok=True)
            # os.makedirs(os.path.join(save_root, "result3"), exist_ok=True)

            image_project = image.copy()
            image_project1 = image.copy()
            image_project2 = image.copy()
            image_project3 = image.copy()

            score_array = []
            for j, point_2d in enumerate(pts_2d):
                x, y = int(point_2d[0]), int(point_2d[1])
                rough_value = voxel_roughness_avg[j]  # 获取该体素的平均粗糙度
                slope = slopes[j]  # 获取该点的坡度
                heigh = object_heights[j]  # 获取该点的高度

                # Ensure the point is within image bounds and calculate score, color, and display
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    score_array.append([x, y, compute_score(slope, heigh, rough_value)])
                    score_value  =  compute_score(slope, heigh, rough_value)
                    
                    color = (0, 255, 0) if score_value <= args.score_threshold else (0, 0, 255)  # 绿色 = ≤阈值，红色 = >阈值
                    cv2.circle(image_project, (x, y), 2, color, -1)  # 绘制圆圈标记
                    color1 = (0, 255, 0) if heigh <= 1.5 else (0, 0, 255)  # 绿色 = ≤阈值，红色 = >阈值
                    cv2.circle(image_project1, (x, y), 2, color1, -1)  # 绘制圆圈标记
                    color2 = (0, 255, 0) if rough_value <= 0.005 else (0, 0, 255)  # 绿色 = ≤阈值，红色 = >阈值
                    cv2.circle(image_project2, (x, y), 2, color2, -1)  # 绘制圆圈标记
                    color3 = (0, 255, 0) if slope <= 10 else (0, 0, 255)  # 绿色 = ≤阈值，红色 = >阈值
                    cv2.circle(image_project3, (x, y), 2, color3, -1)  # 绘制圆圈标记

            # 保存计算结果
            save_image(image_project, image_file_path)
            save_image(image_project1, image_file_path1)
            save_image(image_project2, image_file_path2)
            save_image(image_project3, image_file_path3)
            np.save(score_file_path, score_array)
