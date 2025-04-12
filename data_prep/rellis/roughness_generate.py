# coding:utf-8
import os
import argparse
import natsort
import numpy as np
import open3d as o3d
from tqdm import tqdm
from numpy.linalg import norm, svd


def planeFit(points):
    """平面拟合函数"""
    points = np.transpose(points)
    points = np.reshape(points, (points.shape[0], -1))
    assert points.shape[0] <= points.shape[1], "Insufficient points for plane fitting"
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    return ctr, svd(M)[0][:, -1]


def compute_roughness(pc, pcd_tree, radius=0.5, k=50):
    """带统计的粗糙度计算"""
    roughness = np.full(len(pc), np.nan)
    valid_count = 0

    for pt_idx in range(len(pc)):
        [k_found, idx, _] = pcd_tree.search_hybrid_vector_3d(pc[pt_idx], radius, k)
        if k_found < 3:
            continue

        try:
            pt, normal = planeFit(pc[idx])
            normal = normal / norm(normal)
            plane_to_point = pc[pt_idx] - pt
            dist = abs(np.dot(normal, plane_to_point))
            roughness[pt_idx] = dist
            valid_count += 1
        except:
            pass

    return roughness, valid_count


def print_statistics(roughness, filename):
    """打印详细的统计信息"""
    valid_data = roughness[~np.isnan(roughness)]

    stats = {
        'Total Points': len(roughness),# 总点数
        'Valid Points': len(valid_data),# 有效点数
        'Invalid Points': np.isnan(roughness).sum(),# 无法计算的点数
        'Mean (mm)': np.mean(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',# 平均粗糙度
        'Std Dev (mm)': np.std(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',# 标准差
        'Max (mm)': np.max(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',# 最大粗糙度
        'Min (mm)': np.min(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',# 最小粗糙度
        'Median (mm)': np.median(valid_data) * 1000 if len(valid_data) > 0 else 'N/A',# 中位数
        '95th Percentile (mm)': np.percentile(valid_data, 95) * 1000 if len(valid_data) > 0 else 'N/A'# 95%的数据点粗糙度低于此值
    }

    # 构建输出信息
    output = [f"\n=== {filename} Statistics ==="]
    for k, v in stats.items():
        output.append(f"{k:<20}: {v}")
    output.append("")

    # 使用tqdm.write保持输出整洁
    tqdm.write("\n".join(output))


def process_sequence(data_root, seq, radius, k):
    """处理流程增强版"""
    seq_str = str(seq).zfill(5)
    save_root = os.path.join(data_root, "Rellis-3D-custom", seq_str, "roughness")
    os.makedirs(save_root, exist_ok=True)

    bin_path = os.path.join(data_root, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin")
    file_list = natsort.natsorted(os.listdir(bin_path))

    for fn in tqdm(file_list, desc=f"Processing Seq {seq}", unit="file"):
        # 读取数据
        pc = np.fromfile(os.path.join(bin_path, fn), dtype=np.float32)
        pc = pc.reshape(-1, 4)[:, :3]

        # 计算粗糙度
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        roughness, valid_count = compute_roughness(pc, pcd_tree, radius, k)

        # 打印统计信息
        print_statistics(roughness, fn)

        # 保存带统计信息的数据
        output = np.column_stack((pc, roughness))
        np.save(os.path.join(save_root, fn[:-4] + "_roughness.npy"), output)

        # 在进度条中显示简要统计
        tqdm.write(f"[{fn}] Valid: {valid_count}/{len(pc)} "
                   f"Avg: {np.nanmean(roughness) * 1000:.2f}mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版粗糙度生成器")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seqs", nargs='+', type=int, required=True)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=50)

    args = parser.parse_args()

    print("\n=== 粗糙度生成管道 ===")
    print(f"参数说明：\n"
          f"半径：控制局部区域大小（建议0.3-1.0m）\n"
          f"k值：控制最大邻域点数（建议30-100）\n"
          f"当前参数：radius={args.radius}m, k={args.k}\n")

    for seq in args.seqs:
        process_sequence(args.data_root, seq, args.radius, args.k)

    print("\n=== 处理完成 ===")