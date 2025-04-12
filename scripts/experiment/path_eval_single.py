import numpy as np
import math
from scipy.interpolate import interp1d


# 读取npy文件
def load_path(file_path):
    return np.load(file_path)


# 将像素坐标转换为米坐标
def convert_to_meters(path, meter_to_pixel):
    return path / meter_to_pixel


# 计算CTE：返回路径中每个点的横向误差，单位为米
def compute_cte(target_path, generated_path, meter_to_pixel):
    cte_values = []

    # 确保路径点数一致，若不一致则进行插值（这里使用线性插值）
    if len(target_path) != len(generated_path):
        # 通过插值使目标路径和生成路径具有相同的点数
        interpolator = interp1d(np.arange(len(generated_path)), generated_path, kind='linear', axis=0)
        generated_path = interpolator(np.linspace(0, len(generated_path) - 1, len(target_path)))

    # 计算横向误差（在像素单位下）
    for i in range(len(target_path)):
        target_point = target_path[i]
        generated_point = generated_path[i]

        if np.isnan(target_point[:2]).any() or np.isnan(generated_point[:2]).any():
            continue

        # 计算欧几里得距离（像素单位）
        cte_pixel = np.linalg.norm(target_point[:2] - generated_point[:2])

        # 将像素距离转换为米
        cte_meter = cte_pixel / meter_to_pixel
        cte_values.append(cte_meter)
    return cte_values


# 计算HD：计算生成路径的方向与目标路径的方向之间的角度差，单位为弧度
def compute_hd(target_path, generated_path, meter_to_pixel):
    hd_values = []

    # 将路径坐标从像素转换为米
    target_path = convert_to_meters(target_path, meter_to_pixel)
    generated_path = convert_to_meters(generated_path, meter_to_pixel)

    # 计算方向偏差
    for i in range(1, min(len(target_path), len(generated_path))):
        target_point_1 = target_path[i - 1]
        target_point_2 = target_path[i]
        generated_point_1 = generated_path[i - 1]
        generated_point_2 = generated_path[i]

        # 计算目标路径和生成路径的方向向量
        target_direction = target_point_2[:2] - target_point_1[:2]
        generated_direction = generated_point_2[:2] - generated_point_1[:2]

        # 计算两向量之间的夹角
        norm_target = np.linalg.norm(target_direction)
        norm_generated = np.linalg.norm(generated_direction)

        # 检查方向向量的长度是否为零，避免除零错误
        if norm_target == 0 or norm_generated == 0:
            continue  # 如果为零，则跳过此点

        dot_product = np.dot(target_direction, generated_direction)
        cos_theta = dot_product / (norm_target * norm_generated)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差导致的超出范围
        angle = np.arccos(cos_theta)  # 计算角度，单位为弧度

        hd_values.append(angle)

    return hd_values


def compute_tr(target_path, generated_path, threshold, meter_to_pixel):
    recalled_points = 0

    # 打印路径的首尾坐标
    print(f"Target Path Start: {target_path[0]}, Target Path End: {target_path[-1]}")
    print(f"Generated Path Start: {generated_path[0]}, Generated Path End: {generated_path[-1]}")

    # 只提取生成路径的前两列（x, y）
    generated_path = generated_path[:, :2]

    # 确保路径点数一致，若不一致则进行插值（这里使用线性插值）
    if len(target_path) != len(generated_path):
        # 通过插值使目标路径和生成路径具有相同的点数
        interpolator = interp1d(np.arange(len(generated_path)), generated_path, kind='linear', axis=0)
        generated_path = interpolator(np.linspace(0, len(generated_path)-1, len(target_path)))

    # 遍历目标路径中的每个点
    for target_point in target_path:
        # 计算与目标路径点的最小距离
        distances = np.linalg.norm(generated_path - target_point[:2], axis=1)  # 只计算 x, y 坐标


        # 将像素距离转换为米
        distances_in_meters = distances / meter_to_pixel

        # 如果与生成路径中的某个点的距离小于阈值，认为该点被召回
        if np.min(distances_in_meters) < threshold:  # 转换为米
            recalled_points += 1

    # 计算召回率
    recall_rate = recalled_points / len(target_path) if len(target_path) > 0 else 0
    return recall_rate

# 加载路径
target_path_file = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\path\gt\map\400_9004\400_900_footprint_path.npy"  # 目标路径
generated_path_file = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\path\result\Mine_Network\00004\400_900.npy"  # 生成路径

target_path = load_path(target_path_file)
generated_path = load_path(generated_path_file)

# meter_to_pixel 为 10，表示每 10 像素对应 1 米; 1 -> 0.1m
meter_to_pixel = 10
threshold = 1.5  # 设置阈值为1米

# 计算CTE和HD
cte_values = compute_cte(target_path, generated_path, meter_to_pixel)
hd_values = compute_hd(target_path, generated_path, meter_to_pixel)

# 计算TR（轨迹召回率）
recall_rate = compute_tr(target_path, generated_path, threshold, meter_to_pixel)

# 输出结果
print("CTE Values (in meters):", cte_values)
print("HD Values (in radians):", hd_values)

# 计算 CTE 和 HD 的平均值（作为最终评估指标）
mean_cte = np.mean(cte_values) if cte_values else 0
mean_hd = np.mean(hd_values) if hd_values else 0

print(f"Mean CTE (in meters): {mean_cte:.4f}")
print(f"Mean HD (in radians): {mean_hd:.4f}")
print(f"Trajectory Recall (TR): {recall_rate:.4f}")
