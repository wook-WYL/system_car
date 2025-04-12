import numpy as np
import cv2
import os
from scipy.interpolate import griddata

def generate_binary_mask_with_interpolation(npy_file, width=1280, height=720, threshold=1, min_area=7000, blur_kernel_size=5):
    # 读取存储在 .npy 文件中的坐标和坡度数据
    coordinates_and_slope = np.load(npy_file)
    # 创建一个黑色图像，尺寸为 1280x720
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建一个空的坡度矩阵（初始化为 NaN）
    slope_matrix = np.full((height, width), np.nan)
    
    # 将现有点的坡度值填入矩阵
    for x, y, slope in coordinates_and_slope:
        if 0 <= x < width and 0 <= y < height:
            slope_matrix[int(y), int(x)] = slope
    
    # 进行双线性插值：使用 scipy.griddata 插值函数
    known_points = np.array([(x, y) for x, y, slope in coordinates_and_slope if 0 <= x < width and 0 <= y < height and not np.isnan(slope_matrix[int(y), int(x)])])
    known_values = np.array([slope for x, y, slope in coordinates_and_slope if 0 <= x < width and 0 <= y < height and not np.isnan(slope_matrix[int(y), int(x)])])
    
    # 创建网格坐标，进行插值
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # 使用 griddata 进行插值，填充缺失值
    interpolated_slope = griddata(known_points, known_values, grid_points, method='linear')
    
    # 将插值结果填回到坡度矩阵
    slope_matrix = interpolated_slope.reshape((height, width))
    
    # 生成二值掩码：如果坡度小于等于阈值，则为白色，否则为黑色
    for y in range(height):
        for x in range(width):
            if not np.isnan(slope_matrix[y, x]):  # 如果该位置有有效插值
                if slope_matrix[y, x] <= threshold:
                    mask[y, x] = 255  # 白色
                else:
                    mask[y, x] = 0    # 黑色
    
    # 过滤掉小面积的区域
    # mask = cv2.dilate(mask, None, iterations=5)
    mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

    # 查找所有的白色区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 保留最大面积的白色区域
    max_area = 0
    max_contour = None
    for contour in contours:
        if cv2.contourArea(contour) > max_area:
            max_area = cv2.contourArea(contour)
            max_contour = contour

    # 清空掩码并填充最大白色区域
    mask[:] = 0
    if max_contour is not None:
        cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)  # 填充最大区域为白色

    # 最终确保二值化
    _, final_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return final_mask

def process_npy_files_in_directory(input_dir, output_dir, width=1280, height=720, threshold=1.5, blur_kernel_size=5):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取指定目录下的所有 .npy 文件
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for npy_file in npy_files:
        npy_file_path = os.path.join(input_dir, npy_file)

        # 调用函数生成二值掩码
        binary_mask = generate_binary_mask_with_interpolation(npy_file_path, width, height, threshold, blur_kernel_size=blur_kernel_size)

        # 定义输出图像路径
        output_image_path = os.path.join(output_dir, f"{os.path.splitext(npy_file)[0]}_binary_mask.png")

        # 保存二值掩码图像
        cv2.imwrite(output_image_path, binary_mask)
        print(f"Processed {npy_file}, saved binary mask to {output_image_path}")

# 测试代码
if __name__ == "__main__":
    input_dir = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score\16237330\score_array"  # 替换为实际的 .npy 文件所在的目录
    output_dir = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score\16237330\output_masks"  # 替换为实际的输出目录

    process_npy_files_in_directory(input_dir, output_dir)
