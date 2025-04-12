import numpy as np
import cv2
import os
from scipy.interpolate import griddata

def generate_binary_mask_with_interpolation(npy_file, width=1280, height=720, threshold=1000, blur_kernel_size=5):
    # 读取存储在 .npy 文件中的坐标和粗糙度数据
    coordinates_and_roughness = np.load(npy_file)
    
    # 创建一个黑色图像，尺寸为 1280x720
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建一个空的粗糙度矩阵（初始化为 NaN）
    roughness_matrix = np.full((height, width), np.nan)
    
    # 将现有点的粗糙度值填入矩阵
    for x, y, roughness in coordinates_and_roughness:
        if 0 <= x < width and 0 <= y < height:
            roughness_matrix[int(y), int(x)] = roughness
    
    # 进行双线性插值：使用 scipy.griddata 插值函数
    # 提取非NaN的点
    known_points = np.array([(x, y) for x, y, roughness in coordinates_and_roughness if 0 <= x < width and 0 <= y < height and not np.isnan(roughness_matrix[int(y), int(x)])])
    known_values = np.array([roughness for x, y, roughness in coordinates_and_roughness if 0 <= x < width and 0 <= y < height and not np.isnan(roughness_matrix[int(y), int(x)])])
    
    # 创建网格坐标，进行插值
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # 使用 griddata 进行插值，填充缺失值
    interpolated_roughness = griddata(known_points, known_values, grid_points, method='linear')
    
    # 将插值结果填回到粗糙度矩阵
    roughness_matrix = interpolated_roughness.reshape((height, width))
    
    # 生成二值掩码：如果粗糙度小于等于阈值，则为白色，否则为黑色
    for y in range(height):
        for x in range(width):
            if not np.isnan(roughness_matrix[y, x]):  # 如果该位置有有效插值
                if roughness_matrix[y, x] <= threshold:
                    mask[y, x] = 255  # 白色
                else:
                    mask[y, x] = 0    # 黑色
    
    # 过滤掉小面积的区域
    mask = cv2.erode(mask, None, iterations=7)
    mask = cv2.dilate(mask, None, iterations=7)
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


def process_all_npy_files(input_folder, output_folder):
    # 获取输入文件夹中的所有 .npy 文件
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    # 遍历每个 .npy 文件并生成二值掩码
    for npy_file in npy_files:
        npy_file_path = os.path.join(input_folder, npy_file)
        binary_mask = generate_binary_mask_with_interpolation(npy_file_path)
        
        # 保存二值掩码图像到输出文件夹
        output_image_path = os.path.join(output_folder, f'{os.path.splitext(npy_file)[0]}_binary_mask.png')
        cv2.imwrite(output_image_path, binary_mask)
        print(f"Processed and saved: {output_image_path}")


# 测试代码
if __name__ == "__main__":
    input_folder = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score\16237330\score_array"  # 替换为实际的输入文件夹路径
    output_folder = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\roughness\16237330\output_masks"  # 替换为实际的输出文件夹路径
    
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 处理所有的 .npy 文件并保存结果
    process_all_npy_files(input_folder, output_folder)
