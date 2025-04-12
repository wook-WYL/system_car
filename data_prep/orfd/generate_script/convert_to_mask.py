import os
import numpy as np
import cv2
from scipy.interpolate import griddata
import argparse

def generate_binary_mask_with_interpolation(npy_file, width=1280, height=720, threshold=1, blur_kernel_size=5,thread_ad = 0.05):
    # 读取存储在 .npy 文件中的坐标和高度数据
    coordinates_and_height = np.load(npy_file)

    # 创建一个黑色图像，尺寸为 1280x720
    mask = np.zeros((height, width), dtype=np.uint8)

    # 创建一个空的高度矩阵（初始化为 NaN）
    height_matrix = np.full((height, width), np.nan)

    # 将现有点的高度值填入矩阵
    for x, y, height_value in coordinates_and_height:
        if 0 <= x < width and 0 <= y < height:
            height_matrix[int(y), int(x)] = height_value

    # 进行双线性插值：使用 scipy.griddata 插值函数
    known_points = np.array([(x, y) for x, y, height_value in coordinates_and_height if 0 <= x < width and 0 <= y < height and not np.isnan(height_matrix[int(y), int(x)])])
    known_values = np.array([height_value for x, y, height_value in coordinates_and_height if 0 <= x < width and 0 <= y < height and not np.isnan(height_matrix[int(y), int(x)])])

    # 创建网格坐标，进行插值
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # 使用 griddata 进行插值，填充缺失值
    interpolated_height = griddata(known_points, known_values, grid_points, method='linear')

    # 将插值结果填回到高度矩阵
    height_matrix = interpolated_height.reshape((height, width))

    # 生成二值掩码
    for y in range(height):

        for x in range(width):
            if not np.isnan(height_matrix[y, x]):  # 如果该位置有有效插值
                if height_matrix[y, x] <= threshold:
                    mask[y, x] = 255  # 白色
                else:
                    mask[y, x] = 0    # 黑色

    # 平滑掩码边缘（可选）
    mask = cv2.erode(mask, None, iterations=11)
    # mask = cv2.erode(mask, None, iterations=11)
    mask = cv2.erode(mask, None, iterations=7)
    mask = cv2.dilate(mask, None, iterations=7)
    mask = cv2.dilate(mask, None, iterations=11)
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

    # 简化轮廓并绘制平滑多边形
    if max_contour is not None:
        # 使用 cv2.approxPolyDP 简化轮廓，获取平滑多边形
        epsilon = thread_ad * cv2.arcLength(max_contour, True)  # 调整精度值
        approx_polygon = cv2.approxPolyDP(max_contour, epsilon, True)

        # 在掩码上绘制平滑多边形轮廓并填充
        cv2.drawContours(mask, [approx_polygon], -1, (255), thickness=cv2.FILLED)  # 使用FILLED参数填充多边形

    # 最终确保二值化
    _, final_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return final_mask


def process_all_npy_files(input_dir, output_dir, threshold=1.5, blur_kernel_size=5 , thread_ad = 0.01):
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    os.makedirs(output_dir, exist_ok=True)

    for npy_file in npy_files:
        npy_file_path = os.path.join(input_dir, npy_file)

        # 生成二值掩码

        binary_mask = generate_binary_mask_with_interpolation(npy_file_path, threshold=threshold, blur_kernel_size=blur_kernel_size,thread_ad = thread_ad)

        # 保存二值掩码图像
        output_image_path = os.path.join(output_dir, npy_file.replace('.npy', '.png'))
        cv2.imwrite(output_image_path, binary_mask)
        print(f"Processed and saved: {output_image_path}")


# 测试代码
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .npy files to generate binary masks.")
    parser.add_argument('--data_root', type=str,default="E:/BaiduNetdiskDownload/ORFD_Dataset_ICRA2022_ZIP/ORFD-custom/training/score1/")
    parser.add_argument('--threshold', type=float, default=1.0, help="Threshold value for height. Default is 1.0.")
    parser.add_argument('--blur_kernel_size', type=int, default=5, help="Size of the blur kernel. Default is 5.")
    parser.add_argument('--file_name', type=str, default="x0613_1627", help="Size")
    parser.add_argument('--thread_ad', type=float, default=0.01, help="Size") # 0.05 0.02
    # 解析命令行参数
    args = parser.parse_args()

    input_directory = os.path.join(args.data_root, args.file_name, "score_array")
    output_directory = os.path.join(args.data_root, args.file_name,"output_masks")


    # 处理所有 .npy 文件
    process_all_npy_files(input_directory, output_directory, threshold=args.threshold,
                          blur_kernel_size=args.blur_kernel_size , thread_ad = args.thread_ad)

    # 处理所有 .npy 文件
    # process_all_npy_files(input_directory, output_directory, threshold=0.005, blur_kernel_size=5)
    # process_all_npy_files(input_directory, output_directory, threshold=1.0, blur_kernel_size=5)
