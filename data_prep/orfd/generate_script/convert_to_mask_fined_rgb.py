import os
import numpy as np
import cv2
from scipy.interpolate import griddata
import argparse

def generate_colored_mask_with_interpolation(npy_file, width=1280, height=720, threshold=1, blur_kernel_size=5, thread_ad=0.05):
    # 读取存储在 .npy 文件中的坐标和高度数据
    coordinates_and_height = np.load(npy_file)

    # 创建一个全黑图像，尺寸为 1280x720
    mask = np.zeros((height, width, 3), dtype=np.uint8)  # 使用3通道来存储颜色

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

    # 根据高度值划分三个区间，分别赋予红色、绿色、蓝色
    for y in range(height):
        for x in range(width):
            if not np.isnan(height_matrix[y, x]):  # 如果该位置有有效插值
                threshold_aa = 0.8 # threshold_ab = 1.95
                threshold_ab = 1.5
                threshold_ac = 1.8
                if 0 <= height_matrix[y, x] <= threshold_aa:
                    # 红色区间 (0 ~ threshold/2 - 0.1)
                    mask[y, x] = [0, 0, 255]  # 红色
                elif threshold_aa < height_matrix[y, x] <= threshold_ab:
                    # 绿色区间 (threshold/2 ~ threshold - 0.1)
                    mask[y, x] = [255, 0, 0]  # 绿色
                # elif threshold_ab < height_matrix[y, x] <= threshold_ac:
                #     # 绿色区间 (threshold/2 ~ threshold - 0.1)
                #     mask[y, x] = [0, 255, 0] # 绿色
                else:
                    # 蓝色区间 (threshold - 0.1 ~ threshold)
                    mask[y, x] =   [0, 255, 0]  # 蓝色
    # [255, 0, 255],  # 红色 Mine
    # [0, 255, 0],  # 绿色 Pnpnet
    # [255, 165, 0],  # 棕色 RoadSeg
    # [0, 255, 255]  # 黄色 wayfast
    # 可选：对掩码进行平滑处理
    # mask = cv2.erode(mask, None, iterations=11)
    # # mask = cv2.erode(mask, None, iterations=11)
    # mask = cv2.erode(mask, None, iterations=7)
    # mask = cv2.dilate(mask, None, iterations=7)
    # mask = cv2.dilate(mask, None, iterations=11)
    # mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

    return mask

def overlay_mask_on_image(original_img_path, mask_img, output_path, red_intensity=0.1, green_intensity=0.1, blue_intensity=0.1):
    # 读取原图
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Error: Unable to load image at {original_img_path}")
        return

    # 创建透明的红色、绿色、蓝色层
    red_layer = np.zeros_like(original_img, dtype=np.uint8)
    red_layer[:, :, 2] = 255  # 红色通道设为255，表示红色
    green_layer = np.zeros_like(original_img, dtype=np.uint8)
    green_layer[:, :, 1] = 255  # 绿色通道设为255，表示绿色
    blue_layer = np.zeros_like(original_img, dtype=np.uint8)
    blue_layer[:, :, 0] = 255  # 蓝色通道设为255，表示蓝色

    # 提取掩码中的红色、绿色和蓝色部分
    mask_red = mask_img[:, :, 2]  # 提取掩码的红色通道部分
    mask_green = mask_img[:, :, 1]  # 提取掩码的绿色通道部分
    mask_blue = mask_img[:, :, 0]  # 提取掩码的蓝色通道部分

    # 将红色、绿色、蓝色层与原图合成，透明度由相应的 intensity 控制
    red_overlay = cv2.addWeighted(original_img, 1 - red_intensity, red_layer, red_intensity, 0)
    green_overlay = cv2.addWeighted(original_img, 1 - green_intensity, green_layer, green_intensity, 0)
    blue_overlay = cv2.addWeighted(original_img, 1 - blue_intensity, blue_layer, blue_intensity, 0)

    # 使用掩码提取红色、绿色、蓝色区域并应用透明度
    result_img_red = cv2.bitwise_and(red_overlay, red_overlay, mask=mask_red)
    result_img_green = cv2.bitwise_and(green_overlay, green_overlay, mask=mask_green)
    result_img_blue = cv2.bitwise_and(blue_overlay, blue_overlay, mask=mask_blue)

    # 使用反掩码提取剩余部分（即没有被叠加的区域）
    remaining_img = cv2.bitwise_and(original_img, original_img, mask=cv2.bitwise_not(mask_red))

    # 合并所有部分
    final_img = cv2.add(result_img_red, result_img_green)
    final_img = cv2.add(final_img, result_img_blue)
    final_img = cv2.add(final_img, remaining_img)

    # 保存或显示结果图像
    cv2.imwrite(output_path, final_img)
    print(f"Saved the result to {output_path}")

def process_all_npy_files(input_dir, output_dir, threshold=1.5, blur_kernel_size=5, thread_ad=0.01):
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    os.makedirs(output_dir, exist_ok=True)

    for npy_file in npy_files:
        npy_file_path = os.path.join(input_dir, npy_file)

        # 生成带有红、绿、蓝区域的掩码
        colored_mask = generate_colored_mask_with_interpolation(npy_file_path, threshold=threshold, blur_kernel_size=blur_kernel_size, thread_ad=thread_ad)

        # 生成原图路径
        original_img_path = f"E:/BaiduNetdiskDownload/ORFD_Dataset_ICRA2022_ZIP/Final_Dataset/training/y0616_1950/image_data/{npy_file.replace('.npy', '.png')}"
        # 生成输出路径
        output_image_path = os.path.join(output_dir, npy_file.replace('.npy', '_overlay.png'))
        # 使用掩码进行可视化叠加
        overlay_mask_on_image(original_img_path, colored_mask, output_image_path, red_intensity=0.5, green_intensity=0.5, blue_intensity=0.5)

        print(f"Processed and saved: {output_image_path}")


# 测试代码
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .npy files to generate colored masks and overlay them on images.")
    parser.add_argument('--data_root', type=str, default="E:/BaiduNetdiskDownload/ORFD_Dataset_ICRA2022_ZIP/ORFD-custom/training/score1/")
    parser.add_argument('--threshold', type=float, default=1.0, help="Threshold value for height. Default is 1.0.")
    parser.add_argument('--blur_kernel_size', type=int, default=5, help="Size of the blur kernel. Default is 5.")
    parser.add_argument('--file_name', type=str, default="c2021_0228_1819", help="Size")
    parser.add_argument('--thread_ad', type=float, default=0.01, help="Size")  # 0.05 0.02

    # 解析命令行参数
    args = parser.parse_args()

    input_directory = os.path.join(args.data_root, args.file_name, "score_array")
    output_directory = os.path.join(args.data_root, args.file_name, "output_masks_rgb")

    # 处理所有 .npy 文件
    process_all_npy_files(input_directory, output_directory, threshold=args.threshold,
                          blur_kernel_size=args.blur_kernel_size, thread_ad=args.thread_ad)
