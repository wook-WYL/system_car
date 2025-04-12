import cv2
import numpy as np
import os

# 定义文件夹路径
image_folder = r"J:\Dataset\ORFD\Training\training\image_data"
mask_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\predict\oldmodel_predict_training_ORFD_ALL\ckpts-model_best-ORFD\ckpts-model_best-ORFD"
output_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\predict\oldmodel_predict_result_orfd1_1"

# 获取文件夹中的所有图像文件（假设文件名对应）
image_files = sorted(os.listdir(image_folder))
mask_files = sorted(os.listdir(mask_folder))

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 红色路径掩码，设置透明度
red_color = [0, 0, 255]  # 红色 (BGR)
alpha = 0.3  # 透明度

# 遍历所有图像文件和掩码文件
for image_file, mask_file in zip(image_files, mask_files):
    # 构造图像和掩码路径
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, mask_file)

    # 读取原图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        continue

    # 读取掩码图像，转换为灰度图
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"错误: 无法读取掩码图像 {mask_path}")
        continue

    # 确保掩码图是二值化的
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 调整掩码图像大小，使其与原图相同
    binary_mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

    # 创建映射后的图像副本
    mapped_image = image.copy()

    # 找到掩码中的白色区域（值为 255）
    white_area = binary_mask_resized == 255

    # 将红色掩码投影到原图上，透明度为 0.1
    mapped_image[white_area] = (alpha * np.array(red_color) + (1 - alpha) * mapped_image[white_area]).astype(np.uint8)

    # 保存结果图像
    output_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_image_path, mapped_image)

    print(f"图像 {image_file} 已保存到 {output_image_path}")

print("所有图像处理完毕。")
