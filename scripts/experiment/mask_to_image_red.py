import cv2
import numpy as np
import os
import glob


def overlay_mask_on_image(original_img_path, mask_img_path, output_path, red_intensity=0.5):
    # 读取原图并检查是否成功加载
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Error: Unable to load image at {original_img_path}")
        return

    # 读取掩码图像并检查是否成功加载
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"Error: Unable to load mask image at {mask_img_path}")
        return

    # 调整掩码图像的大小为原图大小
    resized_mask = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 确保掩码是二值图像（0或255）
    _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    # 创建一个与原图相同大小的红色层
    red_layer = np.zeros_like(original_img, dtype=np.uint8)
    red_layer[:, :, 2] = 255  # 红色通道设为255，表示红色

    # 将红色层与原图合成，透明度为red_intensity
    red_overlay = cv2.addWeighted(original_img, 1 - red_intensity, red_layer, red_intensity, 0)

    # 将掩码区域（白色部分）应用到红色层，其他区域保持原图
    result_img = cv2.bitwise_and(red_overlay, red_overlay, mask=binary_mask)
    remaining_img = cv2.bitwise_and(original_img, original_img, mask=cv2.bitwise_not(binary_mask))

    # 将两部分合并，得到最终的图像
    final_img = cv2.add(result_img, remaining_img)

    # 保存或显示结果图像
    cv2.imwrite(output_path, final_img)
    print(f"Saved the result to {output_path}")


def process_images_in_folder(input_folder, mask_folder, output_folder, red_intensity=0.5):
    # 获取所有图片文件路径，支持多种图片格式
    image_paths = glob.glob(os.path.join(input_folder, "*.*"))  # 支持多种格式

    for img_path in image_paths:
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 只处理图像文件
            continue

        # 从图片文件名提取对应的掩码文件路径
        mask_path = os.path.join(mask_folder, os.path.basename(img_path))

        if os.path.exists(mask_path):
            # 生成输出路径
            output_path = os.path.join(output_folder, os.path.basename(img_path))

            # 调用叠加函数
            overlay_mask_on_image(img_path, mask_path, output_path, red_intensity)
        else:
            print(f"Mask for {img_path} not found, skipping.")


# 输入文件夹路径（原图路径）、掩码文件夹路径和输出文件夹路径
# input_folder = r"E:\BaiduNetdiskDownload\video\video2"
# mask_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\CKPT\ORFD\oldmodel\3c\model_best_2"
# output_folder = r"E:\BaiduNetdiskDownload\video\video2_red"
# video1
# input_folder = r"E:\BaiduNetdiskDownload\video\video3_resize"
# mask_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\CKPT\ORFD\oldmodel\3c\1111"
# output_folder = r"E:\BaiduNetdiskDownload\video\video3_red1"

# input_folder = r"E:\BaiduNetdiskDownload\video\video3_resize"
# mask_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\CKPT\ORFD\oldmodel\3c\11111\root"
# output_folder = r"E:\BaiduNetdiskDownload\video\video3_red11"

# video2
# input_folder = r"E:\BaiduNetdiskDownload\video\video2"
# mask_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\CKPT\ORFD\oldmodel\3c\2222"
# output_folder = r"E:\BaiduNetdiskDownload\video\video2_red"

# # real v1
input_folder = r"H:\IROS\Video\video\Video1\RGB_Depth_Custom\RGB"
mask_folder = r"C:\Users\Administrator\Desktop\ckpts-model_best-ORFD"
output_folder = r"H:\IROS\Video\video\Video1\mask_output\red_mask2"

# # real v2
# input_folder = r"E:\BaiduNetdiskDownload\IROS\Video\video\Video2\RGB_Depth_Custom\RGB"
# mask_folder = r"E:\BaiduNetdiskDownload\IROS\Video\video\Video2\mask_output\mask"
# output_folder = r"E:\BaiduNetdiskDownload\IROS\Video\video\Video2\mask_output\red_mask"

# input_folder = r"H:\dataset\ORFD\ORFD_Dataset_ICRA2022_ZIP\Final_Dataset\training\c2021_0228_1819\image_data"
# mask_folder = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score1\c2021_0228_1819\output_masks"
# output_folder = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score1\c2021_0228_1819\red_masks"

# input_folder = r"E:\BaiduNetdiskDownload\video\video2"
# mask_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\CKPT\ORFD\offnet\ckpts-model_best_7_V2-ORFD"
# output_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\CKPT\ORFD\offnet\video2_red"

# 202531
# input_folder = r"E:\BaiduNetdiskDownload\video\video202531\rgb_202531"
# mask_folder = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\CKPT\ORFD\oldmodel\1c\2-model_best-ORFD"
# output_folder = r"E:\BaiduNetdiskDownload\video\video202531\rgb_202531_red"

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 调用函数批量处理
process_images_in_folder(input_folder, mask_folder, output_folder, red_intensity=0.3)
