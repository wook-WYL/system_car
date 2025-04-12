import cv2
import numpy as np
import os

def overlay_mask_on_image(original_img_path, mask_img_path, output_path, red_intensity=0.5):
    """
    将掩码图像叠加到原图上，产生红色透明叠加效果。

    :param original_img_path: 原图的路径
    :param mask_img_path: 掩码图像的路径
    :param output_path: 输出结果图像的保存路径
    :param red_intensity: 红色叠加的强度（0-1之间）
    """
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

    # 确保掩码是二值图像（0或255）
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    # 确保掩码和原图大小一致，如果不一致则调整掩码图像的大小
    if binary_mask.shape != original_img.shape[:2]:
        print(f"Warning: Mask size does not match image size. Resizing mask to match image.")
        binary_mask = cv2.resize(binary_mask, (original_img.shape[1], original_img.shape[0]))

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

def main():
    # 输入单张图片的路径
    original_img_path = r"J:\Dataset\ORFD\Training\training\image_data\1620497380127.png"  # 修改为你的图片路径
    # original_img_path = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\Final_Dataset\testing\image_data\1620497380127.png"  # 修改为你的图片路径

    # 输入单张掩码图像的路径
    mask_img_path = r"E:\BaiduNetdiskDownload\footprint\report\1620497381726.png"  # 修改为你的掩码路径

    # 输出文件夹路径
    output_folder = r"E:\BaiduNetdiskDownload\footprint\report_mask"  # 修改为你的输出文件夹路径

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 从图片文件名提取输出文件路径
    output_path = os.path.join(output_folder, os.path.basename(original_img_path))

    # 叠加掩码到图像上
    overlay_mask_on_image(original_img_path, mask_img_path, output_path, red_intensity=0.3)

if __name__ == "__main__":
    main()
