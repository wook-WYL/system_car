import cv2
import numpy as np

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

    # 确保掩码是二值图像（0或255）
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

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

# 使用示例
original_img_path = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\Final_Dataset\training\c2021_0228_1819\image_data\1620331020738.png"  # 修改为实际路径
mask_img_path = r'E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score1\c2021_0228_1819\output_masks\1620331020738.png'  # 修改为实际路径
output_path = r'E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score1\c2021_0228_1819\1620331020738.png'  # 修改为保存路径

# 调用函数，设置透明度（0到1之间，0为完全透明，1为不透明）
overlay_mask_on_image(original_img_path, mask_img_path, output_path, red_intensity=0.3)
