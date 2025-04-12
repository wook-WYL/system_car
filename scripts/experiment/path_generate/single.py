import cv2
import numpy as np

# 输入文件路径
bev_image_path = r"H:\IROS\Video\video\Path_Generate_Video\00004\wayfast\400_900_ckpts-model_best-RELLIS_3D_00_38_21_trajectory_overlay.png"
mask_image_path = r"H:\IROS\Ablation_Study\path\gt\map\400_9004\400_900_footprint_xiugai.png"

# 红色路径掩码
red_color = [255, 0, 0]  # 红色 (BGR)

# 读取BEV图像
bev_image = cv2.imread(bev_image_path)
if bev_image is None:
    print(f"错误: 无法读取 BEV 图像 {bev_image_path}")
    exit()

# 读取掩码图像，灰度图
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
if mask_image is None:
    print(f"错误: 无法读取掩码图像 {mask_image_path}")
    exit()

# 确保掩码图是二值化的
_, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

# 获取掩码中的黑色部分，路径通常是黑色线条，值为0
path_mask = binary_mask == 0  # 找到黑色部分

# 将黑色路径部分映射到原图
mapped_bev_image = bev_image.copy()  # 创建副本以避免直接修改原图
mapped_bev_image[path_mask] = red_color  # 将路径部分设置为红色

# 显示映射后的图像
cv2.imshow('Mapped BEV Image with Red Path', mapped_bev_image)

# 保存映射后的图像
output_path = r"H:\IROS\Video\video\Path_Generate_Video\00004\wayfast\400_900_mapped_red_path111.png"
cv2.imwrite(output_path, mapped_bev_image)

# 等待按键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Mapped BEV image with red path saved to {output_path}")
