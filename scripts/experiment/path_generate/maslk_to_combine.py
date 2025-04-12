import cv2
import numpy as np

# # 00000
# bev_image_path = r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\gt\map\400_9000\400_900_mapped_red_path.png"
#
# mask_image_paths = [
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Mine_Network\00000\good\3.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\PnpNet\00000\pnpnet_path_0\400_900\400_900_RELLIS_3D_PNPNET00000_16_13_27_trajectory_map.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Road_Seg\00000\path_road_seg\400_900\400_900_RELLIS_3D-20-02-25-23_06_55-model_best-RELLIS_3D_15_26_38_trajectory_map.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\wayfast\000\400_900\400_900_RELLIS_3D-22-02-25-03_06_40-model_best-RELLIS_3D_14_08_04_trajectory_map.png"
# ]
# bev_image_path = r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\gt\map\400_9000\400_900_mapped_red_path.png"
#
# mask_image_paths = [
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Mine_Network\00000\good\3.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\PnpNet\00000\pnpnet_path_0\400_900\400_900_RELLIS_3D_PNPNET00000_16_13_27_trajectory_map.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Road_Seg\00000\path_road_seg\400_900\400_900_RELLIS_3D-20-02-25-23_06_55-model_best-RELLIS_3D_15_26_38_trajectory_map.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\wayfast\000\400_900\400_900_RELLIS_3D-22-02-25-03_06_40-model_best-RELLIS_3D_14_08_04_trajectory_map.png"
# ]

# 00001
# bev_image_path = r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\gt\map\400_9001\400_900_mapped_red_path.png"
# # bev_image_path = r"C:\Users\Administrator\Desktop\IROS\Ablation_Study\path\result\Mine_Network\00001\400_900\400_900_RELLIS_3D_00001_22_23_48_trajectory_overlay.png"
#
# mask_image_paths = [
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Road_Seg\00001\400_900\400_900_RELLIS_ROADSEG_00001_02_04_05_trajectory_map.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\PnpNet\00001\400_900\400_900_pnpnet_00001_01_11_35_trajectory_map.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Road_Seg\00001\400_900\400_900_RELLIS_ROADSEG_00001_02_05_05_trajectory_map.png",
#     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\wayfast\001\400_900\400_900_RELLIS_3D-22-02-25_wayfast_00001_15_05_43_trajectory_map.png"
# ]
# # 00004
bev_image_path = r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\gt\map\400_9004\400_900_mapped_red_path.png"

mask_image_paths = [
    r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Mine_Network\00004\good\400_900_trajectory_map.png",
     r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\PnpNet\00004\400_900\400_900_RELLIS_3D-20-02-25-23_24_08-model_best-RELLIS_3D_00_07_38_trajectory_map.png",
    r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Road_Seg\00004\400_900\400_900_RELLIS_ROADSEG_00004_16_24_32_trajectory_map.png",
        r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\Road_Seg\00004\400_900\400_900_RELLIS_ROADSEG_00004_16_23_29_trajectory_map.png"
]

# 定义四个颜色：红色（Mine）、绿色（PnpNet）、棕色（RoadSeg）、黄色（wayfast）
colors = [
    [255, 0, 255],  # 紫色 Mine
    [0, 255, 0],  # 绿色 Pnpnet
    [255, 165, 0],  # 橙色 RoadSeg
    [0, 255, 255]  # 青色 wayfast
]

# 读取BEV图像
bev_image = cv2.imread(bev_image_path)
if bev_image is None:
    print(f"错误: 无法读取 BEV 图像 {bev_image_path}")
    exit()

# 创建一个副本来显示所有轨迹
mapped_bev_image = bev_image.copy()

# 读取并处理所有掩码路径
for idx, mask_image_path in enumerate(mask_image_paths):
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    if mask_image is None:
        print(f"错误: 无法读取掩码图像 {mask_image_path}")
        continue

    # 确保掩码图是二值化的
    _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

    # 使用形态学操作腐蚀掩码，使轨迹线变细
    kernel = np.ones((3, 3), np.uint8)  # 定义腐蚀操作的核
    thin_mask = cv2.erode(binary_mask, kernel, iterations=1)  # 执行腐蚀操作

    # 获取对应颜色
    color = colors[idx % len(colors)]  # 按照索引顺序获取颜色（循环使用）

    # 将掩码图中值为255的位置的BEV图像素设置为对应颜色
    mapped_bev_image[thin_mask == 255] = color  # 填充为指定颜色

# 显示映射后的图像
cv2.imshow('Mapped BEV Image with Multiple Trajectories', mapped_bev_image)

# 等待按键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果到指定路径
output_path = r"E:\BaiduNetdiskDownload\IROS\Ablation_Study\path\result\combine\mapped_trajectory_image_multiple.png"
cv2.imwrite(output_path, mapped_bev_image)