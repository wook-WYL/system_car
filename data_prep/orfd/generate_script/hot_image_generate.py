import numpy as np
import matplotlib.pyplot as plt
import cv2


def overlay_heatmap(image_path, score_path, output_path=None, colormap='jet', alpha=0.6):
    # 读取原始图片
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 读取score数据，score文件格式为[x, y, score]
    scores = np.load(score_path)  # [x, y, score]

    # 获取图像的尺寸
    img_height, img_width, _ = image_rgb.shape

    # 创建一个与图像相同大小的全零矩阵来存放热力图分数
    heatmap = np.zeros((img_height, img_width), dtype=np.float32)

    # 将分数映射到heatmap上
    for x, y, score in scores:
        # 将x, y 转换为整数
        x, y = int(x), int(y)

        # 确保坐标在图像范围内
        if 0 <= x < img_width and 0 <= y < img_height:
            heatmap[y, x] = score  # 注意: [y, x] 对应 (行, 列)

    # 归一化heatmap为0到1的范围
    heatmap_min = np.nanmin(heatmap)  # 获取最小值
    heatmap_max = np.nanmax(heatmap)  # 获取最大值

    # 如果最大值和最小值相同，直接返回一个黑色图像
    if heatmap_max == heatmap_min:
        heatmap = np.zeros_like(heatmap)
    else:
        # 将热力图归一化到 [0, 1] 范围
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    # 使用热力图的色图将score值映射成颜色
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 将热力图与原图进行叠加，alpha控制透明度
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    # 如果提供了输出路径，则保存图片
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 显示结果
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()


# 调用示例
image_path = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\Final_Dataset\training\c2021_0228_1819\image_data\1620331020738.png"
score_path = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score1\c2021_0228_1819\score_array\1620331020738.npy"
output_path = r'E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score_experiment\output_image.jpg'

overlay_heatmap(image_path, score_path, output_path)
