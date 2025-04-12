import numpy as np
import cv2

def generate_binary_mask(npy_file, width=1280, height=720, threshold=10):
    # 读取存储在 .npy 文件中的坐标和粗糙度数据
    coordinates_and_roughness = np.load(npy_file)
    
    # 创建一个黑色图像，尺寸为 1280x720
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 遍历每个点，设置掩码像素值
    for x, y, roughness in coordinates_and_roughness:
        # 确保坐标 (x, y) 在图像范围内
        if 0 <= x < width and 0 <= y < height:
            # 如果粗糙度小于或等于阈值，设置为白色 (255)，否则为黑色 (0)
            if roughness <= threshold:
                mask[int(y), int(x)] = 255  # 设置为白色
            else:
                mask[int(y), int(x)] = 0    # 设置为黑色

    return mask

# 测试代码
if __name__ == "__main__":
    npy_file = r"E:\BaiduNetdiskDownload\ORFD_Dataset_ICRA2022_ZIP\ORFD-custom\training\score\16237330\score_array\1623733074172.npy"  # 替换为实际的 npy 文件路径
    binary_mask = generate_binary_mask(npy_file)

    # 保存二值掩码图像
    output_image_path = 'binary_mask.png'
    cv2.imwrite(output_image_path, binary_mask)

    # 可选：显示图像
    cv2.imshow('Binary Mask', binary_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
