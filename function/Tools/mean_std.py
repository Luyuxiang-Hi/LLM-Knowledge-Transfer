import os
from PIL import Image
import numpy as np

def compute_stats(folder_path):
    channel_sums = np.zeros(3)
    channel_sq_diff_sums = np.zeros(3)
    pixel_count = 0

    # 首先，计算每个通道的总和和像素数量
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    # 提取前三个通道
                    r, g, b, a = img.split()
                    img = Image.merge("RGB", (r,g,b))
                np_img = np.array(img)
                channel_sums += np_img.sum(axis=(0, 1))
                pixel_count += np_img.shape[0] * np_img.shape[1]

    # 计算每个通道的平均值
    means = channel_sums / pixel_count

    # 接着，计算每个像素与平均值之差的平方和
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    # 提取前三个通道
                    r, g, b, a = img.split()
                    img = Image.merge("RGB", (r,g,b))
                np_img = np.array(img)
                diff = np_img - means
                channel_sq_diff_sums += np.square(diff).sum(axis=(0, 1))

    # 最后，计算标准差
    stds = np.sqrt(channel_sq_diff_sums / pixel_count)

    return means, stds

# 使用文件夹路径作为参数调用函数
folder_path = r'./02pv'
means, stds = compute_stats(folder_path)
print("平均值:", means)
print("标准差:", stds)