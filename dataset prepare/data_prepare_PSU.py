# -*- coding:utf-8 -*-
"""
Author：R
Date：04-07-2025
"""
import torch
import torch.nn as nn
import numpy as np
import math
import random
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from tqdm import tqdm
import shutil
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def npy_generation(base_dir):
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, f))
                  and f.startswith('img')]

    # 处理每个子文件夹
    for folder in tqdm(subfolders, desc="处理进度"):
        folder_path = os.path.join(base_dir, folder)

        # 构建文件路径
        img_name = folder  # 文件夹名即基础文件名
        image_path = os.path.join(folder_path, f"{img_name}.bmp")
        anno_path = os.path.join(folder_path, f"{img_name}_detection.mat")
        output_path = os.path.join(folder_path, f"{img_name}_detection.npy")

        # 跳过已处理的文件
        if os.path.exists(output_path):
            continue

        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告：无法读取图像 {image_path}")
                continue

            # 转换颜色空间（仅用于可视化）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]

            # 创建标签图
            label_map = np.zeros((height, width), dtype=np.uint8)

            # 读取坐标
            mat_data = scipy.io.loadmat(anno_path)
            points = mat_data['detection']

            # 标记坐标点
            cell_count = 0
            for point in points:
                x, y = map(int, point)
                if 0 <= y < height and 0 <= x < width:
                    label_map[y, x] = 1
                    cell_count += 1

            # 保存为.npy格式
            np.save(output_path, label_map)

            # 可选：可视化（每10个显示一次）
            if int(folder[3:]) % 10 == 0:
                print(cell_count)
                plt.figure(figsize=(12, 6))
                plt.subplot(121).imshow(img_rgb), plt.title(f'Original: {folder}')
                plt.subplot(122).imshow(label_map, cmap='gray'), plt.title(
                    f'Label Map: {folder} \nCells: {label_map.sum()}')
                plt.show()

        except Exception as e:
            print(f"处理 {folder} 时出错: {str(e)}")

        print("所有标签图生成完成！")


def organize_dataset_files(base_dir, base_des_dir):
    # 创建目标文件夹
    image_dir = os.path.join(base_des_dir, "images")
    mat_dir = os.path.join(base_des_dir, "annotations")
    npy_dir = os.path.join(base_des_dir, "dot_maps")

    for dir_path in [image_dir, mat_dir, npy_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 遍历所有子文件夹
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if not os.path.isdir(folder_path) or not folder.startswith('img'):
            continue

        # 处理每个子文件夹中的文件
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            # 移动并重命名文件
            if file.endswith('.bmp'):
                dest = os.path.join(image_dir, f"{folder}_cell.bmp")
                shutil.copy(file_path, dest)

            elif file.endswith('.mat'):
                dest = os.path.join(mat_dir, f"{folder}_detection.mat")
                shutil.copy(file_path, dest)

            elif file.endswith('.npy'):
                dest = os.path.join(npy_dir, f"{folder}_detection.npy")
                shutil.copy(file_path, dest)

    print(f"文件整理完成！\n")


if __name__ == '__main__':
    base_dir = r"F://New dataset 20250703/PSU Dataset_processed/PSU_cell_detection_dataset"
    npy_generation(base_dir)

    base_des_dir = r"F://New dataset 20250703/PSU Dataset_processed/PSU"
    organize_dataset_files(base_dir, base_des_dir)
