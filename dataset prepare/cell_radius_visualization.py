# -*- coding:utf-8 -*-
"""
Author：R
Date：28-06-2026
"""
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def visualize_cells(image_dir, label_dir, radius=10, rows=4, cols=6):
    # 支持的图片格式
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    # 获取所有图片路径
    all_files = os.listdir(image_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("未在指定的文件夹中找到图片！")
        return

    # 随机打乱图片顺序 (Shuffle)
    random.shuffle(image_files)

    images_per_page = rows * cols
    total_images = len(image_files)
    total_pages = int(np.ceil(total_images / images_per_page))

    print(f"共找到 {total_images} 张图片，将分为 {total_pages} 页显示。")

    # 分页显示
    for page in range(total_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
        fig.canvas.manager.set_window_title(f'Cell Visualization - Page {page + 1}/{total_pages}')
        axes = axes.flatten()  # 将二维坐标轴数组展平，方便遍历

        # 获取当前页的图片
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, total_images)
        current_batch = image_files[start_idx:end_idx]

        for i, ax in enumerate(axes):
            if i < len(current_batch):
                img_name = current_batch[i]
                img_path = os.path.join(image_dir, img_name)

                # 假设 label 的命名与图片相同，只是后缀为 .npy
                base_name = os.path.splitext(img_name)[0]
                label_path = os.path.join(label_dir, base_name + '.npy')

                # 读取图片
                try:
                    img = Image.open(img_path).convert('RGB')
                    ax.imshow(img)
                except Exception as e:
                    ax.set_title("Image Load Error")
                    ax.axis('off')
                    continue

                # 读取并解析 label
                if os.path.exists(label_path):
                    label_data = np.load(label_path)

                    # 判断 label_data 的形状
                    # 情况1: 如果是与图片大小相同的二维矩阵 (Dot Map)
                    if label_data.ndim == 2 and label_data.shape == (img.height, img.width):
                        y_coords, x_coords = np.where(label_data > 0)
                    # 情况2: 如果是坐标列表，形状类似于 (N, 2)
                    elif label_data.ndim == 2 and label_data.shape[1] == 2:
                        x_coords = label_data[:, 0]
                        y_coords = label_data[:, 1]
                    else:
                        print(f"警告: {label_path} 的数据格式无法识别，跳过绘制标签。")
                        x_coords, y_coords = [], []

                    # 绘制半径为 10 的圆圈
                    for x, y in zip(x_coords, y_coords):
                        # fill=False 表示只画圆圈边框，不填充内部
                        # edgecolor='red' 设置圆圈颜色为红色
                        circle = patches.Circle((x, y), radius=radius, edgecolor='red', facecolor='none', linewidth=1.5)
                        ax.add_patch(circle)
                else:
                    ax.set_title("No Label", fontsize=10, color='red')

                # 设置子图标题为图片名（截断过长的名字以防重叠）
                ax.set_title(img_name[:15] + '...' if len(img_name) > 15 else img_name, fontsize=10)

            # 隐藏坐标轴
            ax.axis('off')

        plt.tight_layout()
        plt.show()  # 阻塞显示，关闭当前窗口后才会显示下一页


if __name__ == "__main__":
    # 请将这里的路径替换为你实际的文件夹路径
    # 假设你的目录结构是:
    # dataset/
    # ├── images/
    # │   ├── dot_maps/
    # │   │   ├── img1.npy
    # │   │   └── img2.npy
    # │   ├── img1.jpg
    # │   └── img2.jpg

    IMAGE_DIR = r'F:\AAAAA-Semi-Supervised-Cell-Detection\datasets\PanNuke\images'
    LABEL_DIR = r'F:\AAAAA-Semi-Supervised-Cell-Detection\datasets\PanNuke\dot_maps'

    visualize_cells(IMAGE_DIR, LABEL_DIR, radius=6, rows=4, cols=6)