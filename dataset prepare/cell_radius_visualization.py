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
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    all_files = os.listdir(image_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("未在指定的文件夹中找到图片！")
        return

    random.shuffle(image_files)

    images_per_page = rows * cols
    total_images = len(image_files)
    total_pages = int(np.ceil(total_images / images_per_page))

    print(f"共找到 {total_images} 张图片，将分为 {total_pages} 页显示。")

    for page in range(total_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
        fig.canvas.manager.set_window_title(f'Cell Visualization - Page {page + 1}/{total_pages}')
        axes = axes.flatten()

        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, total_images)
        current_batch = image_files[start_idx:end_idx]

        for i, ax in enumerate(axes):
            if i < len(current_batch):
                img_name = current_batch[i]
                img_path = os.path.join(image_dir, img_name)

                base_name = os.path.splitext(img_name)[0]
                label_path = os.path.join(label_dir, base_name + '.npy')

                try:
                    img = Image.open(img_path).convert('RGB')
                    ax.imshow(img)
                except Exception as e:
                    ax.set_title("Image Load Error")
                    ax.axis('off')
                    continue

                if os.path.exists(label_path):
                    label_data = np.load(label_path)

                    if label_data.ndim == 2 and label_data.shape == (img.height, img.width):
                        y_coords, x_coords = np.where(label_data > 0)
                    elif label_data.ndim == 2 and label_data.shape[1] == 2:
                        x_coords = label_data[:, 0]
                        y_coords = label_data[:, 1]
                    else:
                        print(f"警告: {label_path} 的数据格式无法识别，跳过绘制标签。")
                        x_coords, y_coords = [], []

                    for x, y in zip(x_coords, y_coords):

                        circle = patches.Circle((x, y), radius=radius, edgecolor='red', facecolor='none', linewidth=1.5)
                        ax.add_patch(circle)
                else:
                    ax.set_title("No Label", fontsize=10, color='red')

                ax.set_title(img_name[:15] + '...' if len(img_name) > 15 else img_name, fontsize=10)

            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    IMAGE_DIR = r'F:\AAAAA-Semi-Supervised-Cell-Detection\datasets\PanNuke\images'
    LABEL_DIR = r'F:\AAAAA-Semi-Supervised-Cell-Detection\datasets\PanNuke\dot_maps'

    visualize_cells(IMAGE_DIR, LABEL_DIR, radius=6, rows=4, cols=6)