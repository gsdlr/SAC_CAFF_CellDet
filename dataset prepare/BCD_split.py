# -*- coding:utf-8 -*-
"""
Author：R
Date：23-07-2026
"""
import os
import glob
import numpy as np
import cv2


def pad_and_crop(image, dot_map=None, patch_size=320):
    """
    对图像和标签进行填充并切割
    """
    h, w = image.shape[:2]

    # 计算需要填充的像素大小，使其成为 patch_size 的整数倍
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # 对图像进行边缘填充 (使用黑色/0填充)
    if len(image.shape) == 3:  # 彩色图
        padded_img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    else:  # 灰度图
        padded_img = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # 如果有标签(dot_map)，对标签也进行完全相同的填充，保证空间对齐
    if dot_map is not None:
        padded_dot = np.pad(dot_map, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:
        padded_dot = None

    new_h, new_w = padded_img.shape[:2]
    img_patches = []
    dot_patches = []
    coords = []  # 记录切割的坐标 (用于命名)

    # 滑动窗口进行切割 (无重叠)
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            # 切割图像
            img_patch = padded_img[i:i + patch_size, j:j + patch_size]
            img_patches.append(img_patch)
            coords.append((i // patch_size, j // patch_size))

            # 同步切割标签
            if padded_dot is not None:
                dot_patch = padded_dot[i:i + patch_size, j:j + patch_size]
                dot_patches.append(dot_patch)

    return img_patches, dot_patches, coords


def process_dataset(base_dir, output_dir, patch_size=320):
    """
    处理整个数据集
    """
    # 定义输入路径
    labeled_img_dir = os.path.join(base_dir, 'labeled', 'images')
    labeled_dot_dir = os.path.join(base_dir, 'labeled', 'dot_maps')
    unlabeled_img_dir = os.path.join(base_dir, 'unlabeled', 'images')

    # 定义输出路径 (这里已经包含了 train 这一层)
    out_labeled_img = os.path.join(output_dir, 'labeled', 'images')
    out_labeled_dot = os.path.join(output_dir, 'labeled', 'dot_maps')
    out_unlabeled_img = os.path.join(output_dir, 'unlabeled', 'images')

    # 创建输出文件夹
    for path in [out_labeled_img, out_labeled_dot, out_unlabeled_img]:
        os.makedirs(path, exist_ok=True)

    # ==========================================
    # 1. 处理 Labeled 数据 (Image + Dot_map 同步切割)
    # ==========================================
    print("开始处理 Labeled 数据 (包含 images 和 dot_map)...")
    labeled_images = glob.glob(os.path.join(labeled_img_dir, '*.png'))
    for img_path in labeled_images:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)  # name 例如 "104_cell"

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # 解决命名不一致问题：将 _cell 替换为 _merged 去寻找对应的 npy
        if name.endswith('_cell'):
            dot_name = name.replace('_cell', '_merged')
        else:
            dot_name = name

        dot_path = os.path.join(labeled_dot_dir, dot_name + '.npy')

        if not os.path.exists(dot_path):
            print(f"警告: 找不到 {filename} 对应的标签文件 ({dot_name}.npy)，跳过。")
            continue

        dot_map = np.load(dot_path)

        # 将 image 和 dot_map 一起传给切割函数，实现同步切割
        img_patches, dot_patches, coords = pad_and_crop(img, dot_map, patch_size)

        # 保存切割后的 image 和 dot_map
        for patch, dot, (row, col) in zip(img_patches, dot_patches, coords):
            patch_name = f"{name}_{row}_{col}"

            # 保存图片
            cv2.imwrite(os.path.join(out_labeled_img, patch_name + ext), patch)
            # 保存对应的 npy 标签
            np.save(os.path.join(out_labeled_dot, patch_name + '.npy'), dot)

    print(f"Labeled 数据处理完成！共处理了 {len(labeled_images)} 张原图。")

    # ==========================================
    # 2. 处理 Unlabeled 数据 (仅 Image 切割)
    # ==========================================
    print("开始处理 Unlabeled 数据 (仅 images)...")
    unlabeled_images = glob.glob(os.path.join(unlabeled_img_dir, '*.png'))
    for img_path in unlabeled_images:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # 无标签数据传入 dot_map=None
        img_patches, _, coords = pad_and_crop(img, None, patch_size)

        # 保存
        for patch, (row, col) in zip(img_patches, coords):
            patch_name = f"{name}_{row}_{col}"
            cv2.imwrite(os.path.join(out_unlabeled_img, patch_name + ext), patch)

    print(f"Unlabeled 数据处理完成！共处理了 {len(unlabeled_images)} 张原图。")
    print(f"所有数据已成功保存至: {output_dir}")


if __name__ == '__main__':
    # 原始数据路径 (指向 train 文件夹)
    INPUT_DIR = r"F:\AAAAA-Semi-Supervised-Cell-Detection\datasets\BCD_20%\train"

    # 修改了这里！输出路径现在包含了 \train 这一层级
    OUTPUT_DIR = r"F:\AAAAA-Semi-Supervised-Cell-Detection\datasets\BCD_20%_train_320\train"

    # 执行处理，统一切割为 320x320
    process_dataset(INPUT_DIR, OUTPUT_DIR, patch_size=320)