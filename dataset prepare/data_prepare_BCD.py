# -*- coding:utf-8 -*-
"""
Author：R
Date：05-07-2025
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import shutil


# Merge the positive dot map and the negative dot map of the same image to be one dot map.
def merge_all_labels(base_dir):

    # 创建输出目录
    output_dir = os.path.join(base_dir, 'dot_maps', 'merged')
    os.makedirs(output_dir, exist_ok=True)

    # merged之后的dot_map的命名方式为‘_merged.npy’,因此也在对应的image后面加‘_cell’,方便sorted之后对齐，创建一个文件夹存放重命名之后的images
    images_cell_dir = os.path.join(base_dir, 'images_cell')
    os.makedirs(images_cell_dir, exist_ok=True)

    # 获取所有样本ID (假设文件名是数字)
    sample_ids = sorted([int(f.split('.')[0]) for f in os.listdir(os.path.join(base_dir, 'images'))
                         if f.endswith('.png')])
    sample_ids.sort()

    # 处理每个样本
    for i, sample_id in enumerate(tqdm(sample_ids, desc="合并标签")):
        # 构建文件路径
        positive_path = os.path.join(base_dir, 'dot_maps', 'positive', f'{sample_id}.npy')
        negative_path = os.path.join(base_dir, 'dot_maps', 'negative', f'{sample_id}.npy')
        image_path = os.path.join(base_dir, 'images', f'{sample_id}.png')

        # 新命名规则
        merged_output_path = os.path.join(output_dir, f'{sample_id}_merged.npy')
        cell_image_path = os.path.join(images_cell_dir, f'{sample_id}_cell.png')

        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [positive_path, negative_path, image_path]):
            print(f"警告: 样本 {sample_id} 文件缺失，跳过")
            continue

        try:
            # 加载文件
            positive = np.load(positive_path)
            negative = np.load(negative_path)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # 验证尺寸
            if positive.shape != negative.shape:
                print(f"警告: 样本 {sample_id} 标签尺寸不匹配 (pos: {positive.shape}, neg: {negative.shape})")
                continue

            # 合并标签
            merged = np.logical_or(positive, negative).astype(np.uint8)

            # 保存合并结果（使用新命名规则）
            np.save(merged_output_path, merged)

            # 复制并重命名图像文件（如果尚未存在）
            if not os.path.exists(cell_image_path):
                shutil.copy2(image_path, cell_image_path)

            # 每50个样本显示一次统计信息和可视化
            if i % 50 == 0:
                # 计算统计信息
                pos_count = np.sum(positive)
                neg_count = np.sum(negative)
                merged_count = np.sum(merged)

                # 打印统计信息
                print(f"\n样本 {sample_id} 统计:")
                print(f"Positive细胞: {pos_count}")
                print(f"Negative细胞: {neg_count}")
                print(f"总细胞: {merged_count}")
                print(f"位置检查: 是否有重叠? {np.any(np.logical_and(positive, negative))}")

                # 可视化
                plt.figure(figsize=(20, 20))
                plt.subplot(221).imshow(img), plt.title(f'Image')
                plt.subplot(223).imshow(positive, cmap='gray'), plt.title(f'Positive\nCount: {pos_count}')
                plt.subplot(224).imshow(negative, cmap='gray'), plt.title(f'Negative\nCount: {neg_count}')
                plt.subplot(222).imshow(merged, cmap='gray'), plt.title(f'Merged\nCount: {merged_count}')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"处理样本 {sample_id} 时出错: {str(e)}")

    print(f"\n合并完成! 共处理 {len(sample_ids)} 个样本")
    print(f"合并结果保存在: {output_dir}")
    print(f"重命名的细胞图像保存在: {images_cell_dir}")


if __name__ == '__main__':
    # 设置基础路径
    base_dir = 'F://New dataset 20250703/BCData_N_803_processed/test'
    merge_all_labels(base_dir)