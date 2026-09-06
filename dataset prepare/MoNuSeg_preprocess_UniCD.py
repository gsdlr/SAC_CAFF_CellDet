import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import random
import math

# ============ 解决中文乱码 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_tiff(tiff_path):
    img = Image.open(tiff_path)
    return np.array(img)


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    all_regions = []
    for region in root.iter('Region'):
        vertices = []
        for vertex in region.iter('Vertex'):
            x = float(vertex.get('X'))
            y = float(vertex.get('Y'))
            vertices.append([x, y])
        if len(vertices) >= 3:
            all_regions.append(np.array(vertices))
    return all_regions


def xml_to_instance_mask(regions, img_shape):
    h, w = img_shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.int32)
    for idx, vertices in enumerate(regions, start=1):
        pts = vertices.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(instance_mask, [pts], color=idx)
    return instance_mask


def get_cell_centers(regions):
    centers = []
    for vertices in regions:
        cx = np.mean(vertices[:, 0])
        cy = np.mean(vertices[:, 1])
        centers.append([int(round(cx)), int(round(cy))])
    return centers


def create_dot_map(centers, img_shape):
    h, w = img_shape[:2]
    dot_map = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in centers:
        if 0 <= cx < w and 0 <= cy < h:
            dot_map[cy, cx] = 1
    return dot_map


def generate_colormap(n):
    random.seed(42)
    colors = []
    for i in range(n):
        h = int((i * 180 / n) % 180)
        s = random.randint(150, 200)
        v = random.randint(150, 200)
        color_hsv = np.array([[[h, s, v]]], dtype=np.uint8)
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(color_rgb.tolist())
    random.shuffle(colors)
    return colors


def crop_and_save(tiff_dir, label_dir, output_dir, patch_size=320, img_offset=0, dataset_name=""):
    """
    将图像裁剪为 patch_size x patch_size 的不重叠patch，
    同时生成对应的 dot_map 并保存
    """
    images_out_dir = os.path.join(output_dir, 'images')
    dotmaps_out_dir = os.path.join(output_dir, 'dot_maps')
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(dotmaps_out_dir, exist_ok=True)

    tiff_files = sorted([f for f in os.listdir(tiff_dir) if f.endswith(('.tif', '.tiff'))])

    print(f"{'=' * 70}")
    print(f"{dataset_name} 裁剪信息 (patch_size={patch_size}x{patch_size}, 图片编号偏移={img_offset})")
    print(f"{'=' * 70}")
    print(f"输入目录: {tiff_dir}")
    print(f"输出目录: {output_dir}")
    print(f"图片数量: {len(tiff_files)}")
    print(f"{'=' * 70}\n")

    total_cells_before = 0
    total_cells_after = 0
    all_patch_info = []
    all_image_info = []

    for file_idx, tiff_file in enumerate(tiff_files, start=1):
        img_idx = file_idx + img_offset
        base_name = tiff_file.rsplit('.', 1)[0]
        img = read_tiff(os.path.join(tiff_dir, tiff_file))
        h, w = img.shape[:2]

        # 读取XML获取regions和centers
        xml_path = os.path.join(label_dir, base_name + '.xml')
        regions = read_xml(xml_path)
        centers = get_cell_centers(regions)
        n_cells_before = len(centers)
        total_cells_before += n_cells_before

        # 记录原图信息
        all_image_info.append({
            'img_idx': img_idx,
            'base_name': base_name,
            'tiff_file': tiff_file,
            'n_cells': n_cells_before,
            'width': w,
            'height': h
        })

        # 计算patch的起始位置
        x_starts = list(range(0, w - patch_size + 1, patch_size))
        y_starts = list(range(0, h - patch_size + 1, patch_size))
        n_patches = len(x_starts) * len(y_starts)

        print(f"图 #{img_idx} [{base_name}] 尺寸:{w}x{h} 细胞数:{n_cells_before} "
              f"-> {len(x_starts)}x{len(y_starts)}={n_patches} patches")

        patch_idx = 0
        img_cells_after = 0
        patch_details = []

        for yi, y0 in enumerate(y_starts):
            for xi, x0 in enumerate(x_starts):
                patch_idx += 1
                y1 = y0 + patch_size
                x1 = x0 + patch_size

                # 裁剪图像
                img_patch = img[y0:y1, x0:x1]

                # 找出中心落在该patch内的细胞
                patch_centers = []
                patch_regions = []
                for c_idx, (cx, cy) in enumerate(centers):
                    if x0 <= cx < x1 and y0 <= cy < y1:
                        patch_centers.append([cx - x0, cy - y0])
                        shifted_region = regions[c_idx].copy()
                        shifted_region[:, 0] -= x0
                        shifted_region[:, 1] -= y0
                        patch_regions.append(shifted_region)

                # 生成dot_map
                dot_map = create_dot_map(patch_centers, (patch_size, patch_size))

                # 保存
                patch_name = f"MoNuSeg_{img_idx}_{patch_idx}"
                img_save_path = os.path.join(images_out_dir, patch_name + '.png')
                dot_save_path = os.path.join(dotmaps_out_dir, patch_name + '.npy')

                Image.fromarray(img_patch).save(img_save_path)
                np.save(dot_save_path, dot_map)

                n_patch_cells = len(patch_centers)
                img_cells_after += n_patch_cells
                patch_details.append(n_patch_cells)

                all_patch_info.append({
                    'img_idx': img_idx,
                    'patch_idx': patch_idx,
                    'patch_name': patch_name,
                    'x0': x0, 'y0': y0,
                    'centers': patch_centers,
                    'regions': patch_regions,
                    'n_cells': n_patch_cells
                })

        total_cells_after += img_cells_after
        print(f"  各patch细胞数: {patch_details}")
        print(f"  patch细胞总数: {img_cells_after} (原图: {n_cells_before}, "
              f"边界丢失: {n_cells_before - img_cells_after})")
        print()

    print(f"{'=' * 70}")
    print(f"{dataset_name}总计: 原始细胞数={total_cells_before}, 裁剪后细胞数={total_cells_after}, "
          f"边界丢失={total_cells_before - total_cells_after}")
    print(f"{dataset_name}总patch数: {len(all_patch_info)}")
    print(f"{'=' * 70}\n")

    return all_patch_info, all_image_info, total_cells_before, total_cells_after


def generate_readme(output_dir, patch_size, train_image_info, test_image_info,
                    train_cells_before, train_cells_after,
                    test_cells_before, test_cells_after,
                    train_patch_info, test_patch_info):
    """根据处理结果自动生成 README.txt"""

    n_train = len(train_image_info)
    n_test = len(test_image_info)
    n_total = n_train + n_test

    total_cells_before = train_cells_before + test_cells_before
    total_cells_after = train_cells_after + test_cells_after
    total_loss = total_cells_before - total_cells_after

    n_train_patches = len(train_patch_info)
    n_test_patches = len(test_patch_info)
    n_total_patches = n_train_patches + n_test_patches

    # 获取原始图像尺寸（取第一张）
    if train_image_info:
        orig_w = train_image_info[0]['width']
        orig_h = train_image_info[0]['height']
    else:
        orig_w = test_image_info[0]['width']
        orig_h = test_image_info[0]['height']

    # 计算裁剪参数
    n_per_dim = orig_w // patch_size
    patches_per_img = n_per_dim * n_per_dim
    effective_range = n_per_dim * patch_size
    margin = orig_w - effective_range

    readme_content = f"""# MoNuSeg 数据集

## 概述

本数据集由 MoNuSeg_original 数据集裁剪而来，用于细胞检测任务。通过将大尺寸图像裁剪为小 patch，增加样本数量，同时适配模型输入尺寸。

## 原始数据

- 原始图像尺寸：{orig_w} × {orig_h} 像素
- 来源路径：`MoNuSeg_original/train/images`（训练集）和 `MoNuSeg_original/test/images`（测试集）

## 裁剪策略

- Patch 尺寸：{patch_size} × {patch_size} 像素
- 从图像左上角开始，均匀排布，无重叠
- 每个维度裁剪数量：{n_per_dim}（{orig_w} // {patch_size} = {n_per_dim}）
- 每张原图产生 patch 数量：{patches_per_img}（{n_per_dim} × {n_per_dim}）
- 有效覆盖范围：{effective_range} × {effective_range} / {orig_w} × {orig_h}（右侧 {margin} 像素和底部 {margin} 像素未被覆盖）

## 命名规则

- 图像：`MoNuSeg_X_Y.png`
- 点标注：`MoNuSeg_X_Y.npy`
- `X`：原始图像编号（训练集 1–{n_train}，测试集 {n_train + 1}–{n_total}）
- `Y`：该图像内的 patch 编号（1–{patches_per_img}，按行优先顺序）

## 数据统计

- 原始图像数量：{n_total}（训练集 {n_train} 张，测试集 {n_test} 张）
- 原始细胞数量：{total_cells_before}（训练集 {train_cells_before}，测试集 {test_cells_before}）
- 裁剪后 patch 总数：{n_total_patches}（训练集 {n_train_patches}，测试集 {n_test_patches}）
- 裁剪后总细胞数量：{total_cells_after}（训练集 {train_cells_after}，测试集 {test_cells_after}）
- 边界丢失细胞数：{total_loss}

## 备注

由于裁剪边界限制，位于未覆盖区域（右侧和底部各 {margin} 像素）内的细胞未被纳入裁剪后的 patch 中，因此裁剪后的总细胞数量略少于原始图像的总细胞数量。
"""

    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content.strip())

    print(f"\nREADME 已生成: {readme_path}")
    print(f"{'=' * 70}\n")


def visualize_top_before_crop(tiff_dir, label_dir, all_image_info, top_n=2, dataset_name=""):
    """可视化细胞数量最多的top_n张原图（裁剪前）"""
    sorted_images = sorted(all_image_info, key=lambda x: x['n_cells'], reverse=True)
    selected_images = sorted_images[:top_n]

    fig, axes = plt.subplots(top_n, 4, figsize=(20, 5 * top_n))
    if top_n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'{dataset_name} - 裁剪前（细胞数最多的{top_n}张原图）', fontsize=14, fontweight='bold')

    for i, img_info in enumerate(selected_images):
        base_name = img_info['base_name']
        tiff_file = img_info['tiff_file']

        img = read_tiff(os.path.join(tiff_dir, tiff_file))
        xml_path = os.path.join(label_dir, base_name + '.xml')
        regions = read_xml(xml_path)
        centers = get_cell_centers(regions)

        # Instance mask
        instance_mask = xml_to_instance_mask(regions, img.shape)
        colors = generate_colormap(len(regions) + 1)
        colored_mask = np.zeros_like(img)
        for j in range(1, instance_mask.max() + 1):
            colored_mask[instance_mask == j] = colors[j % len(colors)]

        # Dot map
        dot_map = create_dot_map(centers, img.shape)

        # Overlay
        overlay = (img.astype(np.float32) * 0.5 + colored_mask.astype(np.float32) * 0.5).astype(np.uint8)
        for vertices in regions:
            pts = vertices.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        for cx, cy in centers:
            cv2.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'#{img_info["img_idx"]} {base_name}\n'
                             f'{img_info["width"]}x{img_info["height"]} | 细胞数:{img_info["n_cells"]}', fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(colored_mask)
        axes[i, 1].set_title(f'Instance Mask ({len(regions)} nuclei)', fontsize=9)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(dot_map, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Dot Map ({len(centers)} centers)', fontsize=9)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay + Contours + Centers', fontsize=9)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_top_after_crop(output_dir, all_patch_info, all_image_info, top_n=2, per_page=3, dataset_name=""):
    """可视化细胞数量最多的top_n张原图对应的patches（裁剪后）"""
    images_dir = os.path.join(output_dir, 'images')
    dotmaps_dir = os.path.join(output_dir, 'dot_maps')

    sorted_images = sorted(all_image_info, key=lambda x: x['n_cells'], reverse=True)
    selected_img_idxs = [img_info['img_idx'] for img_info in sorted_images[:top_n]]

    selected_patches = [p for p in all_patch_info if p['img_idx'] in selected_img_idxs]

    img_info_map = {info['img_idx']: info for info in all_image_info}

    n_pages = math.ceil(len(selected_patches) / per_page)

    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, len(selected_patches))
        current_patches = selected_patches[start:end]
        n_rows = len(current_patches)

        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f'{dataset_name} - 裁剪后（细胞数最多的{top_n}张原图的Patches, '
                     f'Page {page+1}/{n_pages}）', fontsize=13, fontweight='bold')

        for i, patch_info in enumerate(current_patches):
            patch_name = patch_info['patch_name']
            patch_centers = patch_info['centers']
            patch_regions = patch_info['regions']
            source_info = img_info_map[patch_info['img_idx']]

            # 读取patch图像
            img_path = os.path.join(images_dir, patch_name + '.png')
            img_patch = read_tiff(img_path)

            # 读取dot_map
            dot_path = os.path.join(dotmaps_dir, patch_name + '.npy')
            dot_map = np.load(dot_path)

            # Instance mask
            patch_size = img_patch.shape[0]
            instance_mask = np.zeros((patch_size, patch_size), dtype=np.int32)
            colors = generate_colormap(len(patch_regions) + 1)
            colored_mask = np.zeros_like(img_patch)

            for j, region in enumerate(patch_regions, start=1):
                pts = region.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(instance_mask, [pts], color=j)

            for j in range(1, instance_mask.max() + 1):
                colored_mask[instance_mask == j] = colors[j % len(colors)]

            # Overlay
            overlay = (img_patch.astype(np.float32) * 0.5 + colored_mask.astype(np.float32) * 0.5).astype(np.uint8)
            for region in patch_regions:
                pts = region.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
            for cx, cy in patch_centers:
                cv2.circle(overlay, (cx, cy), 2, (255, 0, 0), -1)

            img_filename = patch_name + '.png'
            dot_filename = patch_name + '.npy'

            axes[i, 0].imshow(img_patch)
            axes[i, 0].set_title(f'{img_filename}\n来源:#{source_info["img_idx"]} {source_info["base_name"]}\n'
                                 f'位置:({patch_info["x0"]},{patch_info["y0"]})', fontsize=8)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(colored_mask)
            axes[i, 1].set_title(f'Instance Mask\n({patch_info["n_cells"]} nuclei)', fontsize=8)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(dot_map, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'{dot_filename}\nDot Map ({patch_info["n_cells"]} centers)', fontsize=8)
            axes[i, 2].axis('off')

            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Overlay + Contours + Centers\n({patch_info["n_cells"]} cells)', fontsize=8)
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    # 输入目录
    train_tiff_dir = r'D:\AA-Datasets\MoNuSeg_original\train\images'
    train_label_dir = r'D:\AA-Datasets\MoNuSeg_original\train\labels'
    test_tiff_dir = r'D:\AA-Datasets\MoNuSeg_original\test\images'
    test_label_dir = r'D:\AA-Datasets\MoNuSeg_original\test\labels'

    # 输出目录
    output_dir = r'D:\AA-Datasets\MoNuSeg'

    patch_size = 320

    # ========== 处理训练集 ==========
    print("\n" + "=" * 70)
    print("处理训练集")
    print("=" * 70)
    train_patch_info, train_image_info, train_cells_before, train_cells_after = crop_and_save(
        train_tiff_dir, train_label_dir, output_dir, patch_size, img_offset=0, dataset_name="训练集"
    )

    # ========== 处理测试集 ==========
    train_tiff_files = sorted([f for f in os.listdir(train_tiff_dir) if f.endswith(('.tif', '.tiff'))])
    offset = len(train_tiff_files)

    print("\n" + "=" * 70)
    print("处理测试集")
    print("=" * 70)
    test_patch_info, test_image_info, test_cells_before, test_cells_after = crop_and_save(
        test_tiff_dir, test_label_dir, output_dir, patch_size, img_offset=offset, dataset_name="测试集"
    )

    # ========== 整个数据集统计 ==========
    print("\n" + "=" * 70)
    print("整个数据集统计:")
    print(f"  原始细胞总数 = {train_cells_before + test_cells_before}")
    print(f"  裁剪后细胞总数 = {train_cells_after + test_cells_after}")
    print(f"  边界丢失总数 = {(train_cells_before - train_cells_after) + (test_cells_before - test_cells_after)}")
    print(f"  总patch数 = {len(train_patch_info) + len(test_patch_info)}")
    print("=" * 70)

    # ========== 生成 README.txt ==========
    generate_readme(
        output_dir, patch_size,
        train_image_info, test_image_info,
        train_cells_before, train_cells_after,
        test_cells_before, test_cells_after,
        train_patch_info, test_patch_info
    )

    # ========== 可视化训练集：细胞数最多的2张原图 裁剪前 ==========
    print("\n可视化训练集 - 裁剪前（细胞数最多的2张原图）:")
    visualize_top_before_crop(train_tiff_dir, train_label_dir, train_image_info,
                             top_n=2, dataset_name="训练集")

    # ========== 可视化训练集：细胞数最多的2张原图 裁剪后 ==========
    print("\n可视化训练集 - 裁剪后（细胞数最多的2张原图的patches）:")
    visualize_top_after_crop(output_dir, train_patch_info, train_image_info,
                            top_n=2, per_page=3, dataset_name="训练集")

    # ========== 可视化测试集：细胞数最多的2张原图 裁剪前 ==========
    print("\n可视化测试集 - 裁剪前（细胞数最多的2张原图）:")
    visualize_top_before_crop(test_tiff_dir, test_label_dir, test_image_info,
                             top_n=2, dataset_name="测试集")

    # ========== 可视化测试集：细胞数最多的2张原图 裁剪后 ==========
    print("\n可视化测试集 - 裁剪后（细胞数最多的2张原图的patches）:")
    visualize_top_after_crop(output_dir, test_patch_info, test_image_info,
                            top_n=2, per_page=3, dataset_name="测试集")


if __name__ == '__main__':
    main()