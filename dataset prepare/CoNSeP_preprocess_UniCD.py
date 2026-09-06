import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
import math
from scipy.io import loadmat

# ============ 解决中文乱码 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_image(img_path):
    """读取PNG图像"""
    img = Image.open(img_path)
    return np.array(img)


def read_mat(mat_path):
    """
    读取CoNSeP的.mat标注文件，返回instance_map和centroids([x, y]格式)
    CoNSeP .mat文件通常包含: inst_map, type_map, inst_type, inst_centroid
    """
    data = loadmat(mat_path)

    # 获取instance map
    if 'inst_map' in data:
        inst_map = data['inst_map']
    elif 'instance_map' in data:
        inst_map = data['instance_map']
    else:
        raise KeyError(f"无法在 {mat_path} 中找到instance map，可用键: {[k for k in data.keys() if not k.startswith('__')]}")

    # 确保是2D数组
    inst_map = np.squeeze(inst_map).astype(np.int32)

    # 从instance map计算centroids (以确保格式一致，返回[x, y])
    centroids = []
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids != 0]  # 去掉背景
    for inst_id in inst_ids:
        mask = (inst_map == inst_id)
        ys, xs = np.where(mask)
        cy = np.mean(ys)
        cx = np.mean(xs)
        centroids.append([cx, cy])
    centroids = np.array(centroids) if len(centroids) > 0 else np.empty((0, 2))

    return inst_map, centroids


def create_dot_map(centers, img_shape):
    """根据细胞中心坐标创建dot map"""
    h, w = img_shape[:2]
    dot_map = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in centers:
        cx_int, cy_int = int(round(cx)), int(round(cy))
        if 0 <= cx_int < w and 0 <= cy_int < h:
            dot_map[cy_int, cx_int] = 1
    return dot_map


def generate_colormap(n):
    """生成n个不同颜色的colormap"""
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


def instance_mask_to_colored(inst_map, img_shape):
    """将instance map转为彩色可视化图"""
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids != 0]
    n_instances = len(inst_ids)
    colors = generate_colormap(max(n_instances + 1, 2))
    colored_mask = np.zeros((*img_shape[:2], 3), dtype=np.uint8)
    for idx, inst_id in enumerate(inst_ids):
        colored_mask[inst_map == inst_id] = colors[(idx + 1) % len(colors)]
    return colored_mask


def crop_and_save(img_dir, label_dir, output_dir, patch_size=320, img_offset=0, dataset_name=""):
    """
    将图像裁剪为 patch_size x patch_size 的不重叠patch，
    同时生成对应的 dot_map 并保存
    """
    images_out_dir = os.path.join(output_dir, 'images')
    dotmaps_out_dir = os.path.join(output_dir, 'dot_maps')
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(dotmaps_out_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    print(f"{'=' * 70}")
    print(f"{dataset_name} 裁剪信息 (patch_size={patch_size}x{patch_size}, 图片编号偏移={img_offset})")
    print(f"{'=' * 70}")
    print(f"输入目录: {img_dir}")
    print(f"输出目录: {output_dir}")
    print(f"图片数量: {len(img_files)}")
    print(f"{'=' * 70}\n")

    total_cells_before = 0
    total_cells_after = 0
    all_patch_info = []
    image_info_list = []

    for file_idx, img_file in enumerate(img_files, start=1):
        img_idx = file_idx + img_offset
        base_name = img_file.rsplit('.', 1)[0]
        img = read_image(os.path.join(img_dir, img_file))
        h, w = img.shape[:2]

        # 读取.mat标注
        mat_path = os.path.join(label_dir, base_name + '.mat')
        inst_map, centroids = read_mat(mat_path)
        n_cells_before = len(centroids)
        total_cells_before += n_cells_before

        # 保存原图信息（用于可视化）
        image_info_list.append({
            'img_idx': img_idx,
            'base_name': base_name,
            'img_file': img_file,
            'img_path': os.path.join(img_dir, img_file),
            'mat_path': mat_path,
            'n_cells': n_cells_before,
            'width': w,
            'height': h,
        })

        # 计算patch的起始位置（从边界开始，不重叠）
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

                # 裁剪instance map
                inst_map_patch = inst_map[y0:y1, x0:x1].copy()

                # 找出中心落在该patch内的细胞
                patch_centers = []
                for cx, cy in centroids:
                    if x0 <= cx < x1 and y0 <= cy < y1:
                        patch_centers.append([cx - x0, cy - y0])

                # 生成dot_map
                dot_map = create_dot_map(patch_centers, (patch_size, patch_size))

                # 保存
                patch_name = f"CoNSeP_{img_idx}_{patch_idx}"
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
                    'n_cells': n_patch_cells,
                    'img_path': os.path.join(img_dir, img_file),
                    'mat_path': mat_path,
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

    return all_patch_info, image_info_list, total_cells_before, total_cells_after


def generate_readme(output_dir, patch_size,
                    train_image_info, test_image_info,
                    train_cells_before, train_cells_after,
                    test_cells_before, test_cells_after,
                    train_patch_info, test_patch_info):
    """
    在输出目录下生成 README.md 文件，记录数据集处理信息
    """
    # 获取图像尺寸信息（取第一张图的尺寸作为代表）
    if train_image_info:
        img_w = train_image_info[0]['width']
        img_h = train_image_info[0]['height']
    elif test_image_info:
        img_w = test_image_info[0]['width']
        img_h = test_image_info[0]['height']
    else:
        img_w, img_h = 0, 0

    # 计算裁剪参数
    n_x = img_w // patch_size  # 每行patch数
    n_y = img_h // patch_size  # 每列patch数
    n_patches_per_img = n_x * n_y
    effective_w = n_x * patch_size
    effective_h = n_y * patch_size
    margin_right = img_w - effective_w
    margin_bottom = img_h - effective_h

    # 图像数量
    n_train_imgs = len(train_image_info)
    n_test_imgs = len(test_image_info)
    n_total_imgs = n_train_imgs + n_test_imgs

    # patch数量
    n_train_patches = len(train_patch_info)
    n_test_patches = len(test_patch_info)
    n_total_patches = n_train_patches + n_test_patches

    # 细胞数量
    total_cells_before = train_cells_before + test_cells_before
    total_cells_after = train_cells_after + test_cells_after
    total_loss = total_cells_before - total_cells_after

    # 编号范围
    train_start = train_image_info[0]['img_idx'] if train_image_info else 0
    train_end = train_image_info[-1]['img_idx'] if train_image_info else 0
    test_start = test_image_info[0]['img_idx'] if test_image_info else 0
    test_end = test_image_info[-1]['img_idx'] if test_image_info else 0

    readme_content = f"""# CoNSeP 数据集

## 概述

本数据集由 CoNSeP_original 数据集裁剪而来，用于细胞检测任务。通过将大尺寸图像裁剪为小 patch，增加样本数量，同时适配模型输入尺寸。

## 原始数据

- 原始图像尺寸：{img_w} × {img_h} 像素
- 来源路径：`CoNSeP_original/train/Images`（训练集）和 `CoNSeP_original/test/Images`（测试集）
- 标注格式：.mat 文件（包含 instance map）

## 裁剪策略

- Patch 尺寸：{patch_size} × {patch_size} 像素
- 从图像左上角开始，均匀排布，无重叠
- 每个维度裁剪数量：{n_x}（{img_w} // {patch_size} = {n_x}）
- 每张原图产生 patch 数量：{n_patches_per_img}（{n_x} × {n_y}）
- 有效覆盖范围：{effective_w} × {effective_h} / {img_w} × {img_h}（右侧 {margin_right} 像素和底部 {margin_bottom} 像素未被覆盖）

## 命名规则

- 图像：`CoNSeP_X_Y.png`
- 点标注：`CoNSeP_X_Y.npy`
- `X`：原始图像编号（训练集 {train_start}–{train_end}，测试集 {test_start}–{test_end}）
- `Y`：该图像内的 patch 编号（1–{n_patches_per_img}，按行优先顺序）

## 数据统计

- 原始图像数量：{n_total_imgs}（训练集 {n_train_imgs} 张，测试集 {n_test_imgs} 张）
- 原始细胞数量：{total_cells_before}（训练集 {train_cells_before}，测试集 {test_cells_before}）
- 裁剪后 patch 总数：{n_total_patches}（训练集 {n_train_patches}，测试集 {n_test_patches}）
- 裁剪后总细胞数量：{total_cells_after}（训练集 {train_cells_after}，测试集 {test_cells_after}）
- 边界丢失细胞数：{total_loss}

## 备注

由于裁剪边界限制，位于未覆盖区域（右侧和底部各 {margin_right} 和 {margin_bottom} 像素）内的细胞未被纳入裁剪后的 patch 中，因此裁剪后的总细胞数量略少于原始图像的总细胞数量。
"""

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"\nREADME.md 已生成: {readme_path}")


def visualize_before_crop(image_info_list, n_images=2, dataset_name=""):
    """
    可视化裁剪前的图片（细胞数量最多的n_images张）
    一页放n_images张图，4列：原图、Instance Mask、Dot Map、Overlay
    """
    # 按细胞数量排序取前n_images
    sorted_info = sorted(image_info_list, key=lambda x: x['n_cells'], reverse=True)
    selected = sorted_info[:n_images]

    fig, axes = plt.subplots(n_images, 4, figsize=(20, 5 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'{dataset_name} - 裁剪前 (细胞数量最多的{n_images}张原始图像)', fontsize=14, fontweight='bold')

    for i, info in enumerate(selected):
        img = read_image(info['img_path'])
        inst_map, centroids = read_mat(info['mat_path'])

        # Instance mask 彩色可视化
        colored_mask = instance_mask_to_colored(inst_map, img.shape)

        # Dot map
        dot_map = create_dot_map(centroids, img.shape)

        # Overlay: 原图 + instance mask + 轮廓 + 中心点
        overlay = (img[:, :, :3].astype(np.float32) * 0.5 + colored_mask.astype(np.float32) * 0.5).astype(np.uint8)
        # 绘制轮廓
        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids != 0]
        for inst_id in inst_ids:
            mask = (inst_map == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        # 绘制中心点
        for cx, cy in centroids:
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), 3, (255, 0, 0), -1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'#{info["img_idx"]} {info["base_name"]}\n{img.shape[1]}x{img.shape[0]}', fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(colored_mask)
        axes[i, 1].set_title(f'Instance Mask ({info["n_cells"]} nuclei)', fontsize=9)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(dot_map, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Dot Map ({info["n_cells"]} centers)', fontsize=9)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay + Contours + Centers', fontsize=9)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_after_crop(output_dir, all_patch_info, image_info_list, n_images=2, per_page=3, dataset_name=""):
    """
    可视化裁剪后的patch（细胞数量最多的n_images张图的patches）
    每页per_page行，4列：原图patch、Instance Mask、Dot Map、Overlay
    """
    images_dir = os.path.join(output_dir, 'images')
    dotmaps_dir = os.path.join(output_dir, 'dot_maps')

    # 找细胞数量最多的n_images张图的img_idx
    sorted_info = sorted(image_info_list, key=lambda x: x['n_cells'], reverse=True)
    top_img_idxs = [info['img_idx'] for info in sorted_info[:n_images]]

    # 筛选这些图的patches
    selected_patches = [p for p in all_patch_info if p['img_idx'] in top_img_idxs]

    n_pages = math.ceil(len(selected_patches) / per_page)

    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, len(selected_patches))
        current_patches = selected_patches[start:end]
        n_rows = len(current_patches)

        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f'{dataset_name} - 裁剪后 (细胞数量最多的{n_images}张图的patches, Page {page + 1}/{n_pages})',
                     fontsize=14, fontweight='bold')

        for i, patch_info in enumerate(current_patches):
            patch_name = patch_info['patch_name']
            patch_centers = patch_info['centers']
            x0, y0 = patch_info['x0'], patch_info['y0']

            # 读取patch图像
            img_path = os.path.join(images_dir, patch_name + '.png')
            img_patch = read_image(img_path)

            # 读取dot_map
            dot_path = os.path.join(dotmaps_dir, patch_name + '.npy')
            dot_map = np.load(dot_path)

            # 重新读取inst_map并裁剪对应区域
            inst_map_full, _ = read_mat(patch_info['mat_path'])
            patch_size = img_patch.shape[0]
            inst_map_patch = inst_map_full[y0:y0 + patch_size, x0:x0 + patch_size]

            # Instance mask 彩色可视化
            colored_mask = instance_mask_to_colored(inst_map_patch, img_patch.shape)

            # Overlay
            overlay = (img_patch[:, :, :3].astype(np.float32) * 0.5 + colored_mask.astype(np.float32) * 0.5).astype(
                np.uint8)
            inst_ids = np.unique(inst_map_patch)
            inst_ids = inst_ids[inst_ids != 0]
            for inst_id in inst_ids:
                mask = (inst_map_patch == inst_id).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            for cx, cy in patch_centers:
                cv2.circle(overlay, (int(round(cx)), int(round(cy))), 2, (255, 0, 0), -1)

            img_filename = patch_name + '.png'
            dot_filename = patch_name + '.npy'

            axes[i, 0].imshow(img_patch)
            axes[i, 0].set_title(f'{img_filename}\n位置:({x0},{y0})', fontsize=8)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(colored_mask)
            axes[i, 1].set_title(f'Instance Mask ({patch_info["n_cells"]} nuclei)', fontsize=8)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(dot_map, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'{dot_filename}\nDot Map ({patch_info["n_cells"]} centers)', fontsize=8)
            axes[i, 2].axis('off')

            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay + Contours + Centers', fontsize=8)
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    # 输入目录
    train_img_dir = r'D:\AA-Datasets\CoNSeP_original\train\Images'
    train_label_dir = r'D:\AA-Datasets\CoNSeP_original\train\Labels'
    test_img_dir = r'D:\AA-Datasets\CoNSeP_original\test\Images'
    test_label_dir = r'D:\AA-Datasets\CoNSeP_original\test\Labels'

    # 输出目录
    output_dir = r'D:\AA-Datasets\CoNSeP'

    patch_size = 320

    # ========== 处理训练集 ==========
    print("\n" + "=" * 70)
    print("处理训练集")
    print("=" * 70)
    train_patch_info, train_image_info, train_cells_before, train_cells_after = crop_and_save(
        train_img_dir, train_label_dir, output_dir, patch_size, img_offset=0, dataset_name="训练集"
    )

    # ========== 处理测试集 ==========
    train_img_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.png')])
    offset = len(train_img_files)

    print("\n" + "=" * 70)
    print("处理测试集")
    print("=" * 70)
    test_patch_info, test_image_info, test_cells_before, test_cells_after = crop_and_save(
        test_img_dir, test_label_dir, output_dir, patch_size, img_offset=offset, dataset_name="测试集"
    )

    # ========== 整个数据集统计 ==========
    print("\n" + "=" * 70)
    print("整个数据集统计:")
    print(f"  原始细胞总数 = {train_cells_before + test_cells_before}")
    print(f"  裁剪后细胞总数 = {train_cells_after + test_cells_after}")
    print(f"  边界丢失总数 = {(train_cells_before - train_cells_after) + (test_cells_before - test_cells_after)}")
    print(f"  总patch数 = {len(train_patch_info) + len(test_patch_info)}")
    print("=" * 70)

    # ========== 生成 README.md ==========
    generate_readme(
        output_dir, patch_size,
        train_image_info, test_image_info,
        train_cells_before, train_cells_after,
        test_cells_before, test_cells_after,
        train_patch_info, test_patch_info
    )

    # ========== 可视化训练集中细胞数量最多的两张图（裁剪前） ==========
    print("\n可视化训练集中细胞数量最多的2张图 (裁剪前):")
    visualize_before_crop(train_image_info, n_images=2, dataset_name="训练集")

    # ========== 可视化训练集中细胞数量最多的两张图（裁剪后） ==========
    print("\n可视化训练集中细胞数量最多的2张图 (裁剪后patches):")
    visualize_after_crop(output_dir, train_patch_info, train_image_info, n_images=2, per_page=3, dataset_name="训练集")

    # ========== 可视化测试集中细胞数量最多的两张图（裁剪前） ==========
    print("\n可视化测试集中细胞数量最多的2张图 (裁剪前):")
    visualize_before_crop(test_image_info, n_images=2, dataset_name="测试集")

    # ========== 可视化测试集中细胞数量最多的两张图（裁剪后） ==========
    print("\n可视化测试集中细胞数量最多的2张图 (裁剪后patches):")
    visualize_after_crop(output_dir, test_patch_info, test_image_info, n_images=2, per_page=3, dataset_name="测试集")


if __name__ == '__main__':
    main()