import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


def extract_cell_centers(masks):
    """
    从实例分割 mask 中提取所有细胞的中心点，生成中心点 binary mask。
    跳过最后一个 Background 通道。

    参数:
        masks: np.ndarray, 形状为 (B, H, W, C)

    返回:
        center_masks: np.ndarray, 形状为 (B, H, W), dtype=uint8
        center_coords: list of list of tuple [(y, x), ...]
    """
    B, H, W, C = masks.shape
    center_masks = np.zeros((B, H, W), dtype=np.uint8)
    center_coords = []

    for b in range(B):
        coords = []
        for ch in range(C - 1):  # 跳过最后一个 Background 通道
            mask_ch = masks[b, :, :, ch]
            instance_ids = np.unique(mask_ch)
            instance_ids = instance_ids[instance_ids != 0]

            for inst_id in instance_ids:
                ys, xs = np.where(mask_ch == inst_id)
                cy = int(np.round(ys.mean()))
                cx = int(np.round(xs.mean()))
                cy = np.clip(cy, 0, H - 1)
                cx = np.clip(cx, 0, W - 1)
                center_masks[b, cy, cx] = 1
                coords.append((cy, cx))

        center_coords.append(coords)

    return center_masks, center_coords


def save_images_and_dot_maps(images, center_masks, save_root, fold_prefix):
    """
    将图片保存为 PNG，center_masks 保存为 .npy（0/1 二值）。

    参数:
        images: np.ndarray, (B, H, W, 3)
        center_masks: np.ndarray, (B, H, W)
        save_root: str, 保存根目录
        fold_prefix: str, 数据部分前缀，如 "2"
    """
    images_dir = os.path.join(save_root, "images")
    dot_maps_dir = os.path.join(save_root, "dot_maps")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(dot_maps_dir, exist_ok=True)

    B = images.shape[0]
    for i in range(B):
        # 保存图片为 PNG
        img_filename = f"{fold_prefix}_{i + 1}.png"
        img = images[i]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(images_dir, img_filename))

        # 保存 center_mask 为 .npy（0/1 二值）
        npy_filename = f"{fold_prefix}_{i + 1}.npy"
        np.save(os.path.join(dot_maps_dir, npy_filename), center_masks[i])

    print(f"已保存 {B} 张图片到: {images_dir}")
    print(f"已保存 {B} 张 dot_maps (.npy) 到: {dot_maps_dir}")


def colorize_instances(masks_single, skip_background=True):
    """
    将单张图的多通道实例 mask 合并为彩色图。

    参数:
        masks_single: np.ndarray, 形状为 (H, W, C)
        skip_background: 是否跳过最后一个通道（Background）

    返回:
        colored: np.ndarray, 形状为 (H, W, 3), float32
    """
    H, W, C = masks_single.shape
    colored = np.zeros((H, W, 3), dtype=np.float32)
    np.random.seed(42)

    channels = range(C - 1) if skip_background else range(C)

    for ch in channels:
        mask_ch = masks_single[:, :, ch]
        instance_ids = np.unique(mask_ch)
        instance_ids = instance_ids[instance_ids != 0]

        for inst_id in instance_ids:
            color = np.random.rand(3) * 0.8 + 0.2
            colored[mask_ch == inst_id] = color

    return colored


def colorize_single_channel(mask_ch):
    """
    单通道实例 mask 转为彩色图。
    """
    H, W = mask_ch.shape
    colored = np.zeros((H, W, 3), dtype=np.float32)
    instance_ids = np.unique(mask_ch)
    instance_ids = instance_ids[instance_ids != 0]

    np.random.seed(42)
    for inst_id in instance_ids:
        color = np.random.rand(3) * 0.8 + 0.2
        colored[mask_ch == inst_id] = color

    return colored


def count_total_cells(masks):
    """
    统计数据集中所有细胞实例的总数（排除背景通道）。

    参数:
        masks: np.ndarray, 形状为 (B, H, W, C)

    返回:
        total_cells: int
    """
    B, H, W, C = masks.shape
    total_cells = 0

    for b in range(B):
        for ch in range(C - 1):  # 跳过最后一个 Background 通道
            mask_ch = masks[b, :, :, ch]
            instance_ids = np.unique(mask_ch)
            instance_ids = instance_ids[instance_ids != 0]
            total_cells += len(instance_ids)

    return total_cells


def visualize(images, masks, center_masks, center_coords, indices, fold_prefix):
    """
    可视化指定索引的图：
      上排：原图（含文件名）| center_mask | 原图 + instance mask（无背景）+ 细胞中心
      下排：6 个通道分别可视化

    参数:
        images: np.ndarray, (B, H, W, 3)
        masks: np.ndarray, (B, H, W, C)
        center_masks: np.ndarray, (B, H, W)
        center_coords: list of list of tuple
        indices: list of int, 要可视化的图片索引
        fold_prefix: str, 数据部分前缀
    """
    channel_names = [
        "Neoplastic",
        "Inflammatory",
        "Connective/Soft tissue",
        "Dead Cells",
        "Epithelial",
        "Background",
    ]

    for i in indices:
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 6, figure=fig)

        img_name = f"{fold_prefix}_{i + 1}.png"
        npy_name = f"{fold_prefix}_{i + 1}.npy"

        img = images[i].copy()
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        # ---- 上排 ----
        # 原图
        ax0 = fig.add_subplot(gs[0, 0:2])
        ax0.imshow(img)
        ax0.set_title(f"{img_name}\nimage")
        ax0.axis("off")

        # Center Mask
        ax1 = fig.add_subplot(gs[0, 2:4])
        ax1.imshow(center_masks[i], cmap="gray")
        ax1.set_title(f"{npy_name}\ncenter_mask ({center_masks[i].sum()} cells)")
        ax1.axis("off")

        # 原图 + instance mask（无背景）+ 细胞中心
        ax2 = fig.add_subplot(gs[0, 4:6])
        inst_colored = colorize_instances(masks[i], skip_background=True)
        blended = 0.5*img + 0.5*inst_colored
        blended = np.clip(blended, 0, 1)
        ax2.imshow(blended)
        # 叠加细胞中心点
        coords = center_coords[i]
        if coords:
            ys, xs = zip(*coords)
            ax2.scatter(xs, ys, c="red", s=15, marker="o", linewidths=0.5, edgecolors="white")
        ax2.set_title("Overlay + Centers")
        ax2.axis("off")

        # ---- 下排：6个通道单独可视化 ----
        for ch in range(6):
            ax = fig.add_subplot(gs[1, ch])
            mask_ch = masks[i, :, :, ch]
            colored = colorize_single_channel(mask_ch)
            ax.imshow(colored)

            # Background 通道直接显示 0 cells
            if ch == 5:
                n_cells = 0
            else:
                n_cells = len(np.unique(mask_ch)) - (1 if 0 in mask_ch else 0)

            cell_str = "cell" if n_cells == 1 else "cells"
            ax.set_title(f"{channel_names[ch]}\n({n_cells} {cell_str})", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def get_fold_prefix(images_path):
    """
    从 images_path 中提取 fold 编号作为前缀。
    匹配路径中 'fold_' 后面紧跟的数字。

    例如:
        "D:\\Datasets\\PanNuke_original\\fold_2\\Fold 2\\images\\fold2\\images.npy" -> "2"
        "D:\\Datasets\\PanNuke_original\\fold_1\\Fold 1\\images\\fold1\\images.npy" -> "1"
    """
    match = re.search(r'fold_(\d+)', images_path, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"无法从路径中提取 fold 编号: {images_path}")


if __name__ == "__main__":
    # ===== 配置 =====
    save_root = r"D:\AA-Datasets\PanNuke"
    images_path = r"D:\AA-Datasets\PanNuke_original\fold_3\Fold 3\images\fold3\images.npy"
    masks_path = r"D:\AA-Datasets\PanNuke_original\fold_3\Fold 3\masks\fold3\masks.npy"

    # 自动从 images_path 中提取 fold 编号
    fold_prefix = get_fold_prefix(images_path)
    print(f"提取到 fold_prefix: {fold_prefix}")

    # ===== 加载数据 =====
    images = np.load(images_path)
    masks = np.load(masks_path)

    print(f"images 形状: {images.shape}")
    print(f"masks 形状: {masks.shape}")

    # ===== 提取细胞中心点 =====
    center_masks, center_coords = extract_cell_centers(masks)

    print(f"center_masks 形状: {center_masks.shape}")
    print(f"前三张图细胞数量: {len(center_coords[0])}, {len(center_coords[1])}, {len(center_coords[2])}")

    # ===== 统计总细胞数（排除背景） =====
    total_cells = count_total_cells(masks)
    print(f"该 fold 数据集中总细胞数量（排除背景）: {total_cells}")

    # ===== 保存图片和 dot_maps =====
    save_images_and_dot_maps(images, center_masks, save_root, fold_prefix)

    # ===== 可视化前3张和最后3张 =====
    B = images.shape[0]
    vis_indices = list(range(3)) + list(range(B - 3, B))
    print(f"可视化索引: {vis_indices}")
    visualize(images, masks, center_masks, center_coords, vis_indices, fold_prefix)