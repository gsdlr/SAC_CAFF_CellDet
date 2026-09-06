import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

# 路径设置
source_dir = r"D:\AA-Datasets\BM_original\source"
label_dir = r"D:\AA-Datasets\BM_original\annotations"
save_img_dir = r"D:\AA-Datasets\BM\images"
save_dot_dir = r"D:\AA-Datasets\BM\dot_maps"

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_dot_dir, exist_ok=True)

# 获取文件列表并排序
source_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

assert len(source_files) == len(label_files), \
    f"File count mismatch: source={len(source_files)}, label={len(label_files)}"

print(f"Total original images: {len(source_files)}")
print("=" * 60)

# Crop参数
patch_size = 320
img_size = 1200
patches_per_dim = img_size // patch_size  # 3
print(f"Patch size: {patch_size}x{patch_size}")
print(f"Patches per dimension: {patches_per_dim}")
print(f"Patches per image: {patches_per_dim * patches_per_dim}")
print(f"Coverage per dimension: {patches_per_dim * patch_size}/{img_size}")
print("=" * 60)

# Crop并保存
total_cells_before = 0
total_cells_after = 0

for idx, (src_name, lbl_name) in enumerate(zip(source_files, label_files)):
    img = np.array(Image.open(os.path.join(source_dir, src_name)))
    label = np.array(Image.open(os.path.join(label_dir, lbl_name)))

    cells_before = np.sum(label > 0)
    total_cells_before += cells_before
    print(f"\n[Image {idx + 1}/{len(source_files)}] {src_name}")
    print(f"  Cell count before crop: {cells_before}")

    patch_count = 0

    for row in range(patches_per_dim):
        for col in range(patches_per_dim):
            y_start = row * patch_size
            x_start = col * patch_size
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            img_patch = img[y_start:y_end, x_start:x_end]
            label_patch = label[y_start:y_end, x_start:x_end]

            patch_count += 1
            patch_cells = np.sum(label_patch > 0)
            total_cells_after += patch_cells

            save_name = f"BM_{idx + 1}_{patch_count}"
            img_save_path = os.path.join(save_img_dir, f"{save_name}.png")
            dot_save_path = os.path.join(save_dot_dir, f"{save_name}.npy")

            Image.fromarray(img_patch).save(img_save_path)
            np.save(dot_save_path, label_patch)

            print(f"    Patch {patch_count} ({save_name}): {patch_cells} cells")

print("\n" + "=" * 60)
print(f"Total cells before crop (all images): {total_cells_before}")
print(f"Total cells after crop (all patches): {total_cells_after}")
print(f"Total patches saved: {len(source_files) * patches_per_dim * patches_per_dim}")
print("=" * 60)

# ============================================================
# 可视化前两张图
# ============================================================
radius = 5
selem = disk(radius)

# --- Crop前可视化：一页2行3列 ---
print("\nVisualizing first 2 images (before crop)...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for row in range(2):
    src_name = source_files[row]
    lbl_name = label_files[row]

    img = np.array(Image.open(os.path.join(source_dir, src_name)))
    label = np.array(Image.open(os.path.join(label_dir, lbl_name)))
    cell_count = np.sum(label > 0)

    binary_label = label > 0
    dilated_label = binary_dilation(binary_label, structure=selem)

    axes[row, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
    axes[row, 0].set_title(f"Image: {src_name}", fontsize=9)
    axes[row, 0].axis('off')

    axes[row, 1].imshow(label, cmap='gray')
    axes[row, 1].set_title(f"Label: {lbl_name}", fontsize=9)
    axes[row, 1].axis('off')

    axes[row, 2].imshow(img, cmap='gray' if img.ndim == 2 else None)
    overlay = np.zeros((*dilated_label.shape, 4))
    overlay[dilated_label] = [1, 0, 0, 0.7]
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f"Overlay (cells={cell_count})", fontsize=9)
    axes[row, 2].axis('off')

plt.tight_layout()
plt.show()

# --- Crop后可视化：每页3行3列 ---
print("Visualizing first 2 images (after crop)...")

for img_idx in range(2):
    img = np.array(Image.open(os.path.join(source_dir, source_files[img_idx])))
    label = np.array(Image.open(os.path.join(label_dir, label_files[img_idx])))

    patches = []
    for row in range(patches_per_dim):
        for col in range(patches_per_dim):
            y_start = row * patch_size
            x_start = col * patch_size
            img_patch = img[y_start:y_start + patch_size, x_start:x_start + patch_size]
            label_patch = label[y_start:y_start + patch_size, x_start:x_start + patch_size]
            patch_count = row * patches_per_dim + col + 1
            save_name = f"BM_{img_idx + 1}_{patch_count}"
            patches.append((img_patch, label_patch, save_name))

    rows_per_page = 3
    num_pages = int(np.ceil(len(patches) / rows_per_page))

    for page in range(num_pages):
        fig, axes = plt.subplots(rows_per_page, 3, figsize=(15, 12))

        for row in range(rows_per_page):
            p_idx = page * rows_per_page + row

            if p_idx >= len(patches):
                for col in range(3):
                    axes[row, col].axis('off')
                continue

            img_patch, label_patch, save_name = patches[p_idx]
            patch_cells = np.sum(label_patch > 0)

            binary_label = label_patch > 0
            dilated_label = binary_dilation(binary_label, structure=selem)

            axes[row, 0].imshow(img_patch, cmap='gray' if img_patch.ndim == 2 else None)
            axes[row, 0].set_title(f"{save_name}.png", fontsize=9)
            axes[row, 0].axis('off')

            axes[row, 1].imshow(label_patch, cmap='gray')
            axes[row, 1].set_title(f"{save_name}.npy (cells={patch_cells})", fontsize=9)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(img_patch, cmap='gray' if img_patch.ndim == 2 else None)
            overlay = np.zeros((*dilated_label.shape, 4))
            overlay[dilated_label] = [1, 0, 0, 0.7]
            axes[row, 2].imshow(overlay)
            axes[row, 2].set_title(f"Overlay (cells={patch_cells})", fontsize=9)
            axes[row, 2].axis('off')

        plt.tight_layout()
        plt.show()

print("Done!")