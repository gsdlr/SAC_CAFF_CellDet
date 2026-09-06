import os
from collections import Counter

import numpy as np
from PIL import Image

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def get_image_shape(img_path):
    """
    读取一张图片，返回其 (C, H, W)。
    C = 通道数，H = 高度，W = 宽度。
    """
    with Image.open(img_path) as img:
        arr = np.array(img)

    if arr.ndim == 2:
        # 灰度图，没有通道维度
        h, w = arr.shape
        c = 1
    elif arr.ndim == 3:
        h, w, c = arr.shape
    else:
        raise ValueError(f"无法解析的图片维度: {arr.shape}")

    return (c, h, w)


def scan_folder(folder, recursive=True):
    """
    遍历文件夹，统计所有图片的 (C, H, W) 尺寸分布。
    recursive=True 时会递归遍历子文件夹。
    """
    size_counter = Counter()
    total = 0
    failed = []

    if recursive:
        walker = os.walk(folder)
    else:
        # 只遍历当前层
        walker = [(folder, [], os.listdir(folder))]

    for root, _, files in walker:
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            try:
                shape = get_image_shape(fpath)
                size_counter[shape] += 1
                total += 1
            except Exception as e:
                failed.append((fpath, str(e)))

    return size_counter, total, failed


def main():
    folder = input("请输入图片文件夹路径: ").strip()

    if not os.path.isdir(folder):
        print(f"错误：'{folder}' 不是有效的文件夹。")
        return

    size_counter, total, failed = scan_folder(folder, recursive=True)

    print(f"\n共成功读取 {total} 张图片。")
    print(f"共发现 {len(size_counter)} 种不同的尺寸 (C × H × W)：\n")

    print(f"{'C × H × W':<25}{'数量':<10}{'占比':<10}")
    print("-" * 45)

    # 按数量从多到少排序
    for (c, h, w), count in size_counter.most_common():
        size_str = f"{c} × {h} × {w}"
        ratio = f"{count / total * 100:.2f}%" if total else "0%"
        print(f"{size_str:<25}{count:<10}{ratio:<10}")

    if failed:
        print(f"\n有 {len(failed)} 张图片读取失败：")
        for path, err in failed:
            print(f"  {path} -> {err}")


if __name__ == "__main__":
    main()


