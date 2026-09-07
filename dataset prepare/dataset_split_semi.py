# -*- coding:utf-8 -*-
"""
Author：R
Date：25-04-2026
"""
import os
import shutil
import random
import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(directory):
    directory = Path(directory)
    return sorted([f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])


def collect_dot_maps(directory):
    directory = Path(directory)
    return sorted([f for f in directory.iterdir() if f.suffix.lower() == ".npy"])


def check_and_pair(images_dir, dot_maps_dir):
    image_files = collect_images(images_dir)
    dot_files = collect_dot_maps(dot_maps_dir)

    n_images = len(image_files)
    n_dots = len(dot_files)

    if n_images == 0:
        raise RuntimeError(f"图片目录为空: {images_dir}")
    if n_dots == 0:
        raise RuntimeError(f"标注目录为空: {dot_maps_dir}")
    if n_images != n_dots:
        raise RuntimeError(
            f"图片数量 ({n_images}) 与标注数量 ({n_dots}) 不一致！\n"
            f"  图片目录: {images_dir}\n"
            f"  标注目录: {dot_maps_dir}"
        )

    return list(zip(image_files, dot_files))


def print_pair_preview(samples, tag=""):
    total = len(samples)
    print(f"\n{'=' * 60}")
    print(f"  {tag}配对预览（前5个 + 后5个），请确认是否正确：")
    print(f"{'=' * 60}")
    preview_indices = list(range(min(5, total)))
    if total > 5:
        preview_indices += list(range(max(5, total - 5), total))
    for i in preview_indices:
        img_name = samples[i][0].name
        dot_name = samples[i][1].name
        print(f"  [{i:>4d}]  {img_name}  <-->  {dot_name}")
    print(f"{'=' * 60}\n")


def create_output_dirs(output_dir):
    dirs = {
        "train_labeled_img":   output_dir / "train" / "labeled" / "images",
        "train_labeled_dot":   output_dir / "train" / "labeled" / "dot_maps",
        "train_unlabeled_img": output_dir / "train" / "unlabeled" / "images",
        "test_img":            output_dir / "test" / "images",
        "test_dot":            output_dir / "test" / "dot_maps",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def transfer_labeled(sample_list, img_dir, dot_dir, transfer_fn):
    for img_path, npy_path in sample_list:
        transfer_fn(str(img_path), str(img_dir / img_path.name))
        transfer_fn(str(npy_path), str(dot_dir / npy_path.name))


def transfer_unlabeled(sample_list, img_dir, transfer_fn):
    for img_path, _ in sample_list:
        transfer_fn(str(img_path), str(img_dir / img_path.name))


def print_result(output_dir, n_labeled_train, n_unlabeled_train, n_test):
    print("\n生成的目录结构:")
    print(f"  {output_dir.name}/")
    print(f"  ├── train/")
    print(f"  │   ├── labeled/")
    print(f"  │   │   ├── images/      ({n_labeled_train} 张)")
    print(f"  │   │   └── dot_maps/    ({n_labeled_train} 个)")
    print(f"  │   └── unlabeled/")
    print(f"  │       └── images/      ({n_unlabeled_train} 张)")
    print(f"  └── test/")
    print(f"      ├── images/          ({n_test} 张)")
    print(f"      └── dot_maps/        ({n_test} 个)")


def detect_dataset_type(root_dir):
    root_dir = Path(root_dir)

    has_train_test = (root_dir / "train").is_dir() and (root_dir / "test").is_dir()
    has_flat = (root_dir / "images").is_dir() and (root_dir / "dot_maps").is_dir()

    if has_train_test and has_flat:
        print("⚠️ 同时检测到 train/test 目录和 images/dot_maps 目录，优先按已划分(presplit)模式处理")
        return "presplit"
    elif has_train_test:
        return "presplit"
    elif has_flat:
        return "flat"
    else:
        raise RuntimeError(
            f"无法识别数据集结构！目录 {root_dir} 下既没有 images/+dot_maps/，"
            f"也没有 train/+test/ 子目录。"
        )


def split_flat_dataset(root_dir, output_dir, train_ratio, labeled_ratio, seed, copy):
    random.seed(seed)
    root_dir = Path(root_dir)

    images_dir = root_dir / "images"
    dot_maps_dir = root_dir / "dot_maps"

    samples = check_and_pair(images_dir, dot_maps_dir)
    total = len(samples)

    print(f"[flat 模式] 共找到 {total} 个样本")
    print_pair_preview(samples, tag="全部数据 ")

    # 随机打乱并划分
    random.shuffle(samples)

    # 第一步：train / test
    n_train = max(1, int(total * train_ratio))
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    # 第二步：train 内部 labeled / unlabeled
    n_labeled_train = max(1, int(n_train * labeled_ratio))
    labeled_train = train_samples[:n_labeled_train]
    unlabeled_train = train_samples[n_labeled_train:]

    print(f"  总样本数        : {total}")
    print(f"  训练集 (train_ratio={train_ratio})    : {n_train}")
    print(f"    ├── 有标签 (labeled_ratio={labeled_ratio}) : {len(labeled_train)}")
    print(f"    └── 无标签                                  : {len(unlabeled_train)}")
    print(f"  测试集 (全部有标签)                            : {len(test_samples)}")

    dirs = create_output_dirs(output_dir)
    transfer_fn = shutil.copy2 if copy else shutil.move
    action_name = "复制" if copy else "移动"

    print(f"\n正在{action_name}文件...")
    transfer_labeled(labeled_train, dirs["train_labeled_img"], dirs["train_labeled_dot"], transfer_fn)
    transfer_unlabeled(unlabeled_train, dirs["train_unlabeled_img"], transfer_fn)
    transfer_labeled(test_samples, dirs["test_img"], dirs["test_dot"], transfer_fn)
    print("完成！")

    print_result(output_dir, len(labeled_train), len(unlabeled_train), len(test_samples))


def split_presplit_dataset(root_dir, output_dir, labeled_ratio, seed, copy):
    random.seed(seed)
    root_dir = Path(root_dir)

    train_images_dir = root_dir / "train" / "images"
    train_dot_maps_dir = root_dir / "train" / "dot_maps"
    test_images_dir = root_dir / "test" / "images"
    test_dot_maps_dir = root_dir / "test" / "dot_maps"

    for d in [train_images_dir, train_dot_maps_dir, test_images_dir, test_dot_maps_dir]:
        if not d.exists():
            raise FileNotFoundError(f"找不到目录: {d}")

    train_samples = check_and_pair(train_images_dir, train_dot_maps_dir)
    test_samples = check_and_pair(test_images_dir, test_dot_maps_dir)

    n_train = len(train_samples)
    n_test = len(test_samples)

    print(f"[presplit 模式] 训练集 {n_train} 个样本，测试集 {n_test} 个样本")
    print_pair_preview(train_samples, tag="训练集 ")

    random.shuffle(train_samples)
    n_labeled_train = max(1, int(n_train * labeled_ratio))
    labeled_train = train_samples[:n_labeled_train]
    unlabeled_train = train_samples[n_labeled_train:]

    print(f"  训练集总数      : {n_train}")
    print(f"    ├── 有标签 (labeled_ratio={labeled_ratio}) : {len(labeled_train)}")
    print(f"    └── 无标签                                  : {len(unlabeled_train)}")
    print(f"  测试集 (不变，全部有标签)                      : {n_test}")

    dirs = create_output_dirs(output_dir)
    transfer_fn = shutil.copy2 if copy else shutil.move
    action_name = "复制" if copy else "移动"

    print(f"\n正在{action_name}文件...")
    transfer_labeled(labeled_train, dirs["train_labeled_img"], dirs["train_labeled_dot"], transfer_fn)
    transfer_unlabeled(unlabeled_train, dirs["train_unlabeled_img"], transfer_fn)
    transfer_labeled(test_samples, dirs["test_img"], dirs["test_dot"], transfer_fn)
    print("完成！")

    print_result(output_dir, len(labeled_train), len(unlabeled_train), n_test)


def split_dataset(
        root_dir,
        output_dir=None,
        train_ratio=0.8,
        labeled_ratio=0.2,
        seed=42,
        copy=True,
):

    root_dir = Path(root_dir).resolve()

    if output_dir is None:
        percent_str = f"{int(labeled_ratio * 100)}%"
        output_dir = root_dir.parent / (root_dir.name + f"_{percent_str}")
    else:
        output_dir = Path(output_dir).resolve()

    print(f"输入目录: {root_dir}")
    print(f"输出目录: {output_dir}\n")

    dataset_type = detect_dataset_type(root_dir)
    print(f"检测到数据集类型: {dataset_type}\n")

    if dataset_type == "flat":
        split_flat_dataset(root_dir, output_dir, train_ratio, labeled_ratio, seed, copy)
    elif dataset_type == "presplit":
        print(f"ℹ️  presplit 模式下 train_ratio={train_ratio} 被忽略（使用数据集自带的 train/test 划分）\n")
        split_presplit_dataset(root_dir, output_dir, labeled_ratio, seed, copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将数据集划分为 train(labeled+unlabeled) / test")

    # ⭐️  --dataset 参数，配合 --root_dir 自动拼接路径
    parser.add_argument("--root_dir", type=str,
                        default=r"F:\AAAAA-Semi-Supervised-Cell-Detection",
                        help="项目根目录路径（datasets 文件夹所在目录）")
    parser.add_argument("--dataset", type=str, default="VGG",
                        help="数据集名称，如 VGG,BCD,UniCD（位于 root_dir/datasets/ 下）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录路径（默认为 数据集目录 + '_XX%%'，如 VGG_20%%）")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集占总数据的比例，仅 flat 类型有效 (默认 0.8)")
    parser.add_argument("--labeled_ratio", type=float, default=0.2,
                        help="训练集中有标签数据的比例 (默认 0.2)")
    parser.add_argument("--seed", type=int, default=3,
                        help="随机种子 (默认 42)")
    parser.add_argument("--move", action="store_true",
                        help="移动文件而不是复制（默认复制）")

    args = parser.parse_args()

    # ⭐️ 根据 dataset 名称自动拼接数据集路径
    dataset_path = os.path.join(args.root_dir, "datasets", args.dataset)

    split_dataset(
        root_dir=dataset_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
        copy=not args.move,
    )