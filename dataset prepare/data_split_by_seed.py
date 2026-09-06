# -*- coding:utf-8 -*-
"""
Author：LIU Rui
Date：2023/03/04
"""

import glob
import numpy as np
import os
import shutil

"""
采用与训练过程中相同的seed来划分数据集，便于post-processing: visualization, 计算每一张测试图片的Error等等
"""


def data_split_by_seed(dataset, root, seed, split):
    images_path = ''
    dot_maps_path = ''
    if 'VGG' in dataset or 'MBM' in dataset:
        images_path = root + '/datasets/' + dataset + "/images/*.png"
        dot_maps_path = root + '/datasets/' + dataset + "/dot_maps/*.npy"
    if 'ADI' in dataset:
        images_path = root + '/datasets/' + dataset + "/images/*.jpeg"
        dot_maps_path = root + '/datasets/' + dataset + "/dot_maps/*.npy"

    images_all = sorted(glob.glob(images_path))
    dot_maps_all = sorted(glob.glob(dot_maps_path))

    np.random.seed(seed)
    index = np.random.permutation(len(images_all))
    images_all = np.array(images_all)[index]
    dot_maps_all = np.array(dot_maps_all)[index]

    images_train = sorted(images_all[:split[dataset]])
    dot_maps_train = sorted(dot_maps_all[:split[dataset]])
    images_test = sorted(images_all[split[dataset]:])
    dot_maps_test = sorted(dot_maps_all[split[dataset]:])

    dest = root + '/datasets/' + dataset + '_five_trials/' + 'seed_' + str(seed) + '/'
    if not os.path.exists(dest):
        os.makedirs(dest)

    train_image_path = dest + 'train/' + 'images/'
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    for image in images_train:
        image_name = image.split('\\')[-1]
        shutil.copy(image, os.path.join(train_image_path, image_name))

    train_dot_path = dest + 'train/' + 'dot_maps/'
    if not os.path.exists(train_dot_path):
        os.makedirs(train_dot_path)
    for dot_map in dot_maps_train:
        dot_map_name = dot_map.split('\\')[-1]
        shutil.copy(dot_map, os.path.join(train_dot_path, dot_map_name))

    test_image_path = dest + 'test/' + 'images/'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    for image in images_test:
        image_name = image.split('\\')[-1]
        shutil.copy(image, os.path.join(test_image_path, image_name))

    test_dot_path = dest + 'test/' + 'dot_maps/'
    if not os.path.exists(test_dot_path):
        os.makedirs(test_dot_path)
    for dot_map in dot_maps_test:
        dot_map_name = dot_map.split('\\')[-1]
        shutil.copy(dot_map, os.path.join(test_dot_path, dot_map_name))


if __name__ == '__main__':
    dataset = 'VGG'
    root = '..'  # 返回上级目录
    seeds = [1, 2, 3, 4, 5]
    split = {'VGG': 50, 'ADI': 50, 'MBM': 15}

    for seed in seeds:
        data_split_by_seed(dataset, root, seed, split)
