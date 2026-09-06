# -*- coding:utf-8 -*-
"""
Author：LIU Rui
Date：2022/11/30
"""
import os
import numpy as np
import cv2
import h5py


def dot_map_generator(dspath, data_name):
    """
    Generate dot map .npy from the annotations.
    """
    global dot_map
    splits = ['train', 'test']

    for split in splits:
        dot_maps_path = dspath + '/' + split + '/dot_maps'
        if not os.path.exists(dot_maps_path):
            os.makedirs(dot_maps_path)
        annos_path = dspath + '/' + split + '/annotations'
        anno_files = os.listdir(annos_path)
        for anno_file in anno_files:
            anno_file_path = os.path.join(annos_path, anno_file)
            anno = cv2.imread(anno_file_path)  # cv2.imread读进来的是np.array格式

            if data_name == 'VGG':
                dot_map = (anno[:, :, 2] == 255) * 1
            elif data_name == 'MBM':
                dot_map = (anno[:, :, 2] == 255) * 1
            elif data_name == 'ADI':
                dot_map = (anno[:, :, 2] > 252) * 1
            elif data_name == 'DCC':
                dot_map = (anno[:, :, 2] < 180) * 1
            elif data_name == 'IMM':
                dot_map = (anno[:, :, 2] == 255) * 1

            # save the dot map. The '.npy' file can better save the array
            dot_map_path = os.path.join(dot_maps_path, anno_file.split('.')[0] + '.npy')
            np.save(dot_map_path, dot_map)


def check_image_size_BCData(dspath):
    num = 0
    splits = ['train', 'validation', 'test']
    for split in splits:
        imgs_path = dspath + '/images' + '/' + split
        img_files = os.listdir(imgs_path)
        for img_file in img_files:
            img_file_path = os.path.join(imgs_path, img_file)
            img = cv2.imread(img_file_path)
            print(img.shape)
            num += 1
            assert img.shape == (640, 640, 3), \
                AssertionError('The shape of {} is not (640, 640, 3)'.format(img))

    print('The number of images is {}'.format(num))
    print('The shape of each images is equal to (640, 640, 3)')


def dot_map_generator_BCData(dspath):
    """
    Generate dot map .npy from the annotations.
    """
    splits = ['train', 'validation', 'test']

    categories = ['positive', 'negative']
    for split in splits:
        for category in categories:
            dot_maps_path = dspath + '/dot_maps' + '/' + split + '/' + category
            if not os.path.exists(dot_maps_path):
                os.makedirs(dot_maps_path)
            annos_path = dspath + '/annotations' + '/' + split + '/' + category
            anno_files = os.listdir(annos_path)
            for anno_file in anno_files:
                anno_file_path = os.path.join(annos_path, anno_file)
                anno = h5py.File(anno_file_path)
                coordinates = np.asarray(anno['coordinates'])

                # generate dot_map
                dot_map = np.zeros((640, 640), np.uint8)
                for coordinate in coordinates:
                    x, y = coordinate[0], coordinate[1]
                    # 注意坐标轴的变换，原图应该是matlab中label出来的，跟ndarray的index正好互换
                    dot_map[y, x] = 1

                # save the dot map. The '.npy' file can better save the array
                dot_map_path = os.path.join(dot_maps_path, anno_file.split('.')[0] + '.npy')
                np.save(dot_map_path, dot_map)


def mian():
    if data_name == 'VGG' or data_name == 'MBM' or data_name == 'ADI' or data_name == 'DCC' or data_name == 'IMM':
        dot_map_generator(dspath, data_name)

    elif data_name == 'BCData' and task == 'check':
        check_image_size_BCData(dspath)

    elif data_name == 'BCData' and task == 'generate':
        dot_map_generator_BCData(dspath)


if __name__ == '__main__':
    dspath = 'E:\Images from Andy\Preprocess\setting2\selected_images_and_annotations'
    data_name = 'IMM'
    task = 'generate'
    mian()
