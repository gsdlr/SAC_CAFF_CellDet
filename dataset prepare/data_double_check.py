# -*- coding:utf-8 -*-
"""
Author：LIU Rui
Date：2022/11/25
"""
import os
import pandas as pd
import cv2
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import h5py
from PIL import Image

"""
This module is only used for double check.
"""


def gt_counter(dspath, data_name):
    """
    To count the numbers of dots in the annotations. Only used for double checking.
    """
    global dot_map, fig, axes, rows, columns
    file_path_gt_count = dspath + '/' + data_name + '_gt_count.xlsx'
    df = pd.DataFrame()
    df.to_excel(file_path_gt_count)

    datapath = dspath + '/data'
    all_paths = os.listdir(datapath)
    images_files = sorted([path for path in all_paths if "cell" in path])
    annos_files = sorted([path for path in all_paths if "dots" in path])

    gt_count_list = []
    den_map_count_list = []
    i = 0
    flag1 = 0
    flag2 = 0
    for anno_file in annos_files:
        anno_file_path = os.path.join(datapath, anno_file)
        anno = cv2.imread(anno_file_path)  # cv2.imread读取出来的是ndarray格式

        if data_name == 'VGG':
            dot_map = (anno[:, :, 2] == 255) * 1
        elif data_name == 'MBM':
            dot_map = (anno[:, :, 2] == 255) * 1
        elif data_name == 'ADI':
            dot_map = (anno[:, :, 2] > 252) * 1
        elif data_name == 'DCC':
            dot_map = (anno[:, :, 2] < 180) * 1

        den_map = ndimage.gaussian_filter(dot_map.astype(np.float32), sigma=(3, 3), order=0)

        # visualize to check the imgs and den_maps
        if flag1 == 0:
            rows = 2
            columns = 3
            # 返回一个 Figure实例fig 和一个 AxesSubplot实例ax。fig代表整个图像，ax代表坐标轴,是array形式的，array的每个元素代表一个坐标轴。
            fig, axes = plt.subplots(rows, columns)
            fig.suptitle('visualization')
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_ylabel("pix", fontsize=5)
                    ax.set_xlabel("pix", fontsize=5)
                    ax.tick_params(labelsize=5)
            flag1 += 1

        if i < rows:
            img_file_path = os.path.join(datapath, images_files[i])
            img = cv2.imread(img_file_path)  # 打开图像,opencv默认读取图片的数据为: (高，宽，通道(B，G，R))
            # img = img[:, :, ::-1]  # 将通道进行调整（R,G,B）, [::-1] 代表顺序相反操作
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 这种方法也可以转成RGB
            print('The shape of img is: ', img.shape)
            print('The shape of anno is: ', anno.shape)
            print('The shape of den_map is: ', den_map.shape)
            # axes是一个（1*2）的一维array，axes的操作类似array
            # column one: input images, column two: den_map
            axes[i][0].set_title('{} imgs sample'.format(data_name), fontsize=8)
            plt.subplot(rows, columns, columns * i + 1)
            plt.imshow(img)
            axes[i][1].set_title('{} anno_file'.format(data_name), fontsize=8)
            plt.subplot(rows, columns, columns * i + 2)
            plt.imshow(anno)
            axes[i][2].set_title('{} den_map'.format(data_name), fontsize=8)
            plt.subplot(rows, columns, columns * i + 3)
            plt.imshow(den_map)
            i += 1

        if i >= rows and flag2 == 0:
            fig.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距
            plt.show()
            flag2 += 1

        gt_count = dot_map.sum()
        den_map_count = den_map.sum()
        gt_count_list.append(gt_count)
        den_map_count_list.append(den_map_count)

    output_excel = {'images': images_files, 'dot_maps': annos_files, 'gt_count': gt_count_list,
                    'den_map_count': den_map_count_list}
    output = pd.DataFrame(output_excel)

    writer = pd.ExcelWriter(file_path_gt_count, mode='a', engine='openpyxl')  # mode='a'的作用是追加新的sheet时不覆盖之前的数据
    output.to_excel(writer, sheet_name='gt_count', index=False)  # 将要加进来的数据写入writer
    writer.close()


def gt_counter_BCD(dspath, data_name, dtype='annotations'):
    """
    To count the numbers of coordinate in the annotations or the dot in the dot_map. Only used for double checking.
    """
    global file_path_gtcount, gt_count
    splits = ['train', 'validation', 'test']
    categories = ['positive', 'negative']

    if dtype == 'annotations':
        file_path_gtcount = dspath + '/' + data_name + '_anno_gt_count.xlsx'
    elif dtype == 'dot_maps':
        file_path_gtcount = dspath + '/' + data_name + '_dm_gt_count.xlsx'
    df = pd.DataFrame()  # 只是一个圆括号的话说明创建的数据表是空的
    df.to_excel(file_path_gtcount)

    output_excel_all = {}
    all_count_list = []
    for split in splits:
        output_excel = {}
        for category in categories:
            path = dspath + '/' + dtype + '/' + split + '/' + category
            files = sorted(os.listdir(path))
            gt_count_list = []
            for file in files:
                gt_path = os.path.join(path, file)
                if dtype == 'annotations':
                    anno = h5py.File(gt_path)
                    coordinates = np.asarray(anno['coordinates'])  # coordinates是一个二维ndarray,行数为点的个数，列数为2（即x,y坐标）
                    gt_count = coordinates.shape[0]
                if dtype == 'dot_maps':
                    dot_map = np.load(gt_path)
                    gt_count = dot_map.sum()
                gt_count_list.append(gt_count)
            output_excel[split + '_' + category + '_files'] = files
            output_excel[split + '_' + category + '_gt_count'] = gt_count_list

        # 写入每个split的gtcount
        output = pd.DataFrame(output_excel)
        writer = pd.ExcelWriter(file_path_gtcount, mode='a', engine='openpyxl')  # mode='a'的作用是追加新的sheet时不覆盖之前的数据
        output.to_excel(writer, sheet_name='{}_gt_count'.format(split), index=False)  # 将要加进来的数据写入writer
        writer.close()

        # 计算每张dotmap中总的count，即positive的count和negative的count之和
        pos_count = output_excel[split + '_' + 'positive' + '_gt_count']
        neg_count = output_excel[split + '_' + 'negative' + '_gt_count']
        all_count = np.sum([pos_count, neg_count], axis=0)  # 对应元素相加
        all_count_list.extend(all_count)

    # 写入总的gtcount
    output_excel_all['all_gt_count'] = all_count_list
    output_all = pd.DataFrame(output_excel_all)
    writer = pd.ExcelWriter(file_path_gtcount, mode='a', engine='openpyxl')  # mode='a'的作用是追加新的sheet时不覆盖之前的数据
    output_all.to_excel(writer, sheet_name='{}_all_gt_count'.format(data_name), index=False)  # 将要加进来的数据写入writer
    writer.close()


def check_diff_BCData(dspath):
    """
    Find the differences between the annotations and dot_maps. We find that in same annotations, there are duplicate
    coordinates. Therefore, the gt_count in the dot_maps is smaller than that in these annotations.
    """
    global coordinates_unique
    num_diff = 0  # The munber of annotations with duplicate coordinates.
    flag = 0
    splits = ['train', 'validation', 'test']
    categories = ['positive', 'negative']
    for split in splits:
        for category in categories:
            anno_path = dspath + '/annotations' + '/' + split + '/' + category
            anno_files = os.listdir(anno_path)
            for anno_file in anno_files:
                anno_file_path = os.path.join(anno_path, anno_file)
                anno = h5py.File(anno_file_path)
                coordinates = np.asarray(anno['coordinates'])
                gt_count = coordinates.shape[0]

                # generate dot_map
                dot_map = np.zeros((640, 640), np.uint8)  #
                for coordinate in coordinates:
                    x, y = coordinate[0], coordinate[1]
                    # 注意坐标轴的变换，原图应该是matlab中label出来的，跟ndarray的index正好互换
                    dot_map[y, x] = 1

                if gt_count != dot_map.sum():
                    print(anno_file_path)
                    print('coordinates_count={}'.format(len(coordinates)))
                    print('dot_map_count={}'.format(dot_map.sum()))

                    # find the unique coordinate on an anno file
                    s = set()
                    for coordinate in coordinates:
                        s.add(tuple(coordinate))
                        coordinates_unique = np.array(list(s))
                    print('coordinates_unique_count={}'.format(len(coordinates_unique)))
                    num_diff += 1

                    # 打印三个anno file 的coordinates来double-check重复坐标情况
                    if flag < 3:
                        print(coordinates)
                        coor_list = coordinates.tolist()
                        for i in range(len(coor_list)):
                            if coor_list.count(coor_list[i]) > 1:
                                print(np.asarray(coor_list[i]))  # duplicate coordinates
                        flag += 1

    print('num_diff={}'.format(num_diff))


def main():
    if data_name == 'VGG' or data_name == 'MBM' or data_name == 'ADI' or data_name == 'DCC':
        gt_counter(dspath, data_name)

    if data_name == 'BCData' and task == 'generate':
        gt_counter_BCD(dspath, data_name, dtype)

    if data_name == 'BCData' and task == 'check':
        check_diff_BCData(dspath)


if __name__ == '__main__':
    dspath = 'E:/AAAAA Object Counting/supporting_materials/dataset_reorganization/DCC'
    data_name = 'DCC'
    dtype = 'annotations'
    task = 'check'
    main()
