# -*- coding:utf-8 -*-
"""
Author：R
Date：27-07-2025
"""
import cv2
import numpy as np
import os
import glob
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import scipy.ndimage as ndimage
from models.DistMGenerator import DistMGenerator
from utils.mydataset import Dataset
from utils.trainer_semi import my_collate
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决plt.imshow绘图时内核崩溃问题


# visualization
def visual_intermed_rep(dataset_name, imgs, DotMs, gts, DenMs, device, num=5):
    """
    Visualize the Input, GT, Output (Prediction) and the detection results
    """

    imgs, gts, DenMs, DotMs = imgs.to(device), gts.to(device), DenMs.to(device), DotMs.to(device)

    plt.figure(figsize=(10, 10))
    for i in range(num):
        row = num
        col = 4
        # col = 2

        # column one: input images
        plt.subplot(row, col, i * col + 1)
        img_np = imgs[i].permute(1, 2, 0).cpu().numpy()
        img_np = 0.5 * img_np + 0.5  # 还原normalization
        plt.title('input')
        if 'MBM' in dataset_name:  # MBM数据集padding到608，可视化时候还原到600
            plt.imshow(img_np[4:604, 4:604, :])
        elif 'ADI' in dataset_name:  # ADI数据集padding到160，可视化时候还原150
            plt.imshow(img_np[5:155, 5:155, :])
        else:
            plt.imshow(img_np)
        plt.axis('off')

        # column two: ground truth
        plt.subplot(row, col, i * col + 2)
        # gt_np = gts[i].permute(1, 2, 0).cpu().numpy()
        gt_np = gts[i].permute(1, 2, 0).cpu().detach().numpy()
        plt.title('GT detection map: {}'.format(DotM[i].sum()))
        # plt.title('DenM_pred: {}'.format(torch.round(gts[i].sum(), decimals=3)))  # 显示预测的密度图，以用于ablation中的画图
        if 'MBM' in dataset_name:  # MBM数据集padding到608，可视化时候还原到600
            plt.imshow(gt_np[4:604, 4:604, :])
        elif 'ADI' in dataset_name:  # ADI数据集padding到160，可视化时候还原150
            plt.imshow(gt_np[5:155, 5:155, :])
        else:
            plt.imshow(gt_np)
        plt.axis('off')

        # column three: density map
        plt.subplot(row, col, i * col + 3)
        denm_np = DenMs[i].permute(1, 2, 0).detach().cpu().numpy()
        plt.title('GT density map: {}'.format(DenMs[i].sum()))
        if 'MBM' in dataset_name:  # MBM数据集padding到608，可视化时候还原到600
            plt.imshow(denm_np[4:604, 4:604, :])
        elif 'ADI' in dataset_name:  # ADI数据集padding到160，可视化时候还原150
            plt.imshow(denm_np[5:155, 5:155, :])
        else:
            plt.imshow(denm_np)
        plt.axis('off')

        # column four: bounding-box annotation
        plt.subplot(row, col, i * col + 4)
        img_det = imgs[i].permute(1, 2, 0).detach().cpu().numpy()  # permute将[c, h, w]转为[h, w, c]
        img_det = cv2.cvtColor(img_det, cv2.COLOR_RGB2BGR)  # 先转换颜色通道，否是后面画的圆没办法显示，原因未知
        coords = torch.nonzero(DotM[i].squeeze()).cpu().numpy()  # 经过pytorch处理之后，DotM[i]的shape是[c, h ,w],其中c=1
        for coord in coords:
            color = (0.0, 1.0, 0.0)  # 颜色转成0~1的float格式，因为图片是这个格式的
            # cv2.circle(img_det, (coord[1], coord[0]), radius=10, color=color, thickness=2)
            cv2.rectangle(img_det, (coord[1]-20, coord[0]-20), (coord[1]+20, coord[0]+20), color=color, thickness=4)
        img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)  # 将颜色通道还原到RGB，方便plt显示
        img_det = 0.5 * img_det + 0.5  # 还原normalization
        plt.title('BBox')
        if 'MBM' in dataset_name:  # MBM数据集padding到608，可视化时候还原到600
            plt.imshow(img_det[4:604, 4:604, :])
        elif 'ADI' in dataset_name:  # ADI数据集padding到160，可视化时候还原150
            plt.imshow(img_det[5:155, 5:155, :])
        else:
            plt.imshow(img_det)
        plt.axis('off')

        # #column five: dot map
        # plt.subplot(row, col, i * col + 2)
        # dotm_np = DotMs[i].permute(1, 2, 0).cpu().numpy()
        # # 扩大点以便可视化
        # coordinates = np.array(np.nonzero(dotm_np)).transpose(1,0)
        # for coord in coordinates:
        #     x = coord[1]
        #     y = coord[0]
        #     dotm_np[y - 2: y + 1, x - 2: x + 1] = 1
        #
        # plt.title('dot map: {}'.format(DotM[i].sum()))
        # if 'MBM' in dataset_name:  # MBM数据集padding到608，可视化时候还原到600
        #     plt.imshow(dotm_np[4:604, 4:604, :], cmap='binary')  # 二值化确保纯黑白显示
        # elif 'ADI' in dataset_name:  # ADI数据集padding到160，可视化时候还原150
        #     plt.imshow(dotm_np[5:155, 5:155, :], cmap='binary')  # 二值化确保纯黑白显示
        # else:
        #     plt.imshow(dotm_np, cmap='binary')  # 二值化确保纯黑白
        #
        # # 添加边框
        # ax = plt.gca()
        # border = patches.Rectangle(
        #     (-1, -1),
        #     dotm_np.shape[1]+2,
        #     dotm_np.shape[0]+2,
        #     linewidth=1,
        #     edgecolor='gray',
        #     facecolor='none'
        # )
        # ax.add_patch(border)
        # plt.axis('off')

    # show the figure 在plt的显示中,背景值为0时, 三通道图像背景显示为黑色,单通道显示为深蓝色
    plt.show()


if __name__ == '__main__':
    print(torch.cuda.device_count())
    device_num = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_num.strip()  # set the visible gpu(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = 'IMM_for_NSFC'
    batch_size = 2
    root = 'F:\AAAAA-TB-FA-Cell-Detection-and-Counting'  # 根目录为当前文件夹
    img_paths = ''
    dot_map_paths = ''
    if 'VGG' in dataset or 'MBM' in dataset:  # 这里主要是为了可视化intermediate representations, 所以不管是train或者test都可以
        base_path = os.path.join(root, 'datasets', dataset)
        img_paths = sorted(glob.glob(os.path.join(base_path, 'images', '*.png')))
        dot_map_paths = sorted(glob.glob(os.path.join(base_path, 'dot_maps', '*.npy')))
    if 'BCD' in dataset:  # BCD数据集官方划分好了训练集和测试集
        base_path = os.path.join(root, 'datasets', dataset)
        img_paths = sorted(glob.glob(os.path.join(base_path, 'train', 'images', '*.png')))
        dot_map_paths = sorted(glob.glob(os.path.join(base_path, 'train', 'dot_maps', '*.npy')))

    if 'IMM' in dataset:  # 这里主要是为了可视化intermediate representations, 所以不管是train或者test都可以
        base_path = os.path.join(root, 'datasets', dataset)
        img_paths = sorted(glob.glob(os.path.join(base_path, 'images', '*.tif')))
        dot_map_paths = sorted(glob.glob(os.path.join(base_path, 'dot_maps', '*.npy')))

    #  这里将mode设置为test,因为在train模式下会有crop, 这里主要是可视化作用,所以不需要crop
    ds = Dataset(img_paths, dot_map_paths, dataset=dataset, mode='train', vis=True)
    dl = data.DataLoader(ds, batch_size=batch_size, collate_fn=my_collate)
    distm_generator = DistMGenerator().to(device)

    print(len(dl.dataset))

    for imgs, DotM, DenM, cells, coords in dl:
        imgs = imgs.to(device)
        DotM = DotM.to(device)
        DenM = DenM.to(device)
        cells = cells.to(device)
        coords = [coord.to(device) for coord in coords]

        # 这里的shape不能用imgs.shape,因为imgs是三通道的,而gt需要是单通道的
        DistM = distm_generator(coords, cells, shape=DenM.shape)

        # 这里主要是为了可视化DistM和DenM, 所以参数做了简化和替换，例如第二个DotM如果是可视化结果时应该为DotM_pred
        visual_intermed_rep(dataset, imgs, DotM, DistM/distm_generator.scale, DenM/dl.dataset.scale, device, num=2)
