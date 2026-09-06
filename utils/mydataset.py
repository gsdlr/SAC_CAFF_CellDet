# -*- coding:utf-8 -*-
"""
Author：LIU Rui
Date：2022/11/08
"""
import numpy as np
from PIL import Image
from torch.utils import data
import scipy.ndimage as ndimage
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import random
from typing import Tuple
from matplotlib.patches import Circle

# mean and std calculation
import cv2


class Dataset(data.Dataset):

    def __init__(self, imgs_paths, dot_maps_paths, dataset, mode, scale=100.0, vis=False):
        self.imgs_paths = imgs_paths
        self.dot_maps_paths = dot_maps_paths
        self.dataset = dataset
        self.mode = mode
        self.scale = scale  # 缩放因子，缓解截断误差
        self.vis = vis  # 将训练图片用于原理可视化时,不需要进行预处理,因为预处理有crop/翻转等

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_target = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, index):
        image = Image.open(self.imgs_paths[index]).convert('RGB')
        dot_map = np.load(self.dot_maps_paths[index]).astype(np.float32)

        dot_map[dot_map > 0] = 1.0

        image = np.array(image)

        crop_size_h = _get_crop_size(size=image.shape[0])
        crop_size_w = _get_crop_size(size=image.shape[1])

        img_crop = MyRandomCrop(crop_size=(crop_size_h, crop_size_w))
        data_preprocess = DataPreprocess(dataset=self.dataset)

        if self.mode == 'train' and not self.vis:
            image, dot_map = img_crop(image, dot_map)
            image, dot_map = data_preprocess(image, dot_map)

        # 对MBM测试集的图片的尺寸pad到16的倍数(600x600->608x608)，防止模型下采后concat时出现图像大小不匹配
        if 'MBM' in self.dataset:
            if self.mode == 'test':
                image = np.array(image)
                image = np.pad(image, ((4, 4), (4, 4), (0, 0)), 'constant')  # 高维pad，在H和W上面pad左右各1个像素
                dot_map = np.pad(dot_map, ((4, 4), (4, 4)), 'constant')

        # 对PSU测试集的图片的尺寸pad到16的倍数(612x452->624x464)，防止模型下采后concat时出现图像大小不匹配
        if 'PSU' in self.dataset:
            if self.mode == 'test':
                image = np.array(image)
                image = np.pad(image, ((6, 6), (6, 6), (0, 0)), 'constant')  # 高维pad，在H和W上面pad左右各1个像素
                dot_map = np.pad(dot_map, ((6, 6), (6, 6)), 'constant')

        # ===========ground truth生成===============
        h, w = dot_map.shape
        coords = np.argwhere(dot_map > 0)  # 细胞点坐标                          # 提前获取坐标

        # 保护：如果当前图/crop中没有任何标注点，直接返回全零图
        if len(coords) == 0:
            inv_dist_map = np.zeros((h, w), dtype=np.float32)
        else:
            # 1. 构建距离变换的输入：细胞点位置为 0，背景为 255
            dist_input = np.ones((h, w), dtype=np.uint8) * 255
            for (r, c) in coords:
                dist_input[r, c] = 0
            # 2. 欧氏距离变换
            dist_map = cv2.distanceTransform(dist_input, cv2.DIST_L2, 0)
            # 3. clip防止exp溢出
            dist_map = np.clip(dist_map, 0, 300)
            inv_dist_map = (1.0 / (0.0001 + np.exp(0.2 * dist_map))).astype(np.float32) * self.scale

        den_map = ndimage.gaussian_filter(dot_map.astype(np.float32), sigma=(6, 6), order=0) * self.scale  # 26联合基金, sigma=10 for BCD, =6 for UniCD

        image = self.transform(image)
        DotM = self.transform_target(dot_map)
        DenM = self.transform_target(den_map)
        IDistM = self.transform_target(inv_dist_map) # GT

        return image, DotM, DenM, IDistM


class UnlabeledDataset(data.Dataset):

    def __init__(self, imgs_paths, dataset):
        self.imgs_paths = imgs_paths
        self.dataset = dataset

        # 用于可视化的原图 transform：只 ToTensor，不做任何颜色变换或归一化
        self.raw_transform = transforms.Compose([
            transforms.ToTensor(),  # [0,255] → [0,1], HWC → CHW
        ])

        # teacher 用的弱增强：只做归一化
        self.weak_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # student 用的强增强：加更多扰动
        self.strong_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, index):
        image = Image.open(self.imgs_paths[index]).convert('RGB')
        image = np.array(image)

        # 将图像裁剪到最近的16的倍数（向下取整，最多去掉15像素边缘）
        h, w = image.shape[0], image.shape[1]
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        image = image[:new_h, :new_w, :]

        # 保存未增强的原图（用于可视化），此时已是16倍数尺寸
        img_raw = self.raw_transform(Image.fromarray(image))  # [3, new_h, new_w], [0,1]

        # 随机翻转（两个版本共享同一个几何变换，只在颜色/噪声上不同）
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
        if random.random() > 0.5:
            image = np.flipud(image).copy()

        # 转成 PIL 给 torchvision transforms 用
        image_pil = Image.fromarray(image)

        # 同一张图，两种不同的增强
        img_weak = self.weak_transform(image_pil)      # 给 teacher
        img_strong = self.strong_transform(image_pil)   # 给 student

        # 返回 3 个元素：原图（可视化）、弱增强（teacher）、强增强（student）
        return img_raw, img_weak, img_strong

def _get_crop_size(size, divisor=16):
    """
    This function ensures that the crop_size can be divisible by the divisor.
    """
    # crop to 0.875 of the original size.
    new_size = 0.875 * size
    if new_size % divisor != 0:
        new_size = new_size // divisor * divisor
    return int(new_size)

# 轻度颜色增强，仅作用于image，不影响label
class MyRandomColorJitter:
    """对图像施加轻度颜色抖动（亮度、对比度、饱和度、色调），不改变label"""
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5):
        self.p = p
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, img, label):
        if torch.rand(1) < self.p:
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_pil = self.color_jitter(img_pil)
            img = np.array(img_pil)
        return img.copy(), label.copy()


class DataPreprocess:
    def __init__(self, dataset, mytransforms=None):

        if mytransforms is None:
            if dataset == 'PSU':  # PSU不是正方形的，旋转之后会导致my_collate报错
                mytransforms = [
                    MyRandomHorizontalFlip(),
                    MyRandomVerticalFlip(),
                    # MyRandomColorJitter(),
                ]
            else:
                mytransforms = [
                    MyRandomHorizontalFlip(),
                    MyRandomVerticalFlip(),
                    MyRandomRotation(),
                    # MyRandomColorJitter(),
                ]

        self.mytranforms = mytransforms

    def __call__(self, img, label):
        for t in self.mytranforms:
            img, label = t(img, label)
        return img, label


class MyRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if torch.rand(1) < self.p:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)
        return img.copy(), label.copy()


class MyRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if torch.rand(1) < self.p:
            img = np.flip(img, axis=0)
            label = np.flip(label, axis=0)
        return img.copy(), label.copy()


class MyRandomRotation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        factor = random.randint(0, 4)
        if torch.rand(1) < self.p:
            img = np.rot90(img, factor)  # factor=1,2,3,4是分别逆时针旋转90，180,270,360度
            label = np.rot90(label, factor)
        return img.copy(), label.copy()


class MyRandomCrop:

    def __init__(self, crop_size: Tuple[int, int] = (128, 128)):
        self.crop_size = crop_size

    def __call__(self, img, label):  # img是shape为[h,w,c]的ndarray格式

        h, w = img.shape[0], img.shape[1]
        min_size = min(h, w)

        assert min_size >= self.crop_size[0], 'The image size is smaller than the crop_size'

        r0 = random.randint(0, h - self.crop_size[0])
        c0 = random.randint(0, w - self.crop_size[1])

        r1 = r0 + self.crop_size[0]
        c1 = c0 + self.crop_size[1]

        img = img[r0:r1, c0:c1]
        label = label[r0:r1, c0:c1]

        return img.copy(), label.copy()