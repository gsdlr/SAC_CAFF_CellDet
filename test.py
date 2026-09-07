# -*- coding:utf-8 -*-
"""
Author：R
Date：19-08-2022
"""

from utils.mydataset import Dataset
from utils.visualization import visualization_test
from local_eval.eval import local_metrics  # 引入定位指标计算
import os
import pandas as pd
import numpy as np
import cv2
import scipy.ndimage as ndimage
import torch
from torch.utils import data
import glob
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from models.FeaAlignNet_TB import MyResUDetectorTB
from models.CAFFNet import CAFFNet
import copy
import argparse
import math
from utils.config import bs, radius

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

args = None


def parser_args():
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('--root', default='.',
                        help='root dir')
    parser.add_argument("--exp_purpose", type=str, default="",
                        help="experiment purpose")
    parser.add_argument('--dataset', type=str, default='',
                        help='dataset',
                        choices=['PanNuke', 'BCD', 'UniCD'])
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--detector_weight', type=str, default='',
                        help='model dir')
    parser.add_argument("--detector_net", type=str, default='CAFFNet',
                        help="detection network", choices=['Detector', 'ResUDetector', 'MyResUDetector'])
    parser.add_argument('--device', default='2',
                        help='assign device')
    args = parser.parse_args()

    return args


def LMDS(inp, thr=0.2):
    inp = inp.clone()

    keep = nn.functional.max_pool2d(inp, (3, 3), stride=1, padding=1)
    keep = (keep == inp).float()
    inp = keep * inp

    B = inp.shape[0]
    for b in range(B):
        sample_max = inp[b].max().item()

        if sample_max < 1e-5:
            inp[b] = 0
            continue

        inp[b][inp[b] < thr * sample_max] = 0

    inp[inp > 0] = 1
    return inp


if __name__ == '__main__':
    print(torch.cuda.device_count())
    args = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 'best'
    seed = 1

    base_path = os.path.join(args.root, 'datasets', args.dataset)
    images_train = sorted(glob.glob(os.path.join(base_path, 'train', 'images', '*.png')))
    dot_maps_train = sorted(glob.glob(os.path.join(base_path, 'train', 'dot_maps', '*.npy')))
    imgs_test = sorted(glob.glob(os.path.join(base_path, 'test', 'images', '*.png')))
    dot_maps_test = sorted(glob.glob(os.path.join(base_path, 'test', 'dot_maps', '*.npy')))

    train_ds = Dataset(images_train, dot_maps_train, args.dataset, mode='train')
    train_dl = data.DataLoader(train_ds, batch_size=args.batch_size)
    test_ds = Dataset(imgs_test, dot_maps_test, args.dataset, mode='test')
    test_dl = data.DataLoader(test_ds, batch_size=args.batch_size)

    scale = test_dl.dataset.scale

    detector = globals()[args.detector_net]().to(device)
    checkpoint = torch.load(args.detector_weight, map_location=device)
    print(detector.load_state_dict(checkpoint, strict=True))

    test_result = './test_results' + '/' + args.exp_purpose
    if not os.path.exists(test_result):
        os.makedirs(test_result)

    file_path_test = test_result + '/' + args.dataset + '_test_record_seed_' + \
                     str(seed) + '_epoch_' + str(epoch) + '.xlsx'

    df = pd.DataFrame()
    df.to_excel(file_path_test)

    count_gt_list, count_pred_list = [], []
    pred_p_epoch_test, gt_p_epoch_test = [], []

    detector.eval()
    with torch.no_grad():
        for imgs, DotM, DenM, IDistM in tqdm(test_dl, desc="Testing"):
            imgs = imgs.to(device)
            DotM = DotM.to(device)
            DenM = DenM.to(device)
            IDistM = IDistM.to(device)

            IDistM_pred, _ = detector(imgs)
            DotM_pred = LMDS(IDistM_pred)

            count_gt = []
            count_pred = []
            pred_p_test, gt_p_test = [], []

            for i in range(len(imgs)):
                DotM_np = DotM[i].cpu().numpy()
                DotM_pred_np = DotM_pred[i].detach().cpu().numpy()

                count_gt.append(DotM_np.sum())
                count_pred.append(DotM_pred_np.sum())

                pred_p_test.append(torch.nonzero(DotM_pred[i].squeeze(), as_tuple=False).cpu().numpy().reshape(-1, 2))
                gt_p_test.append(torch.nonzero(DotM[i].squeeze(), as_tuple=False).cpu().numpy().reshape(-1, 2))

            count_gt_list.extend(count_gt)
            count_pred_list.extend(count_pred)
            pred_p_epoch_test.extend(pred_p_test)
            gt_p_epoch_test.extend(gt_p_test)

            visualization_test(args.dataset, imgs, IDistM / scale, IDistM_pred / scale, DotM, DotM_pred, device,
                               num=len(imgs))

    er_list = abs(np.array(count_gt_list) - np.array(count_pred_list))
    MAE = np.mean(er_list)
    STD = np.std(er_list)

    ap_epoch_test, ar_epoch_test, fm_epoch_test, _, _ = local_metrics(pred_p_epoch_test, gt_p_epoch_test, sigma=radius)

    print('\n=========================================')
    print('-----Counting Performance (Test)-----')
    print('MAE_test: ', round(MAE, 3))
    print('STD_test: ', round(STD, 3))

    print('-----Localization Performance (Test)-----')
    print('precision_test:', round(ap_epoch_test, 5))
    print('recall_test:', round(ar_epoch_test, 5))
    print('F-measure_test:', round(fm_epoch_test, 5))
    print('=========================================\n')

    output_details = pd.DataFrame({
        'image_name': imgs_test,
        'GT': count_gt_list,
        'Pred': count_pred_list,
        'Absolute_Error': er_list
    })

    output_summary = pd.DataFrame({
        'Metric': ['MAE', 'STD', 'Precision', 'Recall', 'F-measure'],
        'Value': [MAE, STD, ap_epoch_test, ar_epoch_test, fm_epoch_test]
    })

    writer = pd.ExcelWriter(file_path_test, mode='a', engine='openpyxl', if_sheet_exists='replace')

    output_details.to_excel(writer, sheet_name='test_details', index=False)

    output_summary.to_excel(writer, sheet_name='test_summary', index=False)

    writer.close()
    print(f"Results successfully saved to {file_path_test}")