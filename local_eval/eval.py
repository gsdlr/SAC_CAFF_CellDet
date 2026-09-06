import torch
import os
import sys
import numpy as np
from scipy import spatial as ss
import pdb
import cv2
from local_eval.utils import AverageMeter, DistMeter, hungarian
import argparse

"""
modified from https://github.com/dk-liang/FIDTM/tree/master/local_eval
"""


def local_metrics(pred_p_list, gt_p_list, sigma=5):
    metrics = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'dist': DistMeter()}

    # 列表为空时直接返回零指标（整个epoch没有样本）
    if len(pred_p_list) == 0 or len(gt_p_list) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    for i_sample in range(len(gt_p_list)):
        # init
        gt_p, pred_p, fn_gt_index, tp_pred_index, fp_pred_index = [], [], [], [], []
        tp, fp, fn = [0, 0, 0]
        dist = []

        # 统一获取当前样本的点数组，并确保至少是二维 (N, 2) 形状
        pred_pts = pred_p_list[i_sample]
        gt_pts = gt_p_list[i_sample]
        if not isinstance(pred_pts, np.ndarray):
            pred_pts = np.array(pred_pts)
        if not isinstance(gt_pts, np.ndarray):
            gt_pts = np.array(gt_pts)
        if pred_pts.ndim == 1:
            pred_pts = pred_pts.reshape(-1, 2) if pred_pts.size > 0 else np.empty((0, 2))
        if gt_pts.ndim == 1:
            gt_pts = gt_pts.reshape(-1, 2) if gt_pts.size > 0 else np.empty((0, 2))

        n_pred = pred_pts.shape[0]
        n_gt = gt_pts.shape[0]

        #GT和Pred都为空 → 该样本无需计算，直接跳过
        if n_gt == 0 and n_pred == 0:
            pass  # tp=fp=fn=0, dist=[], 不影响全局指标

        elif n_gt == 0 and n_pred != 0:
            # 修复原始 bug: pred_p_list.shape[0] → n_pred
            fp_pred_index = np.array(range(n_pred))
            fp = fp_pred_index.shape[0]

        elif n_pred == 0 and n_gt != 0:
            fn_gt_index = np.array(range(n_gt))
            fn = fn_gt_index.shape[0]

        else:  # n_gt != 0 and n_pred != 0
            # dist
            dist_matrix = ss.distance_matrix(pred_pts, gt_pts, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)

            # sigma_s and sigma_l
            tp, fp, fn, dist = compute_metrics(dist_matrix, match_matrix, n_pred, n_gt, sigma)

        metrics['tp'].update(tp)
        metrics['fp'].update(fp)
        metrics['fn'].update(fn)
        metrics['dist'].update1(dist)

    ap = metrics['tp'].sum / (metrics['tp'].sum + metrics['fp'].sum + 1e-20)  # precision
    ar = metrics['tp'].sum / (metrics['tp'].sum + metrics['fn'].sum + 1e-20)  # recall

    fm = 2 * ap * ar / (ap + ar + 1e-20)  # f1 score

    avg_dist = metrics['dist'].avg
    std_dist = metrics['dist'].std

    return ap, ar, fm, avg_dist, std_dist


def compute_metrics(dist_matrix, match_matrix, pred_num, gt_num, sigma):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    dist = []
    idx = np.nonzero(assign)
    coords = list(zip(idx[0], idx[1]))
    for coord in coords:
        row = coord[0]
        col = coord[1]
        dist.append(dist_matrix[row][col])

    return tp, fp, fn, dist
