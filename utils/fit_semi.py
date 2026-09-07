# -*- coding:utf-8 -*-
"""
Author：LIU Rui
Date：2022/11/08
"""
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure
from local_eval.eval import local_metrics
from utils.visualization import visualize_predictions_train, visualize_affinity_consistency
from utils.consistency import rampup, ema_decay_schedule, _orthogonal_select, extract_agents, compute_affinity_consistency
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 训练函数
def train_epoch(device, run, runs, epoch, total_epochs, detector, teacher_detector, optim_det, scheduler_det,
                trainloader, testloader, radius, unlabeled_dl=None, trainer=None,
                ema_decay=0.999, pc_weight=1, ac_weight=0.1, num_agents=64, K_max_agents=32,
                global_step=0, ema_warmup_steps=500, sup_only_epochs=50, vis_save_dir=None):

    scale = trainloader.dataset.scale  # 缩放因子，缓解截断误差

    if epoch == 50 or epoch == 200 or epoch == 500 or epoch == total_epochs - 1:
        actual_save_dir = vis_save_dir if vis_save_dir is not None else './vis_debug'

        vis_num_pred = 9
        vis_num_ac = 3

        vis_labeled_batch = None
        collected_parts = []
        collected_count = 0
        temp_iter = iter(trainloader)
        while collected_count < vis_num_pred:
            try:
                batch = next(temp_iter)
            except StopIteration:
                break
            collected_parts.append(batch)
            collected_count += batch[0].shape[0]  # batch[0] 是 imgs

        if collected_parts:
            vis_labeled_batch = []
            for field_idx in range(len(collected_parts[0])):
                cat_field = torch.cat([b[field_idx] for b in collected_parts], dim=0)[:vis_num_pred]
                vis_labeled_batch.append(cat_field)
            vis_labeled_batch = tuple(vis_labeled_batch)

        vis_unlabeled_batch = None
        if unlabeled_dl is not None:
            vis_num_collect = max(vis_num_pred, vis_num_ac)  # = 8
            collected_parts = []
            collected_count = 0
            temp_iter = iter(unlabeled_dl)
            while collected_count < vis_num_collect:
                try:
                    batch = next(temp_iter)
                except StopIteration:
                    break
                collected_parts.append(batch)
                collected_count += batch[1].shape[0]  # batch[1] 是 img_weak

            if collected_parts:
                vis_unlabeled_batch = []
                for field_idx in range(len(collected_parts[0])):
                    cat_field = torch.cat([b[field_idx] for b in collected_parts], dim=0)[:vis_num_collect]
                    vis_unlabeled_batch.append(cat_field)
                vis_unlabeled_batch = tuple(vis_unlabeled_batch)

        visualize_predictions_train(device, epoch, detector, teacher_detector,
                                    vis_labeled_batch=vis_labeled_batch,
                                    vis_unlabeled_batch=vis_unlabeled_batch,
                                    scale=trainloader.dataset.scale, save_dir=actual_save_dir,
                                    num_samples=vis_num_pred, samples_per_page=3)

        if vis_unlabeled_batch is not None and ac_weight > 0:
            ac_save_dir = vis_save_dir.replace('vis_debug', 'vis_affinity_consistency') \
                if vis_save_dir is not None else './vis_affinity_consistency'
            visualize_affinity_consistency(device, epoch, detector, teacher_detector,
                                           vis_unlabeled_batch=vis_unlabeled_batch,
                                           num_agents=num_agents, K_max_agents=K_max_agents, temperature=0.2,
                                           save_dir=ac_save_dir, num_samples=vis_num_ac, num_agents_to_show=4,
                                           scale=scale)

    cur_pc_weight = rampup(epoch, total_epochs, max_weight=pc_weight, sup_only_epochs=sup_only_epochs)
    cur_ac_weight = rampup(epoch, total_epochs, max_weight=ac_weight, sup_only_epochs=sup_only_epochs)

    unlabeled_iter = None
    if unlabeled_dl is not None:
        unlabeled_iter = iter(unlabeled_dl)

    """Taining process"""
    loss_sup_train = 0
    loss_pc_train = 0
    loss_ac_train = 0

    AE_epoch_train = []
    pred_p_epoch_train, gt_p_epoch_train = [], []

    detector.train()
    teacher_detector.eval()
    for imgs, DotM, DenM, IDistM in tqdm(trainloader):
        imgs = imgs.to(device)
        DotM = DotM.to(device)
        DenM = DenM.to(device)
        IDistM = IDistM.to(device)

        optim_det.zero_grad()

        IDistM_pred, _ = detector(imgs)
        IDistM_pred = IDistM_pred.to(device)

        assert IDistM_pred.shape == IDistM.shape, \
            f"Shape mismatch! IDistM_pred: {IDistM_pred.shape}, IDistM: {IDistM.shape}"

        B, _, H, W = IDistM_pred.shape
        loss_sup = ((IDistM_pred - IDistM) ** 2).sum() / (B * np.power(scale, 2))   # per image loss

        loss_pc = torch.tensor(0.0, device=device)
        loss_ac = torch.tensor(0.0, device=device)

        if unlabeled_iter is not None:
            try:
                unlabeled_imgs = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_dl)
                unlabeled_imgs = next(unlabeled_iter)

            assert isinstance(unlabeled_imgs, (list, tuple)) and len(unlabeled_imgs) >= 3
            img_weak = unlabeled_imgs[1].to(device)
            img_strong = unlabeled_imgs[2].to(device)

            with torch.no_grad():
                teacher_pred, teacher_feat = teacher_detector(img_weak)
                teacher_pred = teacher_pred.detach()
                teacher_feat = teacher_feat.detach()

            student_pred, student_feat = detector(img_strong)
            b, _, h, w = student_feat.shape

            assert student_pred.shape == teacher_pred.shape, \
                f"Shape mismatch! student_pred: {student_pred.shape}, teacher_pred: {teacher_pred.shape}"

            B_u, _, H_u, W_u = teacher_pred.shape
            if cur_pc_weight > 0:
                loss_pc = ((student_pred - teacher_pred) ** 2).sum() / (B_u * np.power(scale, 2))

            if cur_ac_weight > 0:
                agents = extract_agents(teacher_pred, teacher_feat,
                                        N=num_agents, K_max=K_max_agents, tau_fg=0.25)  # [B, C, N], already detached

                loss_ac = compute_affinity_consistency(teacher_feat, student_feat, agents)

        loss_total = loss_sup + cur_pc_weight * loss_pc + cur_ac_weight * loss_ac

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            print(f"⚠️ [Epoch {epoch}] NaN/Inf detected! "
                  f"loss_sup={loss_sup.item():.6f}, "
                  f"loss_pc={loss_pc.item():.6f}, "
                  f"loss_ac={loss_ac.item():.6f}, "  # ⭐️
                  f"cur_pc_weight={cur_pc_weight:.6f}, "
                  f"cur_ac_weight={cur_ac_weight:.6f}  → skipping this step")  # ⭐️
            optim_det.zero_grad()
            if epoch >= sup_only_epochs:
                global_step += 1
            continue

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=10.0)
        optim_det.step()

        if epoch >= sup_only_epochs and trainer is not None:
            cur_ema_decay = ema_decay_schedule(global_step, ema_warmup_steps, base_decay=ema_decay)
            trainer._update_teacher(teacher_detector, detector, cur_ema_decay)
            global_step += 1

        with torch.no_grad():
            loss_sup_train += (loss_sup/(H*W)).item()
            if unlabeled_iter is not None:
                loss_pc_train += (loss_pc / (H_u * W_u)).item()
                loss_ac_train += (loss_ac / (h * w)).item()

            DotM_pred = LMDS(IDistM_pred)

            pred_p_train, gt_p_train = [], []
            for i in range(len(imgs)):
                pred_p_train.append(torch.nonzero(DotM_pred[i].squeeze(), as_tuple=False).cpu().numpy().reshape(-1, 2))
                gt_p_train.append(torch.nonzero(DotM[i].squeeze(), as_tuple=False).cpu().numpy().reshape(-1, 2))

            pred_p_epoch_train.extend(pred_p_train)
            gt_p_epoch_train.extend(gt_p_train)

            # calculate AE for each gt-pred pair
            AE_train = abs(DotM.view(DotM.shape[0], -1).sum(dim=1).detach().cpu().numpy()
                               - DotM_pred.view(DotM_pred.shape[0], -1).sum(dim=1).detach().cpu().numpy())
            AE_epoch_train.extend(AE_train)

    # localization metrics
    ap_epoch_train, ar_epoch_train, fm_epoch_train, _, _ = local_metrics(pred_p_epoch_train, gt_p_epoch_train, sigma=radius)

    num_train_steps = len(trainloader)
    loss_sup_epoch_train = loss_sup_train / num_train_steps
    loss_pc_epoch_train = loss_pc_train / num_train_steps   #
    loss_ac_epoch_train = loss_ac_train / num_train_steps
    MAE_epoch_train = np.mean(AE_epoch_train)
    STD_epoch_train = np.std(AE_epoch_train)

    # =====================================================================
    # Validation process
    # =====================================================================

    loss_sup_test = 0
    AE_epoch_test = []
    pred_p_epoch_test, gt_p_epoch_test = [], []

    detector.eval()
    teacher_detector.eval()
    for imgs, DotM, DenM, IDistM in testloader:
        imgs = imgs.to(device)
        DotM = DotM.to(device)
        DenM = DenM.to(device)
        IDistM = IDistM.to(device)

        with torch.no_grad():
            # ========== 预测 ==========
            IDistM_pred, _ = detector(imgs)  # test by student
            IDistM_pred = IDistM_pred.to(device)

            B_t, _, H_t, W_t = IDistM_pred.shape
            loss_sup_t = ((IDistM_pred - IDistM) ** 2).sum() / (B_t * np.power(scale, 2))

            loss_sup_test += (loss_sup_t/(H_t*W_t)).item()

            DotM_pred = LMDS(IDistM_pred)

            pred_p_test, gt_p_test = [], []
            for i in range(len(imgs)):
                pred_p_test.append(torch.nonzero(DotM_pred[i].squeeze(), as_tuple=False).cpu().numpy().reshape(-1, 2))
                gt_p_test.append(torch.nonzero(DotM[i].squeeze(), as_tuple=False).cpu().numpy().reshape(-1, 2))
            pred_p_epoch_test.extend(pred_p_test)
            gt_p_epoch_test.extend(gt_p_test)

            AE_test = abs(DotM.view(DotM.shape[0], -1).sum(dim=1).detach().cpu().numpy()
                                - DotM_pred.view(DotM_pred.shape[0], -1).sum(dim=1).detach().cpu().numpy())
            AE_epoch_test.extend(AE_test)

    # ---------- Teacher test metrics ----------
    ap_epoch_test, ar_epoch_test, fm_epoch_test, _, _ = local_metrics(pred_p_epoch_test, gt_p_epoch_test, sigma=radius)

    num_test_steps = len(testloader)
    loss_sup_epoch_test = loss_sup_test / num_test_steps
    MAE_epoch_test = np.mean(AE_epoch_test)
    STD_epoch_test = np.std(AE_epoch_test)

    print('run: [{}/{}]'.format(run + 1, runs), 'epoch: {}'.format(epoch), 'cur_lr_det:',
          optim_det.param_groups[0]['lr'],)

    print('-----Counting Performance (Train)-----')
    print('loss_sup_train:', round(loss_sup_epoch_train, 8))
    print('loss_pc_train:', round(loss_pc_epoch_train, 8))
    print('loss_ac_train:', round(loss_ac_epoch_train, 8))
    print('MAE_train:', round(MAE_epoch_train, 3))
    print('STD_train:', round(STD_epoch_train, 3))

    print('-----Counting Performance (Test)-----')
    print('loss_sup_test:', round(loss_sup_epoch_test, 8))
    print('MAE_test:', round(MAE_epoch_test, 3))
    print('STD_test:', round(STD_epoch_test, 3))

    print('-----Localization Performance (Train)-----')
    print('precision_train:', round(ap_epoch_train, 5))
    print('recall_train:', round(ar_epoch_train, 5))
    print('F-measure_train:', round(fm_epoch_train, 5))

    print('-----Localization Performance (Test)-----')
    print('precision_test:', round(ap_epoch_test, 5))
    print('recall_test:', round(ar_epoch_test, 5))
    print('F-measure_test:', round(fm_epoch_test, 5))

    scheduler_det.step()

    return loss_sup_epoch_train, loss_pc_epoch_train, loss_ac_epoch_train, MAE_epoch_train, STD_epoch_train, \
           loss_sup_epoch_test, MAE_epoch_test, STD_epoch_test, \
           ap_epoch_train, ar_epoch_train, fm_epoch_train, \
           ap_epoch_test, ar_epoch_test, fm_epoch_test, \
           global_step


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
