# -*- coding:utf-8 -*-
"""
Author：R
Date：30-04-2026
"""
import numpy as np
import torch
import torch.nn.functional as F
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def rampup(epoch, total_epochs, max_weight=0.2, rampup_ratio=0.5, sup_only_epochs=50):
    if epoch < sup_only_epochs:
        return 0.0
    effective_epoch = epoch - sup_only_epochs
    effective_total = total_epochs - sup_only_epochs
    rampup_length = int(effective_total * rampup_ratio)
    if rampup_length == 0:
        return max_weight
    if effective_epoch >= rampup_length:
        return max_weight
    ratio = effective_epoch / rampup_length
    return max_weight * np.exp(-5.0 * (1.0 - ratio) ** 2)


def ema_decay_schedule(global_step, warmup_steps, base_decay=0.999):
    if global_step >= warmup_steps:
        return base_decay
    return 0.9 + (base_decay - 0.9) * global_step / warmup_steps


def _orthogonal_select(features, K, seed_feats=None):
    C, num = features.shape
    if num == 0:
        return []
    if K <= 0:
        return []
    if num <= K and seed_feats is None:
        return list(range(num))

    feats_norm = F.normalize(features, dim=0)  # [C, num]

    if seed_feats is not None and seed_feats.shape[1] > 0:
        seed_norm = F.normalize(seed_feats, dim=0)  # [C, num_seed]
        sim_with_seed = torch.mm(feats_norm.T, seed_norm)  # [num, num_seed]
        max_sim, _ = sim_with_seed.max(dim=1)  # [num]
    else:
        max_sim = torch.zeros(num, device=features.device)

    selected = []
    used = torch.zeros(num, dtype=torch.bool, device=features.device)

    for _ in range(min(K, num)):
        scores = max_sim.clone()
        scores[used] = float('inf')

        next_idx = torch.argmin(scores).item()
        selected.append(next_idx)
        used[next_idx] = True

        new_sim = torch.mv(feats_norm.T, feats_norm[:, next_idx])  # [num]
        max_sim = torch.max(max_sim, new_sim)

    return selected


def extract_agents(teacher_pred, teacher_feat, N=32, K_max=16, tau_fg=0.2,
                   return_positions=False):
    B, C, h, w = teacher_feat.shape
    _, _, H, W = teacher_pred.shape

    agents_batch = []
    positions_batch = []

    for b in range(B):
        pred_b = teacher_pred[b, 0]  # [H, W]
        feat_b = teacher_feat[b]     # [C, h, w]

        pred_max = pred_b.max().item()
        positions_b = []  # 当前样本的位置列表

        if pred_max < 1e-5:
            feat_flat = feat_b.view(C, -1)  # [C, h*w]
            selected_all = _orthogonal_select(feat_flat, K=N, seed_feats=None)
            agents_b = feat_flat[:, selected_all]
            agents_batch.append(agents_b)
            if return_positions:
                for si in selected_all:
                    ri = si // w
                    ci = si % w
                    positions_b.append((ri, ci, 'orth'))
                positions_batch.append(positions_b)
            continue

        pred_unsq = pred_b.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        local_max = F.max_pool2d(pred_unsq, kernel_size=3, stride=1, padding=1)
        is_peak = (pred_unsq == local_max).squeeze(0).squeeze(0)  # [H, W]

        fg_mask = (pred_b >= tau_fg * pred_max) & is_peak
        peak_coords = torch.nonzero(fg_mask, as_tuple=False)  # [num_peaks, 2] (row, col)

        if peak_coords.shape[0] > 0:
            peak_values = pred_b[peak_coords[:, 0], peak_coords[:, 1]]

            sorted_indices = torch.argsort(peak_values, descending=True)
            top_n = min(2 * K_max, peak_coords.shape[0])
            candidate_coords = peak_coords[sorted_indices[:top_n]]  # [top_n, 2]

            feat_coords_r = (candidate_coords[:, 0].float() / H * h).long().clamp(0, h - 1)
            feat_coords_c = (candidate_coords[:, 1].float() / W * w).long().clamp(0, w - 1)

            candidate_feats = feat_b[:, feat_coords_r, feat_coords_c]  # [C, top_n]

            K_actual = min(K_max, candidate_feats.shape[1])
            selected_indices = _orthogonal_select(candidate_feats, K_actual)
            cell_agents = candidate_feats[:, selected_indices]  # [C, K_actual]

            for si in selected_indices:
                positions_b.append((feat_coords_r[si].item(), feat_coords_c[si].item(), 'fg'))
        else:
            K_actual = 0
            cell_agents = torch.zeros(C, 0, device=teacher_feat.device)

        M = N - K_actual

        nonfg_mask = pred_b < tau_fg * pred_max
        nonfg_mask_ds = F.interpolate(
            nonfg_mask.float().unsqueeze(0).unsqueeze(0),
            size=(h, w), mode='nearest'
        ).squeeze(0).squeeze(0).bool()

        nonfg_coords = torch.nonzero(nonfg_mask_ds, as_tuple=False)  # [num_nonfg, 2]

        if nonfg_coords.shape[0] > 0 and M > 0:
            nonfg_feats = feat_b[:, nonfg_coords[:, 0], nonfg_coords[:, 1]]  # [C, num_nonfg]
            M_actual = min(M, nonfg_feats.shape[1])
            selected_nonfg = _orthogonal_select(nonfg_feats, M_actual, seed_feats=cell_agents)
            nonfg_agents = nonfg_feats[:, selected_nonfg]  # [C, M_actual]

            for si in selected_nonfg:
                positions_b.append((nonfg_coords[si, 0].item(), nonfg_coords[si, 1].item(), 'nonfg'))
        else:
            nonfg_agents = torch.zeros(C, 0, device=teacher_feat.device)
            M_actual = 0

        agents_b = torch.cat([cell_agents, nonfg_agents], dim=1)  # [C, K_actual + M_actual]

        if agents_b.shape[1] < N:
            deficit = N - agents_b.shape[1]
            feat_flat = feat_b.view(C, -1)  # [C, h*w]

            used_positions = set()
            for (r, c, _) in positions_b:
                used_positions.add(r * w + c)

            mask_available = torch.ones(h * w, dtype=torch.bool, device=teacher_feat.device)
            for pos in used_positions:
                mask_available[pos] = False
            available_indices = torch.arange(h * w, device=teacher_feat.device)[mask_available]

            available_feats = feat_flat[:, available_indices]  # [C, num_available]

            pad_selected = _orthogonal_select(
                available_feats,
                K=deficit,
                seed_feats=agents_b
            )

            pad_agents = available_feats[:, pad_selected]  # [C, len(pad_selected)]
            agents_b = torch.cat([agents_b, pad_agents], dim=1)

            for si in pad_selected:
                real_idx = available_indices[si].item()
                ri = real_idx // w
                ci = real_idx % w
                positions_b.append((ri, ci, 'orth'))

        agents_batch.append(agents_b[:, :N])
        if return_positions:
            positions_batch.append(positions_b[:N])

    agents = torch.stack(agents_batch, dim=0)  # [B, C, N]

    if return_positions:
        return agents.detach(), positions_batch
    return agents.detach()  # stop gradient


def compute_affinity_consistency(teacher_feat, student_feat, agents, temperature=0.4):
    B, C, h, w = teacher_feat.shape
    hw = h * w

    # Flatten spatial: [B, C, h, w] -> [B, hw, C]
    teacher_flat = teacher_feat.view(B, C, hw).permute(0, 2, 1)  # [B, hw, C]
    student_flat = student_feat.view(B, C, hw).permute(0, 2, 1)  # [B, hw, C]

    teacher_flat = F.normalize(teacher_flat, dim=-1)  # [B, hw, C] 在特征维归一化
    student_flat = F.normalize(student_flat, dim=-1)  # [B, hw, C]
    agents_norm = F.normalize(agents, dim=1)  # [B, C, N] 在C维归一化

    aff_teacher = torch.bmm(teacher_flat, agents_norm) / temperature  # [B, hw, N]
    aff_student = torch.bmm(student_flat, agents_norm) / temperature  # [B, hw, N]

    log_prob_teacher = F.log_softmax(aff_teacher, dim=-1)   # [B, hw, N]
    log_prob_student = F.log_softmax(aff_student, dim=-1)   # [B, hw, N]

    loss_ac = F.kl_div(log_prob_student, log_prob_teacher, reduction='batchmean', log_target=True)

    return loss_ac

