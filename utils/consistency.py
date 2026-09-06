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

# ===== ramp-up 函数，用于逐渐增大无监督损失权重 =====
def rampup(epoch, total_epochs, max_weight=0.2, rampup_ratio=0.5, sup_only_epochs=50):
    """
    在 sup_only_epochs 之前返回 0（纯监督阶段）。
    之后按 sigmoid ramp-up 逐步升到 max_weight。
    """
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


# EMA decay warm-up 调度函数，按 step 调度
def ema_decay_schedule(global_step, warmup_steps, base_decay=0.999):
    if global_step >= warmup_steps:
        return base_decay
    return 0.9 + (base_decay - 0.9) * global_step / warmup_steps


# ⭐️⭐️⭐️ ==================== affinity Consistency 相关函数 ==================== ⭐️⭐️⭐️
def _orthogonal_select(features, K, seed_feats=None):
    """
    贪心正交选择：从候选特征中选出 K 个最多样化的向量。
    若提供 seed_feats，则新选出的向量还会与 seed 保持最大差异。

    Args:
        features: [C, num_candidates] - 候选特征向量（列向量）
        K: 要选出的数量
        seed_feats: [C, num_seed] or None - 已有的 agent 特征，作为"已占据的方向"参照。
                    当提供 seed 时，算法会确保新选出的向量不仅彼此不同，
                    还与 seed 中的每一个都尽量不同，避免与已有 agents 冗余。
                    典型用途：fg agents 已选好后，补充非前景 agent 时将它们作为 seed 传入。
                    为 None 时退化为普通正交选择（从 index 0 开始贪心）。

    Returns:
        selected_indices: 长度为 min(K, num_candidates) 的索引列表
    """
    C, num = features.shape
    if num == 0:
        return []
    if K <= 0:
        return []
    if num <= K and seed_feats is None:
        return list(range(num))

    # L2 归一化用于计算余弦相似度
    feats_norm = F.normalize(features, dim=0)  # [C, num]

    # 初始化 max_sim：若有 seed，先计算候选与 seed 的相似度作为起点
    if seed_feats is not None and seed_feats.shape[1] > 0:
        seed_norm = F.normalize(seed_feats, dim=0)  # [C, num_seed]
        sim_with_seed = torch.mm(feats_norm.T, seed_norm)  # [num, num_seed]
        max_sim, _ = sim_with_seed.max(dim=1)  # [num]
    else:
        # 无 seed：所有候选的初始"已知最大相似度"为 0
        max_sim = torch.zeros(num, device=features.device)

    selected = []
    used = torch.zeros(num, dtype=torch.bool, device=features.device)

    for _ in range(min(K, num)):
        # 已选的设为 inf，使其不会被再次选中
        scores = max_sim.clone()
        scores[used] = float('inf')

        # 选择与"已有集合"（seed + 已选）最不相似的那个
        next_idx = torch.argmin(scores).item()
        selected.append(next_idx)
        used[next_idx] = True

        # 更新 max_sim：新选入的向量也加入"已占据方向"集合
        new_sim = torch.mv(feats_norm.T, feats_norm[:, next_idx])  # [num]
        max_sim = torch.max(max_sim, new_sim)

    return selected


def extract_agents(teacher_pred, teacher_feat, N=32, K_max=16, tau_fg=0.2,
                   return_positions=False):
    """
    从 Teacher 的预测和特征中提取 cell-aware agents。

    逻辑：
      1. 在 teacher 的逆距离变换预测图上做 local maximum 检测，找到细胞候选位置
         按峰值排序取 top-2K，再正交选择 K_actual 个作为前景 agents
      2. 在非前景区域（<tau_fg * pred_max）正交选择剩余 N-K_actual 个 agents，
         以前景 agents 作为 seed_feats 保证全局多样性
      3. 极端情况下若仍不足 N 个，从全图可用位置带 seed 正交补齐

    Args:
        teacher_pred: [B, 1, H, W] - Teacher 预测的逆距离变换图
        teacher_feat: [B, C, h, w] - Teacher encoder 末端特征（1/16 分辨率）
        N: 总 agent 数量（默认 32）
        K_max: 前景 agent 上限（默认 16）
        tau_fg: 前景阈值比例（相对于样本最大值）
        return_positions: 是否同时返回 agent 在特征图坐标系下的位置

    Returns:
        agents: [B, C, N] - 每个样本的 agent 特征向量（已 detach）
        positions (optional): list of list of (row, col, type_str)
                              type_str ∈ {'fg', 'nonfg', 'orth'}
    """
    B, C, h, w = teacher_feat.shape
    _, _, H, W = teacher_pred.shape

    agents_batch = []
    positions_batch = []  # 每个样本的 agent 位置列表

    for b in range(B):
        pred_b = teacher_pred[b, 0]  # [H, W]
        feat_b = teacher_feat[b]     # [C, h, w]

        pred_max = pred_b.max().item()
        positions_b = []  # 当前样本的位置列表

        # 如果预测图几乎全为 0（空白图），用正交选择填充（无 seed）
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

        # ========== 前景 agents：基于 local maximum 检测 ==========
        pred_unsq = pred_b.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        local_max = F.max_pool2d(pred_unsq, kernel_size=3, stride=1, padding=1)
        is_peak = (pred_unsq == local_max).squeeze(0).squeeze(0)  # [H, W]

        # 保留峰值 > tau_fg * max 的局部极大值点
        fg_mask = (pred_b >= tau_fg * pred_max) & is_peak
        peak_coords = torch.nonzero(fg_mask, as_tuple=False)  # [num_peaks, 2] (row, col)

        if peak_coords.shape[0] > 0:
            peak_values = pred_b[peak_coords[:, 0], peak_coords[:, 1]]

            # 按峰值降序排列，取 top-2K 候选
            sorted_indices = torch.argsort(peak_values, descending=True)
            top_n = min(2 * K_max, peak_coords.shape[0])
            candidate_coords = peak_coords[sorted_indices[:top_n]]  # [top_n, 2]

            # 映射到特征图坐标 (H,W) -> (h,w)
            feat_coords_r = (candidate_coords[:, 0].float() / H * h).long().clamp(0, h - 1)
            feat_coords_c = (candidate_coords[:, 1].float() / W * w).long().clamp(0, w - 1)

            # 提取候选特征
            candidate_feats = feat_b[:, feat_coords_r, feat_coords_c]  # [C, top_n]

            # 正交选择 K_actual 个（无 seed，前景内部多样化）
            K_actual = min(K_max, candidate_feats.shape[1])
            selected_indices = _orthogonal_select(candidate_feats, K_actual)
            cell_agents = candidate_feats[:, selected_indices]  # [C, K_actual]

            # 记录前景 agent 的特征图坐标
            for si in selected_indices:
                positions_b.append((feat_coords_r[si].item(), feat_coords_c[si].item(), 'fg'))
        else:
            K_actual = 0
            cell_agents = torch.zeros(C, 0, device=teacher_feat.device)

        # ========== 非前景 agents：< tau_fg * pred_max 区域（去掉 tau_bg） ==========
        M = N - K_actual

        # 非前景掩码：预测值 < tau_fg * pred_max 的区域（原来是 < tau_bg * pred_max）
        nonfg_mask = pred_b < tau_fg * pred_max
        # 下采样到特征图分辨率
        nonfg_mask_ds = F.interpolate(
            nonfg_mask.float().unsqueeze(0).unsqueeze(0),
            size=(h, w), mode='nearest'
        ).squeeze(0).squeeze(0).bool()

        nonfg_coords = torch.nonzero(nonfg_mask_ds, as_tuple=False)  # [num_nonfg, 2]

        # 从非前景区域正交选择，以前景 agents 为 seed
        if nonfg_coords.shape[0] > 0 and M > 0:
            nonfg_feats = feat_b[:, nonfg_coords[:, 0], nonfg_coords[:, 1]]  # [C, num_nonfg]
            M_actual = min(M, nonfg_feats.shape[1])
            selected_nonfg = _orthogonal_select(nonfg_feats, M_actual, seed_feats=cell_agents)
            nonfg_agents = nonfg_feats[:, selected_nonfg]  # [C, M_actual]

            # 记录非前景 agent 的特征图坐标（标记为 'nonfg'）
            for si in selected_nonfg:
                positions_b.append((nonfg_coords[si, 0].item(), nonfg_coords[si, 1].item(), 'nonfg'))
        else:
            nonfg_agents = torch.zeros(C, 0, device=teacher_feat.device)
            M_actual = 0

        # ========== 安全兜底：极端情况下仍不足 N 个时，从全图可用位置补齐 ==========
        agents_b = torch.cat([cell_agents, nonfg_agents], dim=1)  # [C, K_actual + M_actual]

        if agents_b.shape[1] < N:
            deficit = N - agents_b.shape[1]
            feat_flat = feat_b.view(C, -1)  # [C, h*w]

            # 构建已使用位置集合，排除已选的位置，防止重叠
            used_positions = set()
            for (r, c, _) in positions_b:
                used_positions.add(r * w + c)

            # 构建可用位置掩码
            mask_available = torch.ones(h * w, dtype=torch.bool, device=teacher_feat.device)
            for pos in used_positions:
                mask_available[pos] = False
            available_indices = torch.arange(h * w, device=teacher_feat.device)[mask_available]

            # 提取可用位置的特征作为候选
            available_feats = feat_flat[:, available_indices]  # [C, num_available]

            # 以已有 agents 为 seed，正交选择 deficit 个补充 agent
            pad_selected = _orthogonal_select(
                available_feats,
                K=deficit,
                seed_feats=agents_b  # 已有的 agents 作为参照
            )

            pad_agents = available_feats[:, pad_selected]  # [C, len(pad_selected)]
            agents_b = torch.cat([agents_b, pad_agents], dim=1)

            # 记录位置，标记为 'orth'
            for si in pad_selected:
                real_idx = available_indices[si].item()
                ri = real_idx // w
                ci = real_idx % w
                positions_b.append((ri, ci, 'orth'))

        agents_batch.append(agents_b[:, :N])  # 确保恰好 N 个
        if return_positions:
            positions_batch.append(positions_b[:N])  # 也截断到 N

    agents = torch.stack(agents_batch, dim=0)  # [B, C, N]

    if return_positions:
        return agents.detach(), positions_batch
    return agents.detach()  # stop gradient


def compute_affinity_consistency(teacher_feat, student_feat, agents, temperature=0.4):
    """
    计算 Teacher 和 Student 之间的 Affinity Consistency Loss。

    对每个像素位置，计算其特征向量与所有 agents 的相似度分布（softmax），
    然后用 KL 散度约束 Student 的分布与 Teacher 一致。

    Args:
        teacher_feat: [B, C, h, w] - Teacher encoder 末端特征
        student_feat: [B, C, h, w] - Student encoder 末端特征
        agents: [B, C, N] - Agent 特征向量（已 detach）
        temperature: float - 控制 softmax 分布的锐度，越小越尖锐

    Returns:
        loss_ac: 标量 tensor
    """

    """
    【关于 Affinity consistency Loss (loss_ac) 尺度的重要说明】

    1. 尺度差异分析：
       - 监督损失 (loss_sup) 和像素损失 (loss_pix) 是在输出端 (原分辨率) 计算的。
       - 亲和力损失 (loss_ac) 是在网络最深处的 Bottleneck 计算的。
       - Bottleneck 经过了 4 次下采样（1/16），因此其特征图的像素总量 (HW) 比原图小了 16 * 16 = 256 倍。

    2. 权重设置建议：
       由于 Bottleneck 的 hw 较小，除以 hw 后得到的 Per Pixel Loss 数值（相比于除以原图 HW）相对较大；
       且深层特征图的单个"像素"包含了丰富的全局语义（感受野极大）。
       因此，在最终的总 Loss 组合中，需要为 loss_ac 乘以一个较小的平衡系数（例如 0.1或0.05），
       以防止深层产生的巨大梯度淹没浅层的监督信号。
    """
    B, C, h, w = teacher_feat.shape
    hw = h * w

    # Flatten spatial: [B, C, h, w] -> [B, hw, C]
    teacher_flat = teacher_feat.view(B, C, hw).permute(0, 2, 1)  # [B, hw, C]
    student_flat = student_feat.view(B, C, hw).permute(0, 2, 1)  # [B, hw, C]

    # ====================控制softmax输入logits 的尺度方案一: / sqrt(C)	防止高维dot product数值过大======================
    # # Affinity: [B, hw, C] x [B, C, N] -> [B, hw, N]
    # scale = math.sqrt(C)
    # aff_teacher = torch.bmm(teacher_flat, agents) / scale  # [B, hw, N]
    # aff_student = torch.bmm(student_flat, agents) / scale  # [B, hw, N]
    # ====================方案一结束==================================================

    # =============控制softmax输入logits的尺度方案二:F.normalize + / temperature 值域锁定 [-1,1]，再用temperature精确控制放大倍数
    # L2 归一化：让 dot product 变成 cosine similarity
    teacher_flat = F.normalize(teacher_flat, dim=-1)  # [B, hw, C] 在特征维归一化
    student_flat = F.normalize(student_flat, dim=-1)  # [B, hw, C]
    agents_norm = F.normalize(agents, dim=1)  # [B, C, N] 在C维归一化

    # Affinity: cosine similarity / temperature
    # [B, hw, C] x [B, C, N] -> [B, hw, N]
    # 归一化后 dot product 值域为 [-1, 1]，除以 temperature 放大差异
    aff_teacher = torch.bmm(teacher_flat, agents_norm) / temperature  # [B, hw, N]
    aff_student = torch.bmm(student_flat, agents_norm) / temperature  # [B, hw, N]
    # ====================方案二结束======================

    # Softmax over agent dimension (得到关于"属于哪个 agent"的概率分布)
    log_prob_teacher = F.log_softmax(aff_teacher, dim=-1)   # [B, hw, N]
    log_prob_student = F.log_softmax(aff_student, dim=-1)   # [B, hw, N]

    # KL(teacher || student) = sum_over_agents[ p_teacher * (log_p_teacher - log_p_student) ]
    # 使用 PyTorch 的 kl_div：输入是 log_prob，target 是 prob

    # per image loss  虽然数值比较小,但是由于bottleneck的特征图尺寸比较小(比输入小了16×16倍),因此要乘以一个比较小的系数来平衡per element loss
    loss_ac = F.kl_div(log_prob_student, log_prob_teacher, reduction='batchmean', log_target=True)

    # per element loss (by R)
    # loss_ac_element = loss_ac / hw  # 除以 hw 得到 per element loss，再乘以一个小系数（如0.1）来平衡尺度

    return loss_ac

# ⭐️⭐️⭐️ ==================== affinity Consistency 相关函数结束 ==================== ⭐️⭐️⭐️
