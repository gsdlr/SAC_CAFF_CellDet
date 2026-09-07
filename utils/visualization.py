# -*- coding:utf-8 -*-
"""
Author：R
Date：29-04-2026
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.consistency import extract_agents
import torch.nn as nn
import math
import os
import cv2
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure
from local_eval.eval import local_metrics
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def visualize_predictions_train(device, epoch, detector, teacher_detector,
                          vis_labeled_batch=None, vis_unlabeled_batch=None,
                          scale=1.0, save_dir='./vis_debug',
                          num_samples=9, samples_per_page=3):
    os.makedirs(save_dir, exist_ok=True)
    detector.eval()
    teacher_detector.eval()

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    def denorm(tensor_img):
        img = tensor_img.cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        return np.clip(img, 0, 1)

    def to_heatmap(tensor_map):
        return tensor_map.squeeze().cpu().numpy()


    if vis_labeled_batch is None:
        print('[Vis] No labeled batch provided, skipping labeled visualization.')
    else:
        with torch.no_grad():
            imgs = vis_labeled_batch[0][:num_samples].to(device)
            DotM = vis_labeled_batch[1][:num_samples].to(device)
            IDistM = vis_labeled_batch[3][:num_samples].to(device)

            IDistM_pred, _ = detector(imgs)

            IDistM = IDistM / scale
            IDistM_pred = IDistM_pred / scale

            total_n = imgs.shape[0]
            num_pages = math.ceil(total_n / samples_per_page)

            for page in range(num_pages):
                start_idx = page * samples_per_page
                end_idx = min(start_idx + samples_per_page, total_n)
                n = end_idx - start_idx

                fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n))
                if n == 1:
                    axes = axes[np.newaxis, :]

                for row, i in enumerate(range(start_idx, end_idx)):
                    axes[row, 0].imshow(denorm(imgs[i]))
                    axes[row, 0].set_title('Input Image', fontsize=12)

                    dot_map_np = DotM[i].squeeze().cpu().numpy()
                    gt_count = dot_map_np.sum()
                    dot_vis = np.ones_like(dot_map_np)
                    dot_vis[dot_map_np > 0] = 0
                    axes[row, 1].imshow(dot_vis, cmap='gray', vmin=0, vmax=1)
                    axes[row, 1].set_title(f'Dot Map (count={int(gt_count)})', fontsize=12)
                    axes[row, 1].set_xticks([])
                    axes[row, 1].set_yticks([])
                    for spine in axes[row, 1].spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor('#CCCCCC')
                        spine.set_linewidth(0.5)

                    gt_map = to_heatmap(IDistM[i])
                    axes[row, 2].imshow(gt_map, cmap='jet')
                    axes[row, 2].set_title(f'GT FADT (sum={gt_map.sum():.2f})', fontsize=12)

                    pred_map = to_heatmap(IDistM_pred[i])
                    axes[row, 3].imshow(pred_map, cmap='jet')
                    axes[row, 3].set_title(f'Student Pred (sum={pred_map.sum():.2f})', fontsize=12)

                    diff_labeled = np.abs(pred_map - gt_map)
                    axes[row, 4].imshow(diff_labeled, cmap='hot')
                    axes[row, 4].set_title(f'|Pred - GT| (MSE={np.mean(diff_labeled**2):.6f})', fontsize=12)

                    for j, ax in enumerate(axes[row]):
                        if j == 1:
                            continue
                        ax.axis('off')

                fig.suptitle(f'Epoch {epoch} — Labeled Data [Page {page+1}/{num_pages}]',
                             fontsize=16, fontweight='bold')
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'epoch{epoch}_labeled_page{page}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'[Vis] Labeled visualization saved to {save_path}')

            print(f'[Vis Stats - Labeled] IDistM_pred  min={IDistM_pred.min():.6f}  max={IDistM_pred.max():.6f}  mean={IDistM_pred.mean():.6f}')
            print(f'[Vis Stats - Labeled] IDistM (GT)  min={IDistM.min():.6f}  max={IDistM.max():.6f}  mean={IDistM.mean():.6f}')


    if vis_unlabeled_batch is None:
        print('[Vis] No unlabeled batch provided, skipping unlabeled visualization.')
        detector.train()
        return

    with torch.no_grad():
        img_weak = vis_unlabeled_batch[1][:num_samples].to(device)
        img_strong = vis_unlabeled_batch[2][:num_samples].to(device)

        teacher_pred, _ = teacher_detector(img_weak)
        student_pred, _ = detector(img_strong)

        teacher_pred = teacher_pred / scale
        student_pred = student_pred / scale

        total_n = img_weak.shape[0]
        num_pages = math.ceil(total_n / samples_per_page)

        for page in range(num_pages):
            start_idx = page * samples_per_page
            end_idx = min(start_idx + samples_per_page, total_n)
            n = end_idx - start_idx

            fig, axes = plt.subplots(n, 6, figsize=(30, 5 * n))
            if n == 1:
                axes = axes[np.newaxis, :]

            for row, i in enumerate(range(start_idx, end_idx)):
                axes[row, 0].imshow(denorm(img_weak[i]))
                axes[row, 0].set_title('Weak Aug Image', fontsize=12)

                axes[row, 1].imshow(denorm(img_strong[i]))
                axes[row, 1].set_title('Strong Aug Image', fontsize=12)

                t_map = to_heatmap(teacher_pred[i])
                axes[row, 2].imshow(t_map, cmap='jet')
                axes[row, 2].set_title(f'Teacher Pred (sum={t_map.sum():.2f})', fontsize=12)

                s_map = to_heatmap(student_pred[i])
                axes[row, 3].imshow(s_map, cmap='jet')
                axes[row, 3].set_title(f'Student Pred (sum={s_map.sum():.2f})', fontsize=12)

                diff = np.abs(s_map - t_map)
                mse_val = np.mean(diff ** 2)
                axes[row, 4].imshow(diff, cmap='hot')
                axes[row, 4].set_title(f'|Student - Teacher| (MSE={mse_val:.6f})', fontsize=12)

                axes[row, 5].hist(t_map.flatten(), bins=100, alpha=0.5, label='Teacher', color='blue')
                axes[row, 5].hist(s_map.flatten(), bins=100, alpha=0.5, label='Student', color='red')
                axes[row, 5].legend(fontsize=10)
                axes[row, 5].set_title('Pixel Value Distribution', fontsize=12)
                axes[row, 5].set_yscale('log')

                for ax in axes[row, :5]:
                    ax.axis('off')

            fig.suptitle(f'Epoch {epoch} — Unlabeled Data [Page {page+1}/{num_pages}]',
                         fontsize=16, fontweight='bold')
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'epoch{epoch}_unlabeled_page{page}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[Vis] Unlabeled visualization saved to {save_path}')

        print(f'[Vis Stats - Unlabeled] teacher_pred  min={teacher_pred.min():.6f}  max={teacher_pred.max():.6f}  mean={teacher_pred.mean():.6f}')
        print(f'[Vis Stats - Unlabeled] student_pred  min={student_pred.min():.6f}  max={student_pred.max():.6f}  mean={student_pred.mean():.6f}')

    detector.train()

def visualize_affinity_consistency(device, epoch, detector, teacher_detector,
                                   vis_unlabeled_batch=None, num_agents=32, K_max_agents=16,
                                   temperature=0.2, save_dir='./vis_affinity_consistency',
                                   num_samples=2, num_agents_to_show=4,
                                   num_points_to_show=5, scale=1.0):

    if vis_unlabeled_batch is None:
        print('[Vis-AC] No unlabeled batch provided, skipping affinity visualization.')
        return

    os.makedirs(save_dir, exist_ok=True)
    detector.eval()
    teacher_detector.eval()

    # 反归一化参数
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    def denorm(tensor_img):
        img = tensor_img.cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img

    def select_representative_points(pred_small, h, w, num_points=5):
        flat = pred_small.flatten()
        sorted_idx = np.argsort(flat)[::-1]

        total_pixels = h * w
        percentiles = [0.01, 0.1, 0.3, 0.6, 0.95]
        labels = ['FG-Core', 'FG-High', 'FG-Mid', 'Transition', 'Background']

        points = []
        for i in range(min(num_points, len(percentiles))):
            idx_in_sorted = int(percentiles[i] * total_pixels)
            idx_in_sorted = min(idx_in_sorted, total_pixels - 1)
            flat_idx = sorted_idx[idx_in_sorted]
            pi = flat_idx // w
            pj = flat_idx % w
            points.append((pi, pj, labels[i]))

        return points

    def kl_div_numpy(p, q, eps=1e-8):
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        return np.sum(p * np.log(p / q))

    def hide_axis_keep_ylabel(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    with torch.no_grad():
        assert isinstance(vis_unlabeled_batch, (list, tuple)) and len(vis_unlabeled_batch) >= 3
        img_raws = vis_unlabeled_batch[0]  # [B, 3, H, W], [0,1]
        img_weak = vis_unlabeled_batch[1].to(device)  # [B, 3, H, W], normalized
        img_strong = vis_unlabeled_batch[2].to(device)  # [B, 3, H, W], normalized

        teacher_pred, teacher_feat = teacher_detector(img_weak)
        student_pred, student_feat = detector(img_strong)

        B, C, h, w = teacher_feat.shape

        agents, agent_positions_batch = extract_agents(
            teacher_pred, teacher_feat,
            N=num_agents, K_max=K_max_agents,
            tau_fg=0.25,
            return_positions=True)

        sample_count = 0

        for b in range(B):
            if sample_count >= num_samples:
                break

            t_feat = teacher_feat[b]  # [C, h, w]
            s_feat = student_feat[b]  # [C, h, w]
            ag = agents[b]            # [C, N]
            N_total = ag.shape[1]
            agent_positions = agent_positions_batch[b]  # list of (r, c, type_str)

            # Flatten & L2 normalize
            t_flat = F.normalize(t_feat.view(C, -1).permute(1, 0), dim=-1)  # [hw, C]
            s_flat = F.normalize(s_feat.view(C, -1).permute(1, 0), dim=-1)  # [hw, C]
            ag_norm = F.normalize(ag, dim=0)  # [C, N]

            logits_t = torch.mm(t_flat, ag_norm) / temperature  # [hw, N]
            logits_s = torch.mm(s_flat, ag_norm) / temperature  # [hw, N]

            prob_t = F.softmax(logits_t, dim=-1)  # [hw, N]
            prob_s = F.softmax(logits_s, dim=-1)  # [hw, N]

            prob_t_spatial = prob_t.view(h, w, N_total)
            prob_s_spatial = prob_s.view(h, w, N_total)
            diff_spatial = prob_t_spatial - prob_s_spatial

            log_prob_t = F.log_softmax(logits_t, dim=-1)
            log_prob_s = F.log_softmax(logits_s, dim=-1)
            kl_per_pixel = (prob_t * (log_prob_t - log_prob_s)).sum(dim=-1)  # [hw]
            kl_map = kl_per_pixel.view(h, w)

            N_show = min(num_agents_to_show, N_total)

            fig, axes = plt.subplots(N_show, 3, figsize=(15, 4 * N_show))
            if N_show == 1:
                axes = axes[np.newaxis, :]

            for n in range(N_show):
                t_map = prob_t_spatial[:, :, n].cpu().numpy()
                s_map = prob_s_spatial[:, :, n].cpu().numpy()
                d_map = diff_spatial[:, :, n].cpu().numpy()

                vmin_shared = min(t_map.min(), s_map.min())
                vmax_shared = max(t_map.max(), s_map.max())

                im_t = axes[n, 0].imshow(t_map, cmap='jet', vmin=vmin_shared, vmax=vmax_shared)
                if n < len(agent_positions):
                    ar, ac, atype = agent_positions[n]
                    row_label = f'Agent {n}\n[{atype}]\n({ar},{ac})'
                else:
                    row_label = f'Agent {n}'
                axes[n, 0].set_ylabel(row_label, fontsize=11, fontweight='bold',
                                       rotation=0, labelpad=60, va='center')
                if n == 0:
                    axes[n, 0].set_title('Teacher P(agent|pixel)', fontsize=12)
                hide_axis_keep_ylabel(axes[n, 0])
                plt.colorbar(im_t, ax=axes[n, 0], fraction=0.046, pad=0.04)

                im_s = axes[n, 1].imshow(s_map, cmap='jet', vmin=vmin_shared, vmax=vmax_shared)
                if n == 0:
                    axes[n, 1].set_title('Student P(agent|pixel)', fontsize=12)
                axes[n, 1].axis('off')
                plt.colorbar(im_s, ax=axes[n, 1], fraction=0.046, pad=0.04)

                abs_max = max(abs(d_map.min()), abs(d_map.max())) + 1e-8
                im_d = axes[n, 2].imshow(d_map, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
                if n == 0:
                    axes[n, 2].set_title('Diff (T − S)', fontsize=12)
                axes[n, 2].axis('off')
                plt.colorbar(im_d, ax=axes[n, 2], fraction=0.046, pad=0.04)


            fig.suptitle(
                f'Epoch {epoch} — Per-Agent Spatial Affinity Map (Sample {sample_count})\n'
                f'Each row corresponds to one Agent: affinity P(agent|pixel) at every spatial location in bottleneck feature map\n'
                f'Left: Teacher affinity | Mid: Student affinity | Right: Diff (T-S), highlighting regions where Student needs to align\n'
                f'(tau={temperature}, N_agents={N_total}, feature_size={h}x{w})',
                fontsize=12, fontweight='bold', y=0.99
            )

            plt.tight_layout(rect=[0.06, 0, 1, 0.96])
            save_path = os.path.join(save_dir, f'epoch{epoch}_per_agent_sample{sample_count}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[Vis-AC] Per-agent affinity saved to {save_path}')

            pred_map = teacher_pred[b]  # [1, H, W]
            pred_small = F.interpolate(
                pred_map.unsqueeze(0), size=(h, w),
                mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()  # [h, w]
            pred_small = pred_small / scale

            points = select_representative_points(pred_small, h, w, num_points=num_points_to_show)

            fig2, axes2 = plt.subplots(num_points_to_show, 3, figsize=(20, 3.5 * num_points_to_show))
            if num_points_to_show == 1:
                axes2 = axes2[np.newaxis, :]

            x_ticks = np.arange(N_total)

            for idx, (pi, pj, label) in enumerate(points):
                prob_t_point = prob_t_spatial[pi, pj, :].cpu().numpy()  # [N]
                prob_s_point = prob_s_spatial[pi, pj, :].cpu().numpy()  # [N]
                kl_val = kl_div_numpy(prob_t_point, prob_s_point)

                y_max = min(1.0, max(prob_t_point.max(), prob_s_point.max()) * 1.3 + 0.05)

                row_label = f'P{idx}: {label}\n({pi},{pj})\npred={pred_small[pi, pj]:.3f}'
                axes2[idx, 0].set_ylabel(row_label, fontsize=10, fontweight='bold',
                                          rotation=0, labelpad=80, va='center')

                axes2[idx, 0].bar(x_ticks, prob_t_point, color='#2196F3', alpha=0.8)
                if idx == num_points_to_show - 1:
                    axes2[idx, 0].set_xlabel('Agent Index')
                else:
                    axes2[idx, 0].set_xlabel('')
                axes2[idx, 0].set_title(
                    f'Teacher P(agent|pixel)' if idx == 0 else '',
                    fontsize=10
                )
                axes2[idx, 0].set_ylim(0, y_max)
                if N_total <= 16:
                    axes2[idx, 0].set_xticks(x_ticks)
                else:
                    axes2[idx, 0].set_xticks(x_ticks[::4])

                axes2[idx, 1].bar(x_ticks, prob_s_point, color='#FF9800', alpha=0.8)
                if idx == num_points_to_show - 1:
                    axes2[idx, 1].set_xlabel('Agent Index')
                else:
                    axes2[idx, 1].set_xlabel('')
                axes2[idx, 1].set_title(
                    f'Student P(agent|pixel)' if idx == 0 else '',
                    fontsize=10
                )
                axes2[idx, 1].set_ylim(0, y_max)
                if N_total <= 16:
                    axes2[idx, 1].set_xticks(x_ticks)
                else:
                    axes2[idx, 1].set_xticks(x_ticks[::4])

                diff_point = prob_t_point - prob_s_point
                colors = ['#F44336' if d > 0 else '#4CAF50' for d in diff_point]
                axes2[idx, 2].bar(x_ticks, diff_point, color=colors, alpha=0.8)
                axes2[idx, 2].axhline(0, color='black', linewidth=0.5)
                if idx == num_points_to_show - 1:
                    axes2[idx, 2].set_xlabel('Agent Index')
                else:
                    axes2[idx, 2].set_xlabel('')
                axes2[idx, 2].set_title(
                    f'Diff (T−S) & KL Divergence' if idx == 0 else '',
                    fontsize=10
                )
                axes2[idx, 2].text(
                    0.95, 0.90, f'KL={kl_val:.4f}',
                    transform=axes2[idx, 2].transAxes,
                    fontsize=10, fontweight='bold', color='darkred',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8)
                )
                abs_max_bar = max(abs(diff_point.min()), abs(diff_point.max())) + 0.05
                axes2[idx, 2].set_ylim(-abs_max_bar, abs_max_bar)
                if N_total <= 16:
                    axes2[idx, 2].set_xticks(x_ticks)
                else:
                    axes2[idx, 2].set_xticks(x_ticks[::4])


            fig2.suptitle(
                f'Epoch {epoch} — Point-to-Agents Affinity Distribution (Sample {sample_count})\n'
                f'Each row shows one representative spatial point: similarity distribution between its bottleneck feature and all {N_total} agents\n'
                f'Left: Teacher dist | Mid: Student dist | Right: Diff (T-S) + KL divergence, measuring Teacher-Student distribution gap\n'
                f'(tau={temperature}, N_agents={N_total}, points selected in descending order of Teacher prediction value)',
                fontsize=11, fontweight='bold', y=0.99
            )

            plt.tight_layout(rect=[0.08, 0, 1, 0.95])
            save_path2 = os.path.join(save_dir, f'epoch{epoch}_point_affinity_sample{sample_count}.png')
            plt.savefig(save_path2, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f'[Vis-AC] Point-agent affinity saved to {save_path2}')

            import matplotlib.gridspec as gridspec

            img_raw_np = img_raws[b].numpy().transpose(1, 2, 0)  # [H, W, 3], [0,1]
            img_h, img_w = img_raw_np.shape[0], img_raw_np.shape[1]

            t_pred_np = (teacher_pred[b] / scale).squeeze().cpu().numpy()
            s_pred_np = (student_pred[b] / scale).squeeze().cpu().numpy()

            t_feat_l2 = torch.norm(t_feat, dim=0).cpu().numpy()
            s_feat_l2 = torch.norm(s_feat, dim=0).cpu().numpy()
            t_feat_vis = (t_feat_l2 - t_feat_l2.min()) / (t_feat_l2.max() - t_feat_l2.min() + 1e-8)
            s_feat_vis = (s_feat_l2 - s_feat_l2.min()) / (s_feat_l2.max() - s_feat_l2.min() + 1e-8)

            img_weak_np = denorm(img_weak[b])
            img_strong_np = denorm(img_strong[b])

            fig3 = plt.figure(figsize=(18, 16))
            gs = gridspec.GridSpec(3, 3, figure=fig3,
                                   width_ratios=[1.0, 1.0, 1.0],
                                   height_ratios=[1.0, 1.0, 1.0],
                                   hspace=0.30, wspace=0.30)

            ax_r0c0 = fig3.add_subplot(gs[0, 0])
            ax_r0c0.imshow(img_raw_np)
            ax_r0c0.set_title(
                f'Original Image\n({img_w}x{img_h})',
                fontsize=9, fontweight='bold', pad=8)
            ax_r0c0.axis('off')

            ax_r0c1 = fig3.add_subplot(gs[0, 1])
            ax_r0c1.axis('off')
            stats_text = (
                f'--- Statistics ---\n'
                f'Image size: {img_w}x{img_h}\n'
                f'Feature size: {h}x{w}\n'
                f'N_agents: {N_total}\n'
                f'Temperature: {temperature}\n'
                f'Scale: {scale}\n\n'
                f'Teacher pred sum: {t_pred_np.sum():.2f}\n'
                f'Teacher pred max: {t_pred_np.max():.4f}\n'
                f'Student pred sum: {s_pred_np.sum():.2f}\n'
                f'Student pred max: {s_pred_np.max():.4f}\n\n'
                f'Teacher feat L2 mean(raw): {t_feat_l2.mean():.4f}\n'
                f'Student feat L2 mean(raw): {s_feat_l2.mean():.4f}'
            )
            ax_r0c1.text(0.05, 0.95, stats_text, transform=ax_r0c1.transAxes,
                         fontsize=9, fontfamily='monospace',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round',
                                   facecolor='lightyellow', alpha=0.8))

            ax_r0c2 = fig3.add_subplot(gs[0, 2])
            ax_r0c2.axis('off')

            ax_r1c0 = fig3.add_subplot(gs[1, 0])
            ax_r1c0.imshow(img_weak_np)
            ax_r1c0.set_title('Weak Augmentation\n(Teacher Input)',
                              fontsize=9, fontweight='bold')
            ax_r1c0.axis('off')

            ax_r1c1 = fig3.add_subplot(gs[1, 1])
            im_tp = ax_r1c1.imshow(t_pred_np, cmap='jet')
            plt.colorbar(im_tp, ax=ax_r1c1, fraction=0.046, pad=0.04)
            ax_r1c1.set_title(f'Teacher Prediction\n(sum={t_pred_np.sum():.2f})',
                              fontsize=9, fontweight='bold')
            ax_r1c1.axis('off')

            ax_r1c2 = fig3.add_subplot(gs[1, 2])
            im_tf = ax_r1c2.imshow(t_feat_vis, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im_tf, ax=ax_r1c2, fraction=0.046, pad=0.04)
            ax_r1c2.set_title(f'Teacher Bottleneck Feature\n(L2 norm, min-max normed, {h}x{w})',
                              fontsize=9, fontweight='bold')
            ax_r1c2.axis('off')

            ax_r2c0 = fig3.add_subplot(gs[2, 0])
            ax_r2c0.imshow(img_strong_np)
            ax_r2c0.set_title('Strong Augmentation\n(Student Input)',
                              fontsize=9, fontweight='bold')
            ax_r2c0.axis('off')

            ax_r2c1 = fig3.add_subplot(gs[2, 1])
            im_sp = ax_r2c1.imshow(s_pred_np, cmap='jet')
            plt.colorbar(im_sp, ax=ax_r2c1, fraction=0.046, pad=0.04)
            ax_r2c1.set_title(f'Student Prediction\n(sum={s_pred_np.sum():.2f})',
                              fontsize=9, fontweight='bold')
            ax_r2c1.axis('off')

            ax_r2c2 = fig3.add_subplot(gs[2, 2])
            im_sf = ax_r2c2.imshow(s_feat_vis, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im_sf, ax=ax_r2c2, fraction=0.046, pad=0.04)
            ax_r2c2.set_title(f'Student Bottleneck Feature\n(L2 norm, min-max normed, {h}x{w})',
                              fontsize=9, fontweight='bold')
            ax_r2c2.axis('off')

            fig3.suptitle(
                f'Epoch {epoch} — Affinity Consistency Overview (Sample {sample_count})\n'
                f'Row 0: Original Image / Stats | '
                f'Row 1: Teacher pipeline | Row 2: Student pipeline',
                fontsize=12, fontweight='bold', y=0.98)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path3 = os.path.join(save_dir, f'epoch{epoch}_overview_sample{sample_count}.png')
            plt.savefig(save_path3, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print(f'[Vis-AC] Overview saved to {save_path3}')

            print(f'[Vis-AC Stats] prob_t  min={prob_t.min():.6f}  max={prob_t.max():.6f}  mean={prob_t.mean():.6f}')
            print(f'[Vis-AC Stats] prob_s  min={prob_s.min():.6f}  max={prob_s.max():.6f}  mean={prob_s.mean():.6f}')
            kl_np = kl_map.cpu().numpy()
            print(f'[Vis-AC Stats] KL      min={kl_np.min():.6f}  max={kl_np.max():.6f}  mean={kl_np.mean():.6f}')

            sample_count += 1

    detector.train()

def visualization_test(dataset_name, imgs, gts, preds, DotM, DotM_pred, device, num=5):

    imgs, gts, preds = imgs.to(device), gts.to(device), preds.to(device)
    DotM, DotM_pred = DotM.to(device), DotM_pred.to(device)

    plt.figure(figsize=(12, 3 * num))

    for i in range(num):
        row = num
        col = 4

        plt.subplot(row, col, i * col + 1)
        img_np = imgs[i].permute(1, 2, 0).cpu().numpy()
        img_np = 0.5 * img_np + 0.5
        plt.title('input')

        plt.imshow(img_np)
        plt.axis('off')

        plt.subplot(row, col, i * col + 2)
        gt_np = gts[i].permute(1, 2, 0).cpu().numpy()
        gt_np = gt_np
        plt.title('ground truth: {}'.format(int(DotM[i].sum().item())))

        plt.imshow(gt_np)
        plt.axis('off')

        plt.subplot(row, col, i * col + 3)
        pred_np = preds[i].permute(1, 2, 0).detach().cpu().numpy()
        pred_np = pred_np
        plt.title('prediction: {}'.format(int(DotM_pred[i].sum().item())))

        plt.imshow(pred_np)
        plt.axis('off')

        img_det = imgs[i].permute(1, 2, 0).detach().cpu().numpy()
        img_det = cv2.cvtColor(img_det, cv2.COLOR_RGB2BGR)
        coords = torch.nonzero(DotM[i].squeeze()).cpu().numpy()
        coords_pred = torch.nonzero(DotM_pred[i].squeeze()).cpu().numpy()

        if 'PanNuke' in dataset_name:
            radius_val = 12
            thickness = 2
            radius_pred = 3
        elif 'BCD' in dataset_name:
            radius_val = 10
            thickness = 2
            radius_pred = 5
        else:
            radius_val = 8
            thickness = 1
            radius_pred = 3

        tp_preds = []
        fp_preds = []
        fn_gts = []

        if len(coords) == 0:
            fp_preds = coords_pred.tolist()
        elif len(coords_pred) == 0:
            fn_gts = coords.tolist()
        else:
            dist_matrix = cdist(coords_pred, coords)
            match_matrix = dist_matrix <= radius_val
            _, assign = hungarian(match_matrix)

            coords_np = np.array(coords)
            coords_pred_np = np.array(coords_pred)

            tp_pred_index = np.where(assign.sum(1) == 1)[0]

            fn_gt_index = np.where(assign.sum(0) == 0)[0]

            fp_pred_index = np.where(assign.sum(1) == 0)[0]

            if len(coords_np) > 0:
                fn_gts.extend(coords_np[fn_gt_index].tolist())

            if len(coords_pred_np) > 0:
                fp_preds.extend(coords_pred_np[fp_pred_index].tolist())
                tp_preds.extend(coords_pred_np[tp_pred_index].tolist())

        color_tp = (0, 1.0, 0)
        color_fp = (0, 1.0, 1.0)
        color_fn = (0, 0, 1.0)

        for pt in tp_preds:
            cv2.circle(img_det, (int(pt[1]), int(pt[0])), radius_pred, color_tp, -1)
        for pt in fp_preds:
            cv2.circle(img_det, (int(pt[1]), int(pt[0])), radius_pred, color_fp, -1)
        for pt in fn_gts:
            cv2.circle(img_det, (int(pt[1]), int(pt[0])), radius_pred, color_fn, -1)

        img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        img_det = 0.5 * img_det + 0.5
        plt.subplot(row, col, i * col + 4)
        plt.title('detection result\n(TP:G, FP:Y, FN:B)')

        plt.imshow(img_det)
        plt.axis('off')

    plt.tight_layout(h_pad=2.0, w_pad=1.0)
    plt.show()
