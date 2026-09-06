# -*- coding:utf-8 -*-
"""
Author：R
Date：19-08-2022
"""
# from utils.fit import train_epoch
from utils.fit_semi import train_epoch
from utils.config import bs, scale, radius
from utils.mydataset import Dataset, UnlabeledDataset
import os
import numpy as np
import math
import torch
from torch.utils import data
import glob
from models.CAFFNet import CAFFNet, CAFFNet_4down
import copy
import argparse
import pandas as pd
import openpyxl


def glob_images(directory):
    """在 directory 下搜索常见图片格式，返回排序后的路径列表"""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


class Trainer(object):

    def __init__(self, args):

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.results_path = './logs/' + self.args.exp_purpose + '_' + self.args.dataset + '_' + self.args.det_net + '_radius_' + self.args.conf
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.weight_path = './checkpoints/' + self.args.exp_purpose + '_' + self.args.dataset + '_' + self.args.det_net + '_radius_' + self.args.conf
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)

    # ===================== 创建 teacher 模型 =====================
    @torch.no_grad()
    def _create_teacher(self, student):
        """深拷贝 student 作为 teacher，并冻结梯度"""
        teacher = copy.deepcopy(student)
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    # ===================== 新增：EMA 更新 teacher =====================
    @torch.no_grad()
    def _update_teacher(self, teacher, student, ema_decay):
        """用指数移动平均更新 teacher 的权重（包括 BN 的 running stats）"""
        for (name_t, param_t), (name_s, param_s) in zip(
                teacher.state_dict().items(), student.state_dict().items()):
            param_t.copy_(ema_decay * param_t + (1.0 - ema_decay) * param_s)

    # 用于搜索最佳超参组合
    def dosearch(self):

        file_path_summary = self.results_path + '/' + self.args.dataset + '_' + self.args.det_net + '_metrics_summary.xlsx'
        df_summary = pd.DataFrame(columns=['learning rate', 'weight decay', 'MAE', 'MSE', 'f-measure', 'precision', 'recall'])  # 定义列表头
        df_summary.to_excel(file_path_summary)

        setting = 0
        for self.lr in self.args.lr:
            for self.wd in self.args.weight_decay:

                avg_metric = self.dotrain()
                df_summary.loc['setting_{}'.format(setting + 1)] = [self.lr, self.wd, avg_metric[0], avg_metric[1],
                                                                    avg_metric[2], avg_metric[3], avg_metric[4]]  # 增加一行
                setting += 1

        writer = pd.ExcelWriter(file_path_summary, engine='openpyxl')
        df_summary.to_excel(writer, sheet_name='summary', index=True)  # 将要加进来的数据写入writer
        writer.close()

    def dotrain(self):

        file_path_train_record = self.results_path + "/" + self.args.dataset + '_' + self.args.det_net \
                          + '_bs_{}'.format(bs[self.args.dataset]) + '_optim_{}'.format(self.args.optim) \
                          + '_lr_{}'.format(self.lr) + '_wd_{}'.format(self.wd) + '_train_record.xlsx'
        file_path_metrics = self.results_path + '/' + self.args.dataset + '_' + self.args.det_net \
                            + '_bs_{}'.format(bs[self.args.dataset]) + '_optim_{}'.format(self.args.optim) \
                            + '_lr_{}'.format(self.lr) + '_wd_{}'.format(self.wd) + '_metrics.xlsx'
        file_random_seed = self.results_path + "/" + self.args.dataset + '_' + self.args.det_net  \
                           + '_bs_{}'.format(bs[self.args.dataset]) + '_optim_{}'.format(self.args.optim) \
                           + '_lr_{}'.format(self.lr) + '_wd_{}'.format(self.wd) + '_random_seed.xlsx'

        # 创建一个空的excel表,用于保存训练过程数据
        df_record = pd.DataFrame()  # 只是一个圆括号的话说明创建的数据表是空的
        df_record.to_excel(file_path_train_record)

        df_metric = pd.DataFrame(columns=[
            'MAE_test', 'STD_test',
            'precision_test', 'recall_test', 'f-measure_test',
        ])

        df_metric.to_excel(file_path_metrics)

        df_seed = pd.DataFrame(columns=['random_seed'])  # 定义列表头
        df_seed.to_excel(file_random_seed)

        for run in range(self.args.runs):

            # ========================加载数据====================================
            random_seed = run + 1  # 数据已预先划分，random_seed仅用于excel记录

            base_path = os.path.join(self.args.root, 'datasets', self.args.dataset)

            # labeled train: train/labeled/images & train/labeled/dot_maps
            images_train = glob_images(os.path.join(base_path, 'train', 'labeled', 'images'))
            dot_maps_train = sorted(glob.glob(os.path.join(base_path, 'train', 'labeled', 'dot_maps', '*.npy')))

            # test: test/images & test/dot_maps
            images_test = glob_images(os.path.join(base_path, 'test', 'images'))
            dot_maps_test = sorted(glob.glob(os.path.join(base_path, 'test', 'dot_maps', '*.npy')))

            # 数量一致性校验
            assert len(images_train) == len(dot_maps_train), \
                f"train/labeled 图片({len(images_train)})与标注({len(dot_maps_train)})数量不一致"
            assert len(images_test) == len(dot_maps_test), \
                f"test 图片({len(images_test)})与标注({len(dot_maps_test)})数量不一致"
            assert len(images_train) > 0, f"train/labeled 为空: {base_path}"
            assert len(images_test) > 0, f"test 为空: {base_path}"

            print(f"labeled train: {len(images_train)}, test: {len(images_test)}")
            # ============================================================

            train_ds = Dataset(images_train, dot_maps_train, self.args.dataset, mode='train', scale=scale[self.args.dataset])
            test_ds = Dataset(images_test, dot_maps_test, self.args.dataset, mode='test', scale=scale[self.args.dataset])

            train_dl = data.DataLoader(train_ds, batch_size=bs[self.args.dataset], shuffle=True,
                                       num_workers=self.args.num_workers, pin_memory=False)
            test_dl = data.DataLoader(test_ds, batch_size=bs[self.args.dataset],
                                      num_workers=self.args.num_workers, pin_memory=False)

            # ============================================================
            # 加载无标签数据: train/unlabeled/images
            # ============================================================
            unlabeled_img_dir = os.path.join(base_path, 'train', 'unlabeled', 'images')
            unlabeled_images = glob_images(unlabeled_img_dir)

            if len(unlabeled_images) > 0:
                print(f"找到 {len(unlabeled_images)} 张无标签图像")
                unlabeled_ds = UnlabeledDataset(unlabeled_images, self.args.dataset)
                unlabeled_dl = data.DataLoader(unlabeled_ds, batch_size=bs[self.args.dataset], shuffle=True,
                                               drop_last=True, num_workers=self.args.num_workers,
                                               pin_memory=False)
            else:
                print("警告: 未找到无标签图像，将仅使用有标签数据进行训练")
                unlabeled_dl = None
            # ============================================================
            # ⭐️ 为当前 run 创建可视化目录（放在 logs 对应实验文件夹内）
            vis_dir = os.path.join(self.results_path, 'vis_debug', f'run_{run}')
            os.makedirs(vis_dir, exist_ok=True)

            detector = globals()[self.args.det_net]().to(self.device)
            # ===================== 在 student 创建后立刻创建 teacher =====================
            teacher_detector = self._create_teacher(detector)

            optim_det = getattr(torch.optim, self.args.optim)(detector.parameters(), lr=self.lr, weight_decay=self.wd)

            def lambda_rule(cur_epoch):
                if cur_epoch < self.args.warm_up_epoch:
                    return 0.95 * (cur_epoch + 1) / self.args.warm_up_epoch
                else:
                    return (self.args.lr_min + 0.5 * (self.lr - self.args.lr_min) * (1.0 + math.cos(
                        (cur_epoch - self.args.warm_up_epoch) /
                        (self.args.epochs - self.args.warm_up_epoch) * math.pi))) / self.lr

            scheduler_det = torch.optim.lr_scheduler.LambdaLR(optim_det, lr_lambda=lambda_rule)

            epoch_list = []
            loss_sup_train, loss_pc_train, loss_ac_train_list, MAE_train, STD_train = [], [], [], [], []
            ap_train, ar_train, fm_train = [], [], []

            # test metrics lists
            loss_sup_test, MAE_test, STD_test = [], [], []
            ap_test, ar_test, fm_test = [], [], []

            best_fm = 0
            # 每次 run 重置 global_step（新的模型从头训练）
            global_step = 0
            sup_only_epochs = 50  # 定义纯监督阶段长度

            print("**********run_{} of dataset_{} bs_{} optim_{} lr_{} weight_decay_{}**********".format(
                run + 1,
                self.args.dataset,
                bs[self.args.dataset],
                self.args.optim,
                self.lr,
                self.wd))

            for epoch in range(self.args.epochs):
                # 在半监督阶段开始时，用当前 student 重建 teacher 并重置 global_step
                if epoch == sup_only_epochs and unlabeled_dl is not None:
                    teacher_detector = self._create_teacher(detector)
                    global_step = 0
                    print(f"\n[Epoch {epoch}] ★ Teacher re-initialized from student. "
                          f"Semi-supervised training begins. global_step reset to 0.\n")
                # 前 sup_only_epochs 个 epoch 传 unlabeled_dl=None（纯监督），之后传实际的 unlabeled_dl
                use_unlabeled = unlabeled_dl if epoch >= sup_only_epochs else None

                # ===== 接收 train_epoch 返回的 15 个值=====
                loss_sup_epoch_train, loss_pc_epoch_train, loss_ac_epoch_train, MAE_epoch_train, STD_epoch_train, \
                    loss_sup_epoch_test, MAE_epoch_test, STD_epoch_test, \
                    ap_epoch_train, ar_epoch_train, fm_epoch_train, \
                    ap_epoch_test, ar_epoch_test, fm_epoch_test, \
                    global_step,\
                    = train_epoch(self.device, run, self.args.runs, epoch, self.args.epochs, detector, teacher_detector,
                                  optim_det, scheduler_det, train_dl, test_dl,
                                  radius[self.args.conf][self.args.dataset], unlabeled_dl=None, trainer=self,
                                  global_step=global_step, sup_only_epochs=sup_only_epochs, vis_save_dir=vis_dir)

                epoch_list.append(epoch)

                # Train metrics
                loss_sup_train.append(loss_sup_epoch_train)
                loss_pc_train.append(loss_pc_epoch_train)
                loss_ac_train_list.append(loss_ac_epoch_train)
                MAE_train.append(MAE_epoch_train)
                STD_train.append(STD_epoch_train)
                ap_train.append(ap_epoch_train)
                ar_train.append(ar_epoch_train)
                fm_train.append(fm_epoch_train)

                # test metrics
                loss_sup_test.append(loss_sup_epoch_test)
                MAE_test.append(MAE_epoch_test)
                STD_test.append(STD_epoch_test)
                ap_test.append(ap_epoch_test)
                ar_test.append(ar_epoch_test)
                fm_test.append(fm_epoch_test)

                # ===== 保存权重：以 best_fm 为准同时保存 student 和 teacher =====
                detector_save = self.weight_path + '/' + self.args.dataset + '_' + self.args.det_net \
                                + '_bs_{}'.format(bs[self.args.dataset]) + '_optim_{}'.format(self.args.optim) \
                                + '_lr_{}'.format(self.lr) + '_wd_{}'.format(self.wd) + '_run_{}'.format(run + 1) \
                                + '.pth'

                teacher_save_best = self.weight_path + '/' + self.args.dataset + '_teacher_' + self.args.det_net \
                                    + '_bs_{}'.format(bs[self.args.dataset]) + '_optim_{}'.format(self.args.optim) \
                                    + '_lr_{}'.format(self.lr) + '_wd_{}'.format(self.wd) + '_run_{}'.format(run + 1) \
                                    + '_best.pth'

                # 以 F-measure 为主指标，模型保存
                if fm_epoch_test > best_fm:
                    best_fm = fm_epoch_test
                    best_detector_wt = copy.deepcopy(detector.state_dict())
                    torch.save(best_detector_wt, detector_save)
                    best_teacher_wt_best = copy.deepcopy(teacher_detector.state_dict())
                    torch.save(best_teacher_wt_best, teacher_save_best)

                # 保存特定 epoch 的权重
                if epoch == 0 or epoch == 9 or epoch == 99 or epoch == 499 or epoch == 999:
                    detector_save_ep = self.weight_path + '/' + self.args.dataset + '_' + self.args.det_net \
                                       + '_bs_{}'.format(bs[self.args.dataset]) + '_optim_{}'.format(self.args.optim) \
                                       + '_lr_{}'.format(self.lr) + '_wd_{}'.format(self.wd) + '_run_{}'.format(run + 1) \
                                       + '_epoch_{}'.format(epoch) + '.pth'

                    teacher_save_ep = self.weight_path + '/' + self.args.dataset + '_teacher_' + self.args.det_net \
                                      + '_bs_{}'.format(bs[self.args.dataset]) + '_optim_{}'.format(self.args.optim) \
                                      + '_lr_{}'.format(self.lr) + '_wd_{}'.format(self.wd) + '_run_{}'.format(run + 1) \
                                      + '_epoch_{}'.format(epoch) + '.pth'

                    detector_wt_ep = copy.deepcopy(detector.state_dict())
                    torch.save(detector_wt_ep, detector_save_ep)

                    teacher_wt_ep = copy.deepcopy(teacher_detector.state_dict())
                    torch.save(teacher_wt_ep, teacher_save_ep)

                # ===== 导出每一次实验的训练过程数据（包含 teacher 和 student 的 test 指标）=====
            output_excel = {
                'epoch': epoch_list,
                # -----Counting Performance (Train)-----
                'loss_sup_train': loss_sup_train, 'loss_pc_train': loss_pc_train,
                'loss_ac_train': loss_ac_train_list,  # ⭐️
                'MAE_train': MAE_train, 'STD_train': STD_train,

                # -----Counting Performance (Test)-----
                'loss_sup_test': loss_sup_test, 'MAE_test': MAE_test, 'STD_test': STD_test,

                #  localization(Train)
                'precision_train': ap_train, 'recall_train': ar_train, 'F-measure_train': fm_train,

                # localization( Test )
                'precision_test': ap_test, 'recall_test': ar_test, 'F-measure_test': fm_test,
               }

            output = pd.DataFrame(output_excel)
            writer = pd.ExcelWriter(file_path_train_record, mode='a', engine='openpyxl')
            output.to_excel(writer, sheet_name='run_{}'.format(run + 1), index=False)
            writer.close()

            # ===== 导出每一次训练的 performance（以 teacher 的 best fm 对应 epoch 为准）=====
            # Teacher best
            fm_t = max(fm_test)
            index_fm_t = fm_test.index(fm_t)
            ap_t = ap_test[index_fm_t]
            ar_t = ar_test[index_fm_t]
            mae_t = MAE_test[index_fm_t]
            std_t = STD_test[index_fm_t]

            df_metric.loc['run_{}'.format(run + 1)] = [
                mae_t, std_t, ap_t, ar_t, fm_t,
            ]

            # 获取到所有数据后，计算 validation 中 metrics 的平均值
            if run == self.args.runs - 1:
                avg_metric = []
                for col in df_metric.columns:
                    avg_metric.append(df_metric[col].mean())
                df_metric.loc['average value'] = tuple(avg_metric)

            writer_metric = pd.ExcelWriter(file_path_metrics, engine='openpyxl')
            df_metric.to_excel(writer_metric, sheet_name='metrics', index=True)
            writer_metric.close()

            df_seed.loc['run_{}'.format(run + 1)] = random_seed

            writer_seed = pd.ExcelWriter(file_random_seed, engine='openpyxl')
            df_seed.to_excel(writer_seed, sheet_name='metrics', index=True)
            writer_seed.close()

        return avg_metric