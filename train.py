# -*- coding:utf-8 -*-
"""
Author：LIU Rui
Date：2022/11/08
"""
import os
import argparse
import torch
# from utils.trainer import Trainer
from utils.trainer_semi import Trainer
import pandas as pd
import numpy as np
import random


args = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument('--root', default='.',
                        help="root dir")

    # The radius is the GT region. The sigma here refers to the gaussian filter for counting, which is different from the sigma is the local_metrics.
    parser.add_argument("--exp_purpose", type=str, default="Exp_159",
                        help="experiment purpose") 
    parser.add_argument("--dataset", type=str, default='PanNuke',
                        help="dataset name", choices=['PanNuke', 'BCD', 'UniCD'])
    parser.add_argument("-det_net", type=str, default='CAFFNet',
                        help="detection network", choices=['CAFFNet', 'CAFFNet_4down'])
    parser.add_argument('--conf', type=str, default='S',
                        help='the localization groud truth region', choices=['L', 'S'])
    parser.add_argument("--runs", type=int, default=5,
                        help="the number of run for each setting")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="the number of training epochs")
    parser.add_argument('--warm_up_epoch', type=int, default=10,
                        help='the warm-up epochs')
    parser.add_argument('--const_epoch', type=int, default=5,
                        help='the constant lr epochs in constant_cosine lr scheduler')
    parser.add_argument("--optim", type=str, default='Adam', nargs="*",
                        help="optimizer")
    parser.add_argument("--lr", type=float, default=[0.0005], nargs="*",
                        help="the base_learning rate")
    parser.add_argument("--lr_gen", type=float, default=1e-6,    
                        help="the base_learning rate of kernel generator")
    parser.add_argument("--lr_min", type=float, default=1e-10,
                        help="the minimum learning rate")
    parser.add_argument('--weight_decay', type=float, default=[0.0001], nargs="*",
                        help='the weight decay')

    # if set to be '0,1', means the cuda 0 and cuda 1 will be visible (used)
    parser.add_argument('--device', default='0',
                        help='assign device')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='the number of training process')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set the visible gpu(s)

    trainer = Trainer(args)

    # search for the best hyperparameter combination
    trainer.dosearch()

    # the final expriment
    # trainer_count.dotrain()
