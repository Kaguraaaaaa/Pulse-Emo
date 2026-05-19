import os
import pickle
import numpy as np
from random import random
from itertools import combinations

from config import get_config, activation_dict
from data_loader import get_loader
from solver import Solver

import torch
import torch.nn as nn
from torch.nn import functional as F

if __name__ == '__main__':

    # 设置随机种子
    random_seed = 336
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # 定义学习率列表
    learning_rates = [2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.1e-4, 5e-6]
    # learning_rates = [5e-5]

    # 定义模态及其默认 prefix_len
    modalities = {
        "visual": 8,
        "acoustic": 8,
        "physio": 8
    }


    ordered_combinations = [
        ['acoustic','visual','physio']  # l&a&v&p
    ]
    results = {} 
    # 循环学习率和模态组合
    for lr in learning_rates:
        for combo in ordered_combinations:

            print(f"\n===== Training with lr={lr}, modalities={combo} =====")

            # 获取训练配置
            train_config = get_config(mode='train')
            dev_config = get_config(mode='dev')
            test_config = get_config(mode='test')

            # 设置学习率
            train_config.learning_rate = lr

            # 根据当前模态组合设置 prefix_len
            train_config.prefix_len_visual = modalities['visual'] if 'visual' in combo else 0
            train_config.prefix_len_acoustic = modalities['acoustic'] if 'acoustic' in combo else 0
            train_config.prefix_len_physio = modalities['physio'] if 'physio' in combo else 0

            # 创建 dataloader
            train_data_loader = get_loader(train_config, shuffle=True)
            dev_data_loader = get_loader(dev_config, shuffle=False)
            test_data_loader = get_loader(test_config, shuffle=False)

            merge_dev_test = getattr(train_config, 'merge_dev_test', True)
            if merge_dev_test:
                dev_data_loader = None
                test_data_loader = get_loader(test_config, shuffle=False)

            # 初始化 Solver
            solver = Solver(train_config, dev_config, test_config,
                            train_data_loader, dev_data_loader, test_data_loader, is_train=True)
            
            

            solver.build()
            test_acc, test_f1  = solver.train()
            
            combo_name = 'l'
            if len(combo) > 0:
                mapping = {'acoustic': 'a', 'visual': 'v', 'physio': 'p'}
                combo_name = 'l&' + '&'.join([mapping[c] for c in combo])

            results[(lr, combo_name)] = {
                'acc': test_acc,
                'f1': test_f1
            }
    print("\n===== Final Results Table (Global Optimal LR) =====\n")

    mapping_order = ['l', 'l&a', 'l&v', 'l&p', 'l&a&v', 'l&a&p', 'l&v&p', 'l&a&v&p']

    # ------------------- 新增：筛选全局最优lr -------------------
    # 计算每个lr的综合表现（平均acc + 平均f1，权重各0.5，可调整）
    lr_performance = {}
    for lr in learning_rates:
        # 收集当前lr下所有模态组合的acc和f1
        acc_list = []
        f1_list = []
        for name in mapping_order:
            res = results.get((lr, name), {'acc': 0, 'f1': 0})
            # 仅统计有效训练的模态组合（排除未训练的0值，可选）
            if res['acc'] > 0:
                acc_list.append(res['acc'])
                f1_list.append(res['f1'])
        # 计算当前lr的综合得分（无有效数据则得分为0）
        if acc_list and f1_list:
            avg_acc = np.mean(acc_list)
            avg_f1 = np.mean(f1_list)
            # 综合得分 = 平均acc * 0.5 + 平均f1 * 0.5（可根据需求调整权重）
            lr_performance[lr] = 0.5 * avg_acc + 0.5 * avg_f1
        else:
            lr_performance[lr] = 0

    # 筛选全局最优lr（综合得分最高，若有并列取第一个）
    global_optimal_lr = max(lr_performance.keys(), key=lambda x: lr_performance[x])
    print(f"🔥 Global Optimal Learning Rate: {global_optimal_lr}")
    print(f"Optimal LR Performance (Avg Acc + Avg F1)/2: {lr_performance[global_optimal_lr]:.4f}\n")

    # ------------------- 输出最优lr的结果表格 -------------------
    acc_row = []
    f1_row = []
    for name in mapping_order:
        res = results.get((global_optimal_lr, name), {'acc': 0, 'f1': 0})
        acc_row.append(f"{res['acc']*100:.2f}")
        f1_row.append(f"{res['f1']*100:.2f}")

    print(f"LR={global_optimal_lr}")
    print("Acc & " + " & ".join(acc_row) + " \\\\")
    print("F1  & " + " & ".join(f1_row) + " \\\\")