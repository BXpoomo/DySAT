# -*- encoding: utf-8 -*-
'''
@File    :   train_classification.py
@Time    :   2023/07/31
@Author  :   [Your Name]
@Contact :   [Your Email]
'''

import torch
import torch.optim as optim
from models.model import DySAT
from utils.minibatch import  MyDataset # 请确保你有一个加载数据的模块或函数
from torch.nn import CrossEntropyLoss

import argparse
import numpy as np
import pickle as pkl
import scipy
from torch.utils.data import DataLoader
from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from models.model import DySAT

def main(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # 加载数据集
    dataset = MyDataset()  # 你需要提供一个数据加载函数
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    
    # 初始化模型、优化器、损失函数
    model = DySAT(args, num_features=dataset.num_features, time_length=dataset.time_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    criterion = CrossEntropyLoss().to(device)  # 节点分类任务的损失函数

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            graphs, labels = batch
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(graphs)
            
            # 注意：根据你的数据和DySAT模型的输出，你可能需要调整下面的损失计算
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # 你可以添加验证集和测试集评估的代码，根据需要进行评估和早停等操作

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--l2_coef', type=float, default=5e-4, help='L2 coefficient.')
    # 可以添加更多的参数，比如：模型的参数、数据集路径等
    
    args = parser.parse_args()
    main(args)
