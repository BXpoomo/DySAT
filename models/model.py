# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss

from models.layers import StructuralAttentionLayer, TemporalAttentionLayer
from utils.utilities import fixed_unigram_candidate_sampler

class DySAT(nn.Module):
    def __init__(self, args, num_features, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.args = args
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features

        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))  # [16, 8, 8] 结构多头信息
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))  # 结构layer层信息
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))  # [16] 时序多头信息
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))  # [128] 时序layer层信息
        self.spatial_drop = args.spatial_drop  # 定义dropout
        self.temporal_drop = args.temporal_drop

        self.structural_attn, self.temporal_attn = self.build_model()

        self.bceloss = BCEWithLogitsLoss()  # 定义loss函数; sigmoid和crossentropy节点在一起

    def forward(self, graphs):

        # Structural Attention forward
        structural_out = []
        for t in range(0, self.num_time_steps):  # 遍历每一个时间步，节点做GAT操作;
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]; 节点聚合邻居后的特征

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]  # 节点数量
        out_dim = structural_outputs[-1].shape[-1]  # 输出特征的数量
        structural_outputs_padded = []
        for out in structural_outputs:  # 对节点进行补0，使其为同一个维度
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)  # padding节点的数量; 保持一定的维度;
            padded = torch.cat((out, zero_padding), dim=0)  # 节点特征拼接
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]; 16个时刻拼接在一起; structural最终输出的节点特征
        
        # Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs_padded)
        
        return temporal_out

    def build_model(self):
        input_dim = self.num_features

        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):  # 结构层信息
            layer = StructuralAttentionLayer(input_dim=input_dim,  # featurs; 143
                                             output_dim=self.structural_layer_config[i],  # output维度
                                             n_heads=self.structural_head_config[i],  # 多头参数
                                             attn_drop=self.spatial_drop,  # drop参数
                                             ffd_drop=self.spatial_drop,
                                             residual=self.args.residual) # 残差连接
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]  # 下一层input维度等于上一层的输出维度
        
        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]  # 时序层信息
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):  # [128]
            layer = TemporalAttentionLayer(input_dim=input_dim,  # 输入维度
                                           n_heads=self.temporal_head_config[i],  # 多头数量
                                           num_time_steps=self.num_time_steps,  # 时间维度
                                           attn_drop=self.temporal_drop,  # dropout
                                           residual=self.args.residual)  # 残差连接
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers  # 定义的结构层和时序层

    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values()
        # run gnn
        final_emb = self.forward(graphs) # [N, T, F]
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            emb_t = final_emb[:, t, :].squeeze() #[N, F];  这一时刻，所有节点的embedding
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)  # 节点内积操作
            neg_score = -torch.sum(source_node_emb[:, None, :]*tart_node_neg_emb, dim=2).flatten()
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + self.args.neg_weight*neg_loss
            self.graph_loss += graphloss
        return self.graph_loss

            




