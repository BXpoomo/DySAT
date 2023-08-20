
import numpy as np
import copy
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from utils.random_walk import Graph_RandomWalk

import torch


"""Random walk-based pair generation."""

def run_random_walks_n2v(graph, adj, num_walks, walk_len):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using 
        the sampling strategy of node2vec (deepwalk)"""
    nx_G = nx.Graph()
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])  # 添加节点; 重复只保留一次; 只将有连接的节点保留，没有连接的去掉（因为在构图时，将之前的节点加入了）
    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]  # 连接的边的权重; 也就是连接的次数
    # node2vec随机游走
    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()  # 游走概率计算
    walks = G.simulate_walks(num_walks, walk_len)  # 随机游走
    WINDOW_SIZE = 10
    pairs = defaultdict(list)
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:  # 该节点的window上下文节点
                if nb_word != word:
                    pairs[word].append(nb_word)  # 和本身节点不同，则作为一对训练语料
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs

def fixed_unigram_candidate_sampler(true_clasees, 
                                    num_true, 
                                    num_sampled, 
                                    unique,  
                                    distortion, 
                                    unigrams):
    # TODO: implementate distortion to unigrams
    assert true_clasees.shape[1] == num_true
    samples = []
    for i in range(true_clasees.shape[0]):  # 遍历正样本节点;
        dist = copy.deepcopy(unigrams)  # 节点的degree
        candidate = list(range(len(dist)))  # self.graphs[t].nodes
        taboo = true_clasees[i].cpu().tolist()  # 这个正样本节点
        for tabo in sorted(taboo, reverse=True):
            candidate.remove(tabo)  # 移除正样本节点
            dist.pop(tabo)  # 该正样本节点degree去除
        sample = np.random.choice(candidate, size=num_sampled, replace=unique, p=dist/np.sum(dist))  # 每个正样本按概率采样10个节点
        samples.append(sample)
    return samples  # 10个正样本，每个正样本采样10个负样本

def to_device(batch, device):
    feed_dict = copy.deepcopy(batch)
    node_1, node_2, node_2_negative, graphs = feed_dict.values()
    # to device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]
    feed_dict["graphs"] = [g.to(device) for g in graphs]

    return feed_dict


        



