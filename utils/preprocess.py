
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from utils.utilities import run_random_walks_n2v

np.random.seed(123)

def load_graphs(dataset_str):
    """Load graph snapshots given the name of dataset"""
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)  # 16个时刻的图，节点，边的信息，节点的特征;
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]  # 每个图的邻接矩阵
    return graphs, adjs

def get_context_pairs(graphs, adjs):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    print("Computing training pairs ...")
    context_pairs_train = []
    for i in range(len(graphs)):
        context_pairs_train.append(run_random_walks_n2v(graphs[i], adjs[i], num_walks=10, walk_len=20))

    return context_pairs_train

def get_evaluation_data(graphs):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx]  # 训练语料, data
    next_graph = graphs[eval_idx+1]  # 测试集下一个图,label
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2,  # 创建数据
                            test_mask_fraction=0.6)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))  # next_graph涉及到的所有edge
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):  # 上一时刻图中是否有该节点
            edges_positive.append(e)  # positive的边
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)  # 负采样
    
    # 划分训练集，测试集，验证集
    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive, 
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos, 
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg
            
def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):  # 采样和positive同等数量的边
        idx_i = np.random.randint(0, nodes_num)  # 随机选择i,j节点
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:  # 自连接
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):  # pos的边
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:  # 存在之前的数据
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg