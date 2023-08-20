import numpy as np
import networkx as nx
import random

# DISCLAIMER:
# Parts of this code file are derived from
#  https://github.com/aditya-grover/node2vec

'''Random walk sampling code'''

class Graph_RandomWalk():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:  # 游走长度20
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))  # 按照节点的邻居节点排序进行概率选择;
            if len(cur_nbrs) > 0:
                if len(walk) == 1:  # 初始节点
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])  # alias: 采样，概率
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],  # node2vec
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):  # 每个节点遍历的次数
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):  # dst节点的邻居节点
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)  # node2vec向回走概率
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]  # 下一次游走的概率

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G  # G.adj
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():  # 节点到下一个节点的采样概率
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]  # 节点邻居节点的权重
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]  # 归一化操作
            alias_nodes[node] = alias_setup(normalized_probs)  # alias采样，目的是降低时间复杂度

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():  # node2vec的采样概率
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])  # node2vec随机游走的概率
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])  # 有向边计算

        self.alias_nodes = alias_nodes  # 节点到下个节点的概率（第一次游走使用）
        self.alias_edges = alias_edges  # node2vec采样概率

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))  # 1~N采样
    if np.random.rand() < q[kk]:  # 0～1生成随机数
        return kk
    else:
        return J[kk]
