import re
import os
import itertools
from collections import defaultdict
from itertools import islice, chain

import networkx as nx
import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix

from datetime import datetime
from datetime import timedelta
import dateutil.parser


def lines_per_n(f, n):
    for line in f:
        yield ''.join(chain([line], itertools.islice(f, n - 1)))

def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    return d

if __name__ == "__main__":

    node_data = defaultdict(lambda : ())
    with open('vis.graph.nodeList.json') as f:
        for chunk in lines_per_n(f, 5):
            chunk = chunk.split("\n")
            id_string = chunk[1].split(":")[1]
            x = [x.start() for x in re.finditer('\"', id_string)]
            id =  id_string[x[0]+1:x[1]]  # 计算id
            # 计算name
            name_string = chunk[2].split(":")[1]
            x = [x.start() for x in re.finditer('\"', name_string)]
            name =  name_string[x[0]+1:x[1]]  # 计算name
            # 计算idx
            idx_string = chunk[3].split(":")[1]
            x1 = idx_string.find('(')
            x2 = idx_string.find(')')
            idx =  idx_string[x1+1:x2]
            
            print("ID:{}, IDX:{:<4}, NAME:{}".format(id, idx, name))
            node_data[name] = (id,idx)  # 节点name:(id, 编号)

    links = []
    ts = []
    with open('vis.digraph.allEdges.json') as f:
        for chunk in lines_per_n(f, 5):
            chunk = chunk.split("\n")
            # 连接的边
            name_string = chunk[2].split(":")[1]
            x = [x.start() for x in re.finditer('\"', name_string)]
            from_id, to_id = name_string[x[0]+1:x[1]].split("_")  # src, dst
            # 时间编码
            time_string = chunk[3].split("ISODate")[1]
            x = [x.start() for x in re.finditer('\"', time_string)]
            timestamp = getDateTimeFromISO8601String(time_string[x[0]+1:x[1]])
            ts.append(timestamp)
            links.append((from_id, to_id, timestamp))
    print (min(ts), max(ts))
    print ("# interactions", len(links))
    links.sort(key =lambda x: x[2])  # 时间排序

    # split edges 
    SLICE_MONTHS = 2  # 按月的时间间隔
    START_DATE = min(ts) + timedelta(200)  # 200天
    END_DATE = max(ts) - timedelta(200)  # 结束日期
    print("Spliting Time Interval: \n Start Time : {}, End Time : {}".format(START_DATE, END_DATE))

    slice_links = defaultdict(lambda: nx.MultiGraph())  # 创建图
    for (a, b, time) in links:
        datetime_object = time
        if datetime_object > END_DATE:  # 超过最大时间，认为是最大时间
            months_diff = (END_DATE - START_DATE).days//30  # 如果时间大于最大，则按照最大最小时间间隔
        else:
            months_diff = (datetime_object - START_DATE).days//30  # 计算时间间隔; 按月计算
        slice_id = months_diff // SLICE_MONTHS  # 进一步分割时间
        slice_id = max(slice_id, 0)

        if slice_id not in slice_links.keys():  # 为每个时间创建快照;
            slice_links[slice_id] = nx.MultiGraph()  # 如果该时刻不存在，建立该时刻的graph
            if slice_id > 0:
                slice_links[slice_id].add_nodes_from(slice_links[slice_id-1].nodes(data=True))  # 将前一时间的节点放入该时刻的图中; 节点不会消失
                assert (len(slice_links[slice_id].edges()) ==0)
        slice_links[slice_id].add_edge(a,b, date=datetime_object)  # 添加ab之间的连接边

    # print statics of each graph
    used_nodes = []
    for id, slice in slice_links.items():
        print("In snapshoot {:<2}, #Nodes={:<5}, #Edges={:<5}".format(id, \
                            slice.number_of_nodes(), slice.number_of_edges()))  # 节点和边的数量
        for node in slice.nodes():  # 遍历该图中所有节点
            if not node in used_nodes:
                used_nodes.append(node)  # 全部节点信息
    # remap nodes in graphs. Cause start time is not zero, the node index is not consistent
    nodes_consistent_map = {node:idx for idx, node in enumerate(used_nodes)}  # 建立节点对应到的索引
    for id, slice in slice_links.items():
        slice_links[id] = nx.relabel_nodes(slice, nodes_consistent_map)  # 重新标记图中节点

    # One-Hot features
    onehot = np.identity(slice_links[max(slice_links.keys())].number_of_nodes())  # 最后一个图中的所有节点; 为每个节点建立one-hot向量
    graphs = []
    for id, slice in slice_links.items():
        tmp_feature = []
        for node in slice.nodes():  # 该图中节点
            tmp_feature.append(onehot[node])
        slice.graph["feature"] = csr_matrix(tmp_feature)  # 稀疏矩阵; 添加图中特征
        graphs.append(slice)  # 将图保存在list中
    
    # save
    save_path = "../../data/Enron/graph.pkl"
    # with open(save_path, "wb") as f:
    #     pkl.dump(graphs, f)
    # print("Processed Data Saved at {}".format(save_path))
