import torch
import dhg
import pickle
import os

from DataSet import *
# from DataLoader import *


# 构建社交普通图
def RelationGraph(data_name, device):
    data = Options(data_name)
    _u2idx = {}

    with open(data.u2idx_dict, 'rb') as f:
        _u2idx = pickle.load(f)

    if os.path.exists(data.net_data):
        with open(data.net_data, 'r') as f:
            edge_list = f.read().strip().split('\n')
            if data_name == 'douban' or data_name == 'twitter' or data_name == 'android':
                edge_list = [edge.split(',') for edge in edge_list]
            else:
                edge_list = [edge.split(' ') for edge in edge_list]

            edge_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in edge_list \
                         if edge[0] in _u2idx and edge[1] in _u2idx]
    else:
        print(f"🔴 社交网络文件未找到！预期的路径是: {data.net_data}")
        return None

    user_size = len(_u2idx)
    relation_graph = dhg.Graph(user_size, edge_list, device=device)
    print(f'#Link: {len(relation_graph.e[0])}')
    return relation_graph


def CascadeHypergraph(cascades, user_size, device):
    # cascades = cascades.tolist()
    edge_list = []
    for cascade in cascades:
        cascade = set(cascade)
        if len(cascade) > 2:
            cascade.discard(0)
        edge_list.append(cascade)

    cascade_hypergraph = dhg.Hypergraph(user_size, edge_list, device=device)

    return cascade_hypergraph


'''
Part of this function is derived from
https://github.com/slingling/MS-HGAT
'''
def DynamicCasHypergraph(examples, examples_times, user_size, device, step_split=8):
    '''
    :param examples: 级联（用户）
    :param examples_times: 级联时间戳（用户参与级联的时间）
    :param user_size: 数据集中的所有用户
    :param device: 所在设备
    :param step_split: 划分几个超图
    :return: 超图序列
    '''

    hypergraph_list = []
    time_sorted = []
    for time in examples_times:
        time_sorted += time[:-1]
    time_sorted = sorted(time_sorted)   # 将所有时间戳升序排列
    split_length = len(time_sorted) // step_split    # 一个时间段包含的时间戳个数
    start_time = 0
    end_time = 0


    for x in range(split_length, split_length * step_split, split_length):
        # if x == split_length:
        #     end_time = time_sorted[x]
        # else:
        #     end_time = time_sorted[x]
        start_time = end_time
        end_time = time_sorted[x]

        selected_examples = []
        for i in range(len(examples)):
            example = examples[i]
            example_times = examples_times[i]
            if isinstance(example, list):
                example = torch.tensor(example)
                example_times = torch.tensor(example_times, dtype=torch.float64)
            selected_example = torch.where((example_times < end_time) & (example_times > start_time), example, torch.zeros_like(example))
            # print(selected_example)
            selected_examples.append(selected_example.numpy().tolist())

        sub_hypergraph = CascadeHypergraph(selected_examples, user_size, device=device)
        # print(sub_hypergraph)
        hypergraph_list.append(sub_hypergraph)

    # =============== 最后一张超图 ===============
    start_time = end_time
    selected_examples = []
    for i in range(len(examples)):
        example = examples[i]
        example_times = examples_times[i]
        if isinstance(example, list):
            example = torch.tensor(example)
            example_times = torch.tensor(example_times, dtype=torch.float64)
        selected_example = torch.where(example_times > start_time, example, torch.zeros_like(example))
        # print(selected_example)
        selected_examples.append(selected_example.numpy().tolist())
    hypergraph_list.append(CascadeHypergraph(selected_examples, user_size, device=device))

    return hypergraph_list
