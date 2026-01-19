# Part of this file is derived from
# https://github.com/albertyang33/FOREST

import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle


class Options(object):

    def __init__(self, data_name='douban'):
        self.data = 'data/' + data_name + '/cascades.txt'
        self.u2idx_dict = 'data/' + data_name + '/u2idx.pickle'
        self.idx2u_dict = 'data/' + data_name + '/idx2u.pickle'
        self.save_path = ''
        self.net_data = 'data/' + data_name + '/edges.txt'
        self.embed_dim = 64


def SplitData(data_name, train_rate=0.8, valid_rate=0.1, random_seed=300, load_dict=True, with_EOS=True):
    options = Options(data_name)
    u2idx = {}
    idx2u = []
    if not load_dict:  # 如果原始数据没处理好，根据原始数据得到用户集合，并保存到本地
        user_size, u2idx, idx2u = buildIndex(options.data)
        with open(options.u2idx_dict, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.idx2u_dict, 'wb') as handle:
            pickle.dump(idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:  # 如果原始数据处理好了，直接读取本地文件
        with open(options.u2idx_dict, 'rb') as handle:
            u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)
        user_size = len(u2idx)

    t_cascades = []  # 级联集合（用户）
    timestamps = []  # 级联集合（时间戳）
    t_cas_len = []  # 级联长度
    for line in open(options.data):
        if len(line.strip()) == 0:
            continue
        #   一个级联={(用户1,时间戳1), (用户2,时间戳2), ..., (用户n,时间戳n)}
        timestamplist = []  # 时间戳
        userlist = []  # 用户
        if data_name == 'memetracker':
            chunks = line.strip().split()
        else:
            chunks = line.strip().split(',')
        # chunks = line.strip().split(',')
        for chunk in chunks:
            try:
                if data_name == 'memetracker' or data_name == 'weibo':
                    user, timestamp = chunk.split(',')
                else:
                    # Twitter,Douban
                    if len(chunk.split()) == 2:
                        user, timestamp = chunk.split()
                    # Android,Christianity
                    elif len(chunk.split()) == 3:
                        root, user, timestamp = chunk.split()
                        if root in u2idx:
                            userlist.append(u2idx[root])
                            timestamplist.append(float(timestamp))
            except:
                print(chunk)
            if user in u2idx:
                userlist.append(u2idx[user])
                timestamplist.append(float(timestamp))

        if len(userlist) > 1 and len(userlist) <= 500:
            # t_cas_len.append(len(userlist))
            if with_EOS:
                userlist.append(Constants.EOS)
                timestamplist.append(Constants.EOS)
            t_cascades.append(userlist)
            timestamps.append(timestamplist)

    '''ordered by timestamps'''  # 按照级联开始的时间，升序排列所有级联
    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1])]
    timestamps = sorted(timestamps)
    t_cascades[:] = [t_cascades[i] for i in order]
    # t_cas_len[:] = [t_cas_len[i] for i in order]
    cas_idx = [i for i in range(len(t_cascades))]

    '''data split'''
    train_idx_ = int(train_rate * len(t_cascades))
    train = t_cascades[0:train_idx_]
    train_t = timestamps[0:train_idx_]
    train_idx = cas_idx[0:train_idx_]
    # train_len = t_cas_len[0:train_idx_]
    shuffer_indices = list(range(len(train_idx)))
    random.seed(random_seed)
    random.shuffle(shuffer_indices)
    train = [train[i] for i in shuffer_indices]
    train_t = [train_t[i] for i in shuffer_indices]
    train_idx = [train_idx[i] for i in shuffer_indices]
    train = [train, train_t, train_idx]

    valid_idx_ = int((train_rate + valid_rate) * len(t_cascades))
    valid = t_cascades[train_idx_:valid_idx_]
    valid_t = timestamps[train_idx_:valid_idx_]
    valid_idx = cas_idx[train_idx_:valid_idx_]
    valid = [valid, valid_t, valid_idx]

    test = t_cascades[valid_idx_:]
    test_t = timestamps[valid_idx_:]
    test_idx = cas_idx[valid_idx_:]
    test = [test, test_t, test_idx]

    total_len = sum(len(i) - 1 for i in t_cascades)
    train_size = len(train_t)
    valid_size = len(valid_t)
    test_size = len(test_t)
    print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
    print("total size:%d " % (len(t_cascades)))
    print("average length:%f" % (total_len / len(t_cascades)))
    print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
    print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))
    print("user size:%d" % (user_size - 2))

    return user_size, t_cascades, timestamps, train, valid, test


"""
    得到用户集合user_set
    cascade.txt中每一行代表一个级联
    一个级联的开头是(根用户，用户，时间戳)元组，后面都是(用户，时间戳)元组
"""


def buildIndex(data):
    user_set = set()  # 用户集合
    u2idx = {}  # u2idx[user]=pos
    idx2u = []  # idx2u[pos]=user

    lineid = 0
    # 自动判断是否是 memetracker 数据（根据路径或文件名）
    is_memetracker = 'memetracker' in data

    for line in open(data):
        lineid += 1
        if len(line.strip()) == 0:
            continue

        # --- 修改点 1: 根据数据集类型选择切分符 ---
        if is_memetracker:
            chunks = line.strip().split()  # Memetracker 用空格分块
        else:
            chunks = line.strip().split(',')  # 其他用逗号分块

        for chunk in chunks:
            user = None  # 初始化 user 防止报错
            try:
                # --- 修改点 2: 增加对 u,t 格式的兼容 ---
                # 情况 A: 也就是 Memetracker (user,timestamp)
                if ',' in chunk and len(chunk.split(',')) == 2:
                    user, timestamp = chunk.split(',')

                # 情况 B: 原有逻辑 (user timestamp)
                elif len(chunk.split()) == 2:
                    user, timestamp = chunk.split()

                # 情况 C: 原有逻辑 (root user timestamp)
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()
                    user_set.add(root)

            except:
                print(f"Error line {lineid}: {chunk}")

            # --- 修改点 3: 只有 user 被成功赋值才加入集合 ---
            if user is not None:
                user_set.add(user)

    pos = 0
    u2idx['<blank>'] = pos
    idx2u.append('<blank>')
    pos += 1
    u2idx['</s>'] = pos
    idx2u.append('</s>')
    pos += 1

    for user in user_set:
        u2idx[user] = pos
        idx2u.append(user)
        pos += 1
    user_size = len(user_set) + 2
    print("user_size : %d" % (user_size))
    return user_size, u2idx, idx2u

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, cas, batch_size=64, load_dict=True, cuda=True, test=False, with_EOS=True):
        self._batch_size = batch_size
        self.cas = cas[0]
        self.time = cas[1]
        self.idx = cas[2]
        self.len = []
        for cas in self.cas:
            self.len.append(len(cas))
        self.test = test
        self.with_EOS = with_EOS
        self.cuda = cuda

        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = 200
            # max_len = 20

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
                for inst in insts])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            # 每个级联的长度（参与级联的用户个数）不同，pad_to_longest将级联的长度统一为200
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            # pytorch两个基本对象：Tensor（张量）和Variable（变量）
            # tensor不能方向传播，variable可以反向传播
            seq_idx = Variable(
                torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test)
            seq_len = Variable(torch.LongTensor(self.len[start_idx:end_idx]), volatile=self.test)

            return seq_data, seq_data_timestamp, seq_idx, seq_len
        else:

            self._iter_count = 0
            raise StopIteration()
