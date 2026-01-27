import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Categorical
from Layer import *

class StructureAwareQCritic(nn.Module):
    """
    输入:
      1. state_emb: SharedLSTM 输出的隐向量 (体现时序和语义)
      2. action_emb: 候选动作的 Embedding
      3. explicit_feats: [Connectivity_Score, Is_Infected] (体现拓扑因果性)
    输出:
      Q(s, a) 标量, 评估在状态 s 下采取动作 a (选择特定用户) 的价值。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(StructureAwareQCritic, self).__init__()
        # 输入维度 = 状态维度 + 动作维度 + 2个显式特征
        self.input_dim = state_dim + action_dim + 2

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # LeakyRLU 对稀疏信号更友好
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output Q-Value
        )
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)

    def forward(self, state_emb, action_emb, explicit_feats):
        # state_emb: [B, D]
        # action_emb: [B, D]
        # explicit_feats: [B, 2]
        x = torch.cat([state_emb, action_emb, explicit_feats], dim=-1)
        return self.net(x)

class RL_MINDS_StructureAware(nn.Module):
    def __init__(self, user_size, embed_dim, step_split=8, max_seq_len=200, device=torch.device('cuda')):
        super(RL_MINDS_StructureAware, self).__init__()
        self.user_size = user_size
        self.emb_dim = embed_dim
        self.device = device

        # --- Encoders ---
        self.dycasHGNN = DynamicCasHGNN(user_size, embed_dim, step_split)
        self.relationGNN = RelationGNN(user_size, embed_dim)
        self.relationLSTM = RelationLSTM(embed_dim)
        self.cascadeLSTM = CascadeLSTM(embed_dim)
        self.sharedLSTM = SharedLSTM(max_seq_len, embed_dim)

        # --- Embeddings & Projections ---
        self.user_embedding = nn.Embedding(user_size, embed_dim)
        self.W_micro = nn.Linear(embed_dim, embed_dim)
        self.W_macro = nn.Linear(embed_dim, embed_dim)

        # --- Heads ---
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, user_size)
        )

        # Critic 输入维度：s_micro(D) + pred_macro(1) = D+1
        self.critic = StructureAwareQCritic(state_dim=embed_dim + 1, action_dim=embed_dim)
        from copy import deepcopy
        self.critic_target = deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.tau = 0.001

        self.macro_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.adj_matrix = None
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.W_micro.weight)
        init.xavier_normal_(self.W_macro.weight)
        init.xavier_normal_(self.user_embedding.weight)

    def soft_update_target(self):
        for parm, target_parm in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_parm.data.copy_(parm.data * self.tau + (1.0 - self.tau) * target_parm.data)

    def set_adjacency_matrix(self, relation_graph):
        if self.adj_matrix is not None: return
        try:
            # 1. 获取 raw 数据
            # 报错显示 relation_graph.e[0] 是 [E, 2]，说明它包含了完整的边信息
            raw_data = relation_graph.e

            src, dst = None, None

            # 情况 A: 如果是元组 (Edges, Weights)
            if isinstance(raw_data, tuple):
                # 通常第一个元素是边索引
                edges = raw_data[0]
            else:
                edges = raw_data

            # 2. 统一转为 Tensor 并移动到设备
            if not isinstance(edges, torch.Tensor):
                edges = torch.tensor(edges, dtype=torch.long)
            edges = edges.to(self.device)

            # 3. 根据形状解析 src, dst
            # 如果形状是 [E, 2] -> 每一行是 (u, v)
            if edges.dim() == 2 and edges.shape[1] == 2:
                src = edges[:, 0]
                dst = edges[:, 1]
            # 如果形状是 [2, E] -> 第一行是 u, 第二行是 v
            elif edges.dim() == 2 and edges.shape[0] == 2:
                src = edges[0]
                dst = edges[1]
            # 如果原本就是 tuple(src, dst) 且被我们误判了，再尝试解包
            elif isinstance(raw_data, tuple) and len(raw_data) == 2 and edges.dim() == 1:
                src = raw_data[0].to(self.device).long()
                dst = raw_data[1].to(self.device).long()
            else:
                raise ValueError(f"Unexpected edge shape: {edges.shape}")

            # 4. 构建稀疏矩阵
            # 确保 src, dst 都是 1D Tensor
            src = src.contiguous().view(-1)
            dst = dst.contiguous().view(-1)

            indices = torch.stack([src, dst], dim=0)
            values = torch.ones(src.size(0)).to(self.device)
            coo_adj = torch.sparse_coo_tensor(
                indices, values, (self.user_size, self.user_size)
            ).to(self.device)
            self.adj_matrix = torch.sparse_coo_tensor(
                indices, values, (self.user_size, self.user_size)
            ).to(self.device)

            # print(f"✅ Adjacency Matrix built. Edges: {src.size(0)}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to build adjacency matrix: {e}")
            # Fallback
            self.adj_matrix = torch.sparse_coo_tensor(
                torch.empty(2, 0).long().to(self.device),
                torch.empty(0).to(self.device),
                (self.user_size, self.user_size)
            )

    def lookup_embedding(self, examples, embeddings):
        # 限制 index 范围防止越界
        examples = torch.clamp(examples, 0, embeddings.size(0) - 1)
        return F.embedding(examples, embeddings)

    def get_shared_state(self, graph_list, relation_graph, examples):
        # 1. 全局图特征 (Batch无关)
        user_cas_embedding = self.dycasHGNN(graph_list, self.device)  # [User, D]
        user_social_embedding = self.relationGNN(relation_graph)  # [User, D]

        # 2. Batch 序列特征
        sender_social_embedding = self.relationLSTM(examples, user_social_embedding)

        # lookup 将全局特征映射到当前 Batch
        sender_cas_embedding_share = self.lookup_embedding(examples, user_cas_embedding)
        sender_social_embedding_share = self.lookup_embedding(examples, user_social_embedding)

        # 3. 融合
        shared_embedding, _ = self.sharedLSTM(sender_cas_embedding_share, sender_social_embedding_share)

        return shared_embedding, user_social_embedding

    def compute_explicit_features(self, current_seq, action_candidates):
        batch_size = current_seq.size(0)

        # 1. Connectivity (稀疏矩阵乘法)
        seq_multi_hot = torch.zeros(batch_size, self.user_size, device=self.device)
        seq_multi_hot.scatter_(1, current_seq, 1.0)
        seq_multi_hot[:, 0] = 0  # Mask PAD

        if self.adj_matrix is not None:
            # [User, User] @ [User, B] -> [User, B] -> [B, User]
            all_connectivity = torch.sparse.mm(self.adj_matrix, seq_multi_hot.t()).t()
            # Gather specific action
            chosen_connectivity = all_connectivity.gather(1, action_candidates.unsqueeze(1))
            seq_lens = (current_seq != 0).sum(dim=1, keepdim=True).float() + 1e-5
            feature_connect = chosen_connectivity / seq_lens
        else:
            feature_connect = torch.zeros(batch_size, 1, device=self.device)

        # 2. Is Infected
        is_infected = (current_seq == action_candidates.unsqueeze(1)).any(dim=1).float().unsqueeze(1)

        return torch.cat([feature_connect, is_infected], dim=-1)

    def get_q_value(self, s_micro, action, graph_emb, current_seq):
        """
        Critic 闭环接口:
        s_micro: State [B, D]
        action: Action Index [B]
        graph_emb: 全图 Embedding [User, D] (用于查找 Action 的向量表示)
        """
        # 1. Action Embedding Lookup
        a_emb = F.embedding(action, graph_emb)

        # 2. Explicit Features
        explicit_feats = self.compute_explicit_features(current_seq, action)

        # 3. Q-Net
        return self.critic(s_micro, a_emb, explicit_feats)

    def get_q_value_with_macro(self, s_micro, pred_macro, action, graph_emb, current_seq):
        """
        Augmented Q interface which includes the macro prediction scalar in the state.
        - s_micro: [B, D]
        - pred_macro: [B] or [B,1] scalar estimate of final cascade size
        Returns: Q(s_aug, a) [B, 1]
        """
        if pred_macro.dim() == 1:
            pred_macro = pred_macro.unsqueeze(1)
        s_aug = torch.cat([s_micro, torch.log2(pred_macro + 1.0) / 7.0], dim=-1)  # [B, D+1]
        a_emb = F.embedding(action, graph_emb)
        explicit_feats = self.compute_explicit_features(current_seq, action)
        return self.critic(s_aug, a_emb, explicit_feats)

    def forward(self, graph_list, relation_graph, examples):
        shared_emb, _ = self.get_shared_state(graph_list, relation_graph, examples)

        s_micro = torch.tanh(self.W_micro(shared_emb))
        s_macro = torch.tanh(self.W_macro(shared_emb))

        actor_logits = self.actor_head(s_micro)

        # Macro Pooling
        batch_size = examples.size(0)
        example_len = torch.count_nonzero(examples, 1)
        s_macro_last = []
        for i in range(batch_size):
            idx = example_len[i] - 1
            if idx < 0: idx = 0
            s_macro_last.append(s_macro[i, idx, :])
        s_macro_last = torch.stack(s_macro_last, dim=0)

        pred_macro = self.macro_head(s_macro_last)

        return actor_logits, pred_macro

    def get_topological_mask(self, current_seq):
        batch_size = current_seq.size(0)
        # 1. 构建当前感染者的 Multi-hot 向量 [B, User]
        infected_mask = torch.zeros(batch_size, self.user_size, device=self.device)
        infected_mask.scatter_(1, current_seq, 1.0)
        infected_mask[:, 0] = 0  # 排除 PAD 位置

        if self.adj_matrix is not None:
            # 2. 修正后的稀疏矩阵乘法逻辑
            # 注意：torch.sparse.mm 在 CUDA 上要求第一个参数为稀疏张量
            # 我们计算 (Adj @ Infected.T).T 来获得 [B, User] 维度的邻居特征
            # adj_matrix: [User, User], infected_mask.t(): [User, B]
            neighbor_logits = torch.sparse.mm(self.adj_matrix, infected_mask.t()).t()

            # 3. 构造掩码：(不是邻居 AND 不是已感染者) 的位置设为 -inf
            topo_mask = torch.where(neighbor_logits > 0, 0.0, -1e9)

            # 4. 强制排除已感染者 (防止回环)
            topo_mask.scatter_(1, current_seq, -1e9)
            return topo_mask

        return torch.zeros(batch_size, self.user_size, device=self.device)

    def forward_step(self, graph_list, relation_graph, current_seq):
        """RL Rollout 接口：增加拓扑约束"""
        if self.adj_matrix is None:
            self.set_adjacency_matrix(relation_graph)

        shared_emb, graph_emb = self.get_shared_state(graph_list, relation_graph, current_seq)
        s_t = shared_emb[:, -1, :]

        s_micro = torch.tanh(self.W_micro(s_t))
        s_macro = torch.tanh(self.W_macro(s_t))

        logits = self.actor_head(s_micro)

        # --- 关键修改：应用拓扑过滤 ---
        topo_mask = self.get_topological_mask(current_seq)
        logits = logits + topo_mask
        pred_size = self.macro_head(s_macro)
        return logits, pred_size, s_micro, graph_emb

class RelationGNN(nn.Module):
    '''社交图GNN'''

    def __init__(self, input_num, embed_dim, dropout=0.5, is_norm=False):
        super().__init__()
        self.user_embedding = nn.Embedding(input_num, embed_dim)
        # self.gcn = GCNconv(embed_dim, embed_dim)
        self.graphsage = GraphSAGEConv(embed_dim, embed_dim)
        self.is_norm = is_norm
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(embed_dim)
        # self.lstm = nn.LSTM(self.embed_dim, self.embed_dim)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.user_embedding.weight)

    def forward(self, relation_graph):
        # gnn_embeddings = self.gcn(self.user_embedding.weight, relation_graph)
        gnn_embeddings = self.graphsage(self.user_embedding.weight, relation_graph)
        gnn_embeddings = self.dropout(gnn_embeddings)
        if self.is_norm:
            gnn_embeddings = self.batch_norm(gnn_embeddings)

        # output_embeddings = gnn_embeddings.unsqueeze(1)  # (input_num, embed_dim) → (input_num, 1, embed_dim)
        # output_embeddings, (h, c) = self.lstm(output_embeddings)
        # output_embeddings = output_embeddings.squeeze(1) # (input_num, 1, embed_dim) → (input_num, embed_dim)
        # return output_embeddings
        return gnn_embeddings


class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        '''
        hidden: 这个子超图HGAT的输入，dy_emb: 这个子超图HGAT的输出
        hidden和dy_emb都是用户embedding矩阵，大小为(用户数, 64)
        '''
        # tensor.unsqueeze(dim) 扩展维度，返回一个新的向量，对输入的既定位置插入维度1
        # tensor.cat(inputs, dim=?) --> Tensor    inputs：待连接的张量序列     dim：选择的扩维，沿着此维连接张量序列
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = nn.functional.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)  # 随机丢弃每个用户embedding的权重
        out = torch.sum(emb_score * emb, dim=0)  # 将输入的embedding和输出的embedding按照对应的用户加权求和
        return out

class DynamicCasHGNN(nn.Module):
    '''超图HGNN'''
    def __init__(self, input_num, embed_dim, step_split=8, dropout=0.5, is_norm=False):
        '''
        :param input_num: 用户个数
        :param embed_dim: embedding维度
        :param step_split: 超图序列中的超图个数
        :param dropout: 丢弃率
        :param is_norm: 是否规则化
        '''
        super().__init__()
        self.input_num = input_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.is_norm = is_norm
        self.step_split = step_split
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(self.embed_dim)
        self.user_embeddings = nn.Embedding(self.input_num, self.embed_dim)
        self.hgnn = HypergraphConv(self.embed_dim, self.embed_dim, drop_rate=self.dropout)  # 超图卷积，学习每个超图中的用户embedding
        # self.lstm = nn.LSTM(self.embed_dim, self.embed_dim, num_layers=1, batch_first=True) # LSTM学习超图间的关系
        self.fus = Fusion(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        '''从正态分布中随机初始化每张超图的初始用户embedding'''
        init.xavier_normal_(self.user_embeddings.weight)

    def forward(self, hypergraph_list, device=torch.device('cuda')):
        # 对每张子超图进行卷积
        hg_embeddings = []
        for i in range(len(hypergraph_list)):
            subhg_embedding = self.hgnn(self.user_embeddings.weight, hypergraph_list[i])
            if i == 0:
                hg_embeddings.append(subhg_embedding)
            else:
                subhg_embedding = self.fus(hg_embeddings[-1], subhg_embedding)
                hg_embeddings.append(subhg_embedding)

            # print(f'self.user_embeddings[{i}].weight = {self.user_embeddings[i].weight}')
        # 返回最后一个时刻的用户embedding
        return hg_embeddings[-1]

class RelationLSTM(nn.Module):
    '''LSTM：对从社交图学到的用户embedding做LSTM'''
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm = nn.LSTM(self.embed_dim, self.embed_dim, num_layers=1, batch_first=True)

    def lookup_embedding(self, examples, embeddings):
        output_embedding = []
        for example in examples:
            index = example.clone().detach()
            temp = torch.index_select(embeddings, dim=0, index=index)
            output_embedding.append(temp)
        output_embedding = torch.stack(output_embedding, 0)
        return output_embedding

    def forward(self, examples, user_social_embedding):
        '''

        :param examples: tensor 级联序列 (batch_size, 200)
        :param user_social_embedding: tensor 用户社交embedding (user_size, emb_dim)
        :return:
        '''
        # example_len = torch.count_nonzero(examples, 1)  # 统计每个观 察到的级联的长度，去掉用户0
        user_embedding = self.lookup_embedding(examples, user_social_embedding) # (batch_size, 200, emb_dim)
        output_embedding, (h_t, c_t) = self.lstm(user_embedding)    # (batch_size, 200, emb_dim)
        # hidden = [] # 每个级联序列的最终时刻的表示
        # for i in range(len(example_len)):
        #     hidden.append(output_embedding[i][example_len[i]])
        # hidden.size() = (batch_size, emb_dim)
        # return output_embedding
        return output_embedding

class CascadeLSTM(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, num_layers=1, batch_first=True)

    def lookup_embedding(self, examples, embeddings):
        output_embedding = []
        for example in examples:
            index = example.clone().detach()
            temp = torch.index_select(embeddings, dim=0, index=index)
            output_embedding.append(temp)
        output_embedding = torch.stack(output_embedding, 0)
        return output_embedding

    def forward(self, examples, user_cas_embedding):
        '''
        :param examples: tensor 级联序列 (batch_size, 200)
        :param user_cas_embedding: tensor 动态级联图中的用户embedding (user_size, emb_dim)
        :return:
        '''
        cas_embedding = self.lookup_embedding(examples, user_cas_embedding)
        # output.size()=(input_num, step_split, embed_dim)
        # h.size()=(1, input_num, embed_dim) lstm中的参数
        # c.size()=(1, input_num, embed_dim) lstm中的参数
        output_embedding, (h_t, c_t) = self.lstm(cas_embedding)

        return output_embedding

class SharedLSTM(nn.Module):
    '''共享LSTM'''
    def __init__(self, input_size, emb_dim):
        '''
        :param input_size: 一个级联序列中的用户个数，默认200
        :param emb_dim: embedding维度
        '''
        super().__init__()
        self.input_size = input_size
        self.emb_dim = emb_dim
        # 处理从级联图中学到的用户向量
        self.W_i = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        # 处理从社交图中学到的用户向量
        self.U_i = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        # 处理隐向量
        self.V_i = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        # 偏置
        self.b_i = nn.Parameter(torch.Tensor(emb_dim))

        # 遗忘门 f_t
        self.W_f = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.U_f = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.V_f = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.b_f = nn.Parameter(torch.Tensor(emb_dim))

        # 输入门 c_t
        self.W_c = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.U_c = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.V_c = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.b_c = nn.Parameter(torch.Tensor(emb_dim))

        # 输出门 o_t
        self.W_o = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.U_o = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.V_o = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.b_o = nn.Parameter(torch.Tensor(emb_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, cas_emb, social_emb, init_states=None):
        '''
        :param cas_emb: 级联图HGNN 学来的用户embedding     (batch_size, 200, emb_dim)
        :param social_emb: 社交图GNN 学来的用户embedding   (batch_size, 200, emb_dim)
        :param init_states: 初始状态，可忽略
        :return: hidden_seq: 最后一层的状态(batch_size, 200, emb_dim)
        '''
        bs, seq_sz, _ = cas_emb.size()    # (batch_size, 200, emb_dim)
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.emb_dim).to(cas_emb.device),
                torch.zeros(bs, self.emb_dim).to(cas_emb.device)
            )
        else:
            h_t, c_t = init_states
        for t in range(seq_sz):
            cas_emb_t = cas_emb[:, t, :]
            social_emb_t = social_emb[:, t, :]

            i_t = torch.sigmoid(cas_emb_t @ self.W_i + social_emb_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(cas_emb_t @ self.W_f + social_emb_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(cas_emb_t @ self.W_c + social_emb_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(cas_emb_t @ self.W_o + social_emb_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape(sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X):
        out = self.relu1(self.linear1(X))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)

        return out