import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Categorical
from Layer import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # pe: [max_len, d_model] -> [1, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class MacroOracle(nn.Module):
    """
    Environment Part: 宏观预测器。
    在 RL 阶段，它的参数将被冻结 (requires_grad=False)，作为外生 Goal 生成器。
    """

    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1, max_seq_len=200):
        super(MacroOracle, self).__init__()
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                    dim_feedforward=embed_dim * 2,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 只预测规模，不预测动作
        self.macro_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)  # log2(size)
        )

    def forward(self, input_emb, current_seq):
        # input_emb: [B, Seq, Dim]
        src = self.input_proj(input_emb)
        src = self.pos_encoder(src)

        # Padding Mask
        key_padding_mask = (current_seq == 0)

        # Causal Mask (SL通常需要防止看到未来)
        seq_len = src.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) == 1).transpose(0, 1).to(input_emb.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        memory = self.transformer_encoder(src, mask=mask, src_key_padding_mask=key_padding_mask)

        # 简单取 Mean Pooling 或者 Last Valid State 来预测整体规模
        # 这里为了稳定性，建议取 Mean Pooling (忽略 padding)
        mask_expanded = (~key_padding_mask).unsqueeze(-1).float()  # [B, T, 1]
        sum_embeddings = torch.sum(memory * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1.0)
        pooled_memory = sum_embeddings / sum_mask  # [B, Dim]

        pred_log_size = self.macro_head(pooled_memory)  # [B, 1]
        return pred_log_size


class MicroPolicy(nn.Module):
    """
    RL Agent: 决策器。
    接收: 历史 Embeddings + 外生目标 Goal (Scalar)
    输出: 下一步动作 Logits
    """

    def __init__(self, user_size, embed_dim, num_heads=4, num_layers=2, dropout=0.1, max_seq_len=200):
        super(MicroPolicy, self).__init__()
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                    dim_feedforward=embed_dim * 2,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Goal 处理
        self.goal_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh()
        )
        self.fusion_gate = nn.Linear(embed_dim * 2, embed_dim)

        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, user_size)
        )

    def forward(self, input_emb, current_seq, external_goal_log):
        """
        external_goal_log: [B, 1] 来自 MacroOracle 的 detach 过的预测值
        """
        src = self.input_proj(input_emb)
        src = self.pos_encoder(src)

        seq_len = src.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) == 1).transpose(0, 1).to(input_emb.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        key_padding_mask = (current_seq == 0)

        memory = self.transformer_encoder(src, mask=mask, src_key_padding_mask=key_padding_mask)

        # === Goal Conditioning ===
        # 将 goal 扩展到序列长度
        g_emb = self.goal_encoder(external_goal_log)  # [B, D]
        g_emb = g_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, D]

        # Gate 融合
        fusion_input = torch.cat([memory, g_emb], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(fusion_input))

        # 融合状态
        state = memory * (1 - gate) + g_emb * gate

        logits = self.actor_head(state)
        return logits, gate  # 返回 gate 用于监控是否塌缩


class MyModule(nn.Module):
    """
    最终整合模型：Macro-Guided + Dynamic Hypergraph Attention
    """

    def __init__(self, user_size, embed_dim, step_split=8, max_seq_len=200, device=torch.device('cuda')):
        super(MyModule, self).__init__()
        self.user_size = user_size
        self.device = device

        # --- Encoders ---
        # 1. 动态超图 (DyHGAT)
        self.dycas_encoder = DynamicHGAT(user_size, embed_dim, step_split, is_norm=True)
        # 2. 社交图 (RelationGNN)
        self.social_encoder = RelationGNN(user_size, embed_dim, is_norm=True)

        # --- Transformer Core ---
        # 3. 宏观引导 Transformer
        self.macro_oracle = MacroOracle(embed_dim, max_seq_len=max_seq_len)
        self.micro_policy = MicroPolicy(user_size, embed_dim, max_seq_len=max_seq_len)

        self.adj_matrix = None

    def set_adjacency_matrix(self, relation_graph):
        if self.adj_matrix is not None: return
        try:
            edges = relation_graph.e[0] if isinstance(relation_graph.e, tuple) else relation_graph.e
            if not isinstance(edges, torch.Tensor): edges = torch.tensor(edges, dtype=torch.long)
            src, dst = edges[:, 0], edges[:, 1]
            indices = torch.stack([src, dst], dim=0)
            values = torch.ones(src.size(0))
            self.adj_matrix = torch.sparse_coo_tensor(indices, values, (self.user_size, self.user_size)).to(self.device)
        except Exception:
            self.adj_matrix = None

    def lookup_and_fuse(self, current_seq, graph_list, relation_graph):
        """
        辅助函数：获取图 Embedding 并融合
        """
        # 1. 获取全局图 Embedding
        # [User, Dim]
        cas_node_emb = self.dycas_encoder(graph_list, self.device)
        social_node_emb = self.social_encoder(relation_graph)

        # 2. 简单加和融合 (也可以做 Concat + Linear)
        # 这种 Early Fusion 让 Transformer 同时看到两类图的信息
        global_node_emb = cas_node_emb + social_node_emb

        # 3. Lookup 序列 Embedding
        seq_emb = F.embedding(torch.clamp(current_seq, 0, self.user_size - 1), global_node_emb)
        return seq_emb, global_node_emb

    def get_topological_mask(self, current_seq):
        """Mask 逻辑保持不变"""
        batch_size = current_seq.size(0)
        infected_mask = torch.zeros(batch_size, self.user_size, device=self.device)
        infected_mask.scatter_(1, current_seq, 1.0)
        infected_mask[:, 0] = 0
        if self.adj_matrix is not None:
            neighbor_logits = torch.sparse.mm(self.adj_matrix, infected_mask.t()).t()
            topo_mask = torch.where(neighbor_logits > 0, 0.0, -1e9)
            topo_mask.scatter_(1, current_seq, -1e9)
            return topo_mask
        return torch.zeros(batch_size, self.user_size, device=self.device)

    def freeze_oracle(self):
        """核心：冻结 Oracle 参数"""
        print(">>> Freezing MacroOracle...")
        self.macro_oracle.eval()
        for p in self.macro_oracle.parameters():
            p.requires_grad = False

    def forward_sl(self, graph_list, relation_graph, current_seq, gt_size=None):
        """
        SL 阶段：同时训练 Oracle 和 Policy (Warm Start)
        """
        seq_emb, _ = self.lookup_and_fuse(current_seq, graph_list, relation_graph)

        # 1. Oracle 预测 (用于 Loss_macro)
        pred_log_size = self.macro_oracle(seq_emb, current_seq)

        # 2. Policy 预测 (用于 Loss_micro)
        # SL 阶段可以用 GT Size 或者 Pred Size 来训练 Policy
        # 为了稳定性，SL 早期可以用 GT 注入 (Teacher Forcing)
        if gt_size is not None and torch.rand(1).item() < 0.5:
            target_signal = torch.log2(gt_size.float() + 1).unsqueeze(1).to(self.device)
        else:
            target_signal = pred_log_size.detach()  # SL阶段也建议detach，防止Policy梯度干扰Oracle

        actor_logits, _ = self.micro_policy(seq_emb, current_seq, target_signal)

        pred_scalar = torch.pow(2, pred_log_size) - 1
        return actor_logits, pred_scalar

    def get_initial_goal(self, graph_list, relation_graph, init_seq):
        """
        RL 辅助：在 Episode 开始时生成一次 Goal
        """
        with torch.no_grad():
            seq_emb, _ = self.lookup_and_fuse(init_seq, graph_list, relation_graph)
            pred_log_size = self.macro_oracle(seq_emb, init_seq)
        return pred_log_size.detach()  # 绝对阻断梯度

    def forward_policy_step(self, graph_list, relation_graph, current_seq, fixed_goal_log):
        """
        RL 单步：只跑 Policy
        """
        if self.adj_matrix is None: self.set_adjacency_matrix(relation_graph)

        seq_emb, global_emb = self.lookup_and_fuse(current_seq, graph_list, relation_graph)

        # Policy Forward
        actor_logits, gate_val = self.micro_policy(seq_emb, current_seq, fixed_goal_log)

        last_logits = actor_logits[:, -1, :]
        topo_mask = self.get_topological_mask(current_seq)
        masked_logits = last_logits + topo_mask

        return masked_logits, gate_val

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


class DynamicHGAT(nn.Module):

    def __init__(self, input_num, embed_dim, step_split=8, dropout=0.5, is_norm=False):
        super().__init__()
        self.input_num = input_num
        self.embed_dim = embed_dim
        self.step_split = step_split

        # 1. 基础 Embedding
        self.user_embeddings = nn.Embedding(input_num, embed_dim)

        # 2. 结构编码器 (HGAT)
        # 使用两层 HGAT 提取高阶特征
        self.hgat1 = HGATLayer(embed_dim, embed_dim, dropout)
        self.hgat2 = HGATLayer(embed_dim, embed_dim, dropout)

        # 3. 时序位置编码 (Learnable Time Embeddings)
        # 为每一个快照 (Snapshot) 学习一个时间向量，替代 LSTM 的序列处理
        self.time_embeddings = nn.Embedding(step_split + 1, embed_dim)

        # 4. 融合层 (Snapshot Fusion)
        # 将多个时间步的图特征融合为一个
        self.fusion_attention = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Tanh()
        )

        self.norm = nn.LayerNorm(embed_dim) if is_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.user_embeddings.weight)
        init.xavier_normal_(self.time_embeddings.weight)

    def forward(self, hypergraph_list, device=torch.device('cuda')):
        # 基础用户特征
        base_emb = self.user_embeddings.weight  # [N, D]

        snapshot_embeddings = []

        for t, hg in enumerate(hypergraph_list):
            # 获取当前时间步的 Time Embedding
            t_emb = self.time_embeddings(torch.tensor(t).to(device))  # [D]

            # 将 Time Embedding 注入到用户特征中 (Broadcast add)
            # 这样 HGAT 处理时就能感知这是“第几个阶段”的结构
            x_t = base_emb + t_emb

            # HGAT 卷积
            x_t = self.hgat1(x_t, hg)
            x_t = self.hgat2(x_t, hg)

            snapshot_embeddings.append(x_t.unsqueeze(0))  # [1, N, D]

        # Stack: [T, N, D]
        all_snapshots = torch.cat(snapshot_embeddings, dim=0)

        # --- Temporal Attention Fusion ---
        # 我们不想只拿最后一个时刻 (LSTM style)，而是融合所有历史结构
        # 计算每个快照的权重 alpha_t = softmax(w * x_t)
        # [T, N, 1]
        attn_scores = self.fusion_attention(all_snapshots)
        # 对 T 维度做 Softmax -> [T, N, 1]
        # 这意味着：对于用户 u，模型会自动判断他在 snapshot t 的结构特征是否重要
        attn_weights = F.softmax(attn_scores, dim=0)

        # 加权求和: sum(weight * emb) -> [N, D]
        final_emb = torch.sum(all_snapshots * attn_weights, dim=0)

        return self.norm(final_emb)

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