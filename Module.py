import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class GraphEncoder(nn.Module):
    """
    轻量级 GCN 编码器，用于提取社交网络中的全局影响力特征
    """

    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(GraphEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # Layer 1
        x = self.linear1(x)
        x = torch.spmm(adj, x)
        x = self.activation(x)
        x = self.dropout(x)
        # Layer 2
        x = self.linear2(x)
        x = torch.spmm(adj, x)
        return x


class GatedFusion(nn.Module):
    """
    创新点：动态门控融合机制
    融合 时序特征(LSTM) 和 社交特征(GCN)，让模型自动决定依赖哪种信息
    """

    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.update_gate = nn.Linear(dim * 2, dim)
        self.source_gate = nn.Linear(dim * 2, dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_lstm, h_social):
        # h_lstm: [batch, dim] (当前的序列状态)
        # h_social: [batch, dim] (当前激活用户的社交特征)

        combined = torch.cat([h_lstm, h_social], dim=-1)
        z = self.sigmoid(self.update_gate(combined))  # 门控系数
        r = self.sigmoid(self.source_gate(combined))

        # 融合后的状态
        h_tilde = self.tanh(r * h_social + h_lstm)
        h_out = (1 - z) * h_lstm + z * h_tilde
        return h_out


class RLCascadeModel(nn.Module):
    def __init__(self, user_size, embed_dim, hidden_dim, adj_indices, adj_values, device, max_len=200):
        super(RLCascadeModel, self).__init__()
        self.user_size = user_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # --- 1. Buffer for Graph ---
        self.register_buffer('adj_indices', adj_indices)
        self.register_buffer('adj_values', adj_values)
        self.adj_shape = torch.Size([user_size, user_size])

        # --- 2. Base Embeddings ---
        self.user_embedding = nn.Embedding(user_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)  # 位置编码

        # --- 3. Encoders ---
        self.social_gcn = GraphEncoder(embed_dim, embed_dim)
        self.cascade_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)  # 使用 GRU 比 LSTM 更适合短序列
        self.fusion = GatedFusion(hidden_dim)

        # --- 4. Three-Head Architecture (关键解耦) ---
        # A. Actor: 预测下一个用户 (Micro)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, user_size)
        )

        # B. Critic: 预测 RL Value (仅用于 RL 训练辅助)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # C. Size Predictor: 预测级联最终长度 (Macro)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.user_embedding.weight)
        init.xavier_uniform_(self.pos_embedding.weight)

    def get_social_graph_emb(self):
        """一次性计算全图的社交特征"""
        adj = torch.sparse_coo_tensor(self.adj_indices, self.adj_values, self.adj_shape)
        return self.social_gcn(self.user_embedding.weight, adj)

    def forward(self, input_seq):
        """
        监督学习模式：输入全序列，输出所有时刻的预测
        """
        batch_size, seq_len = input_seq.size()

        # 1. Prepare Features
        social_embs_all = self.get_social_graph_emb()  # (N, dim)
        seq_social_emb = F.embedding(input_seq, social_embs_all)  # (Batch, Seq, dim)

        # 添加位置编码
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        seq_input = seq_social_emb + self.pos_embedding(positions)

        # 2. Sequential Encoding
        rnn_out, _ = self.cascade_rnn(seq_input)  # (Batch, Seq, Hidden)

        # 3. Fusion (模拟每一步的融合)
        # 监督学习下为了并行，简化为直接使用 RNN out，
        # 或者可以将 RNN out 和 当前节点的 Social emb 再次融合
        fused_state = self.fusion(rnn_out, seq_social_emb)

        # 4. Heads
        micro_logits = self.actor_head(fused_state)  # (Batch, Seq, User_Size)
        pred_size = self.size_head(fused_state)  # (Batch, Seq, 1)
        # 注意：这里不需要输出 critic value，因为监督学习不需要 V 值

        return micro_logits, pred_size

    def forward_step(self, input_node, hidden_state, step_idx):
        """
        RL Rollout 模式：单步预测
        input_node: (Batch, 1)
        hidden_state: GRU hidden state
        step_idx: int, 当前步数(用于位置编码)
        """
        # 1. Prepare Features
        social_embs_all = self.get_social_graph_emb()
        node_social_emb = F.embedding(input_node, social_embs_all)  # (Batch, 1, dim)

        # 位置编码
        pos_idx = torch.tensor([step_idx], device=self.device).view(1, 1)
        step_input = node_social_emb + self.pos_embedding(pos_idx)

        # 2. RNN Step
        rnn_out, new_hidden = self.cascade_rnn(step_input, hidden_state)  # rnn_out: (Batch, 1, Hidden)

        # 3. Fusion
        fused_state = self.fusion(rnn_out.squeeze(1), node_social_emb.squeeze(1))  # (Batch, Hidden)

        # 4. Heads
        micro_logits = self.actor_head(fused_state)  # (Batch, User_Size)
        pred_value = self.critic_head(fused_state)  # (Batch, 1) -> 这一步状态好不好？
        pred_size = self.size_head(fused_state)  # (Batch, 1) -> 预测最终长度是多少？

        return micro_logits, pred_value, pred_size, new_hidden