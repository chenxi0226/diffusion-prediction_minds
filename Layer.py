import math

import torch.nn as nn
import torch

'''动态级联超图卷积层'''
class HypergraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, hypergraph):
        X = self.theta(X)
        Y = hypergraph.v2e(X, aggr="mean")
        X_ = hypergraph.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_))
        return X_


'''社交图GCN卷积层，得到用户embedding'''
class GCNconv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5):
        super(GCNconv, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, relationgraph):
        X = self.theta(X)
        X_ = relationgraph.smoothing_with_GCN(X)
        X_ = self.drop(self.act(X_))
        return X_

'''社交图GraphSAGE卷积层，得到用户embedding'''
class GraphSAGEConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5):
        super(GraphSAGEConv, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim*2, output_dim, bias=bias)

    def forward(self, X, relationgraph):
        X_nbr = relationgraph.v2v(X, aggr="mean")
        X = torch.cat([X, X_nbr], dim=1)
        X_ = self.theta(X)
        X_ = self.drop(self.act(X_))
        return X_

'''级联有向图GAT卷积层'''
class GATConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5, atten_neg_slope=0.2):
        super().__init__()
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)
        self.atten_src = nn.Linear(output_dim, 1, bias=False)
        self.atten_dst = nn.Linear(output_dim, 1, bias=False)

    def forward(self, X, cascade_graph):
        X = self.theta(X)
        X_for_src = self.atten_src(X)
        X_for_dst = self.atten_dst(X)
        e_atten_score = X_for_src[cascade_graph.e_src] + X_for_dst[cascade_graph.e_dst]
        e_atten_score = self.atten_dropout(self.act(e_atten_score).squeeze())
        X_ = cascade_graph.v2v(X, aggr='softmax_then_sum', e_weight=e_atten_score)
        X_ = self.act(X_)
        return X_



class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, is_regularize=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_regularize = is_regularize
        self.U = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.b = nn.Parameter(torch.Tensor(self.output_dim))

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.input_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, shared_embedding, device):
        loss_l2 = torch.zeros(1, dtype=torch.float32, device=device)
        output = shared_embedding @ self.U + self.b
        if self.is_regularize:
            loss_l2 += torch.norm(self.U)**2/2 + torch.norm(self.b)**2/2

        return output, loss_l2


class HGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.a_ne = nn.Linear(out_dim, 1, bias=False)

        # a2: 用于 Edge -> Node
        self.a_en = nn.Linear(out_dim, 1, bias=False)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, X, hg):
        """
        X: [N, C] 用户 Embedding
        hg: dhg.Hypergraph 对象
        """
        X_trans = self.W_q(X)  # [N, out_dim]

        Y_temp = hg.v2e(X_trans, aggr='mean')
        e_weight = torch.sigmoid(self.a_ne(Y_temp))  # [M, 1] 边的权重
        Y = hg.v2e(X_trans, aggr='mean') * e_weight  # [M, out_dim]

        n_weight = torch.sigmoid(self.a_en(X_trans))  # [N, 1] 节点的接受门控

        X_new = hg.e2v(Y, aggr='mean') * n_weight

        X_out = self.act(X_new + X_trans)  # 残差连接防止梯度消失
        return self.dropout(X_out)
