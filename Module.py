import math

import torch
import torch.nn as nn
import torch.nn.init as init
from Layer import *

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


class Module(nn.Module):
    def __init__(self, user_size, embed_dim, step_split=8, max_seq_len=200, task_num=2, device=torch.device('cuda')):
        '''
        :param user_size: 数据集中的用户个数
        :param embed_dim: embedding的维度
        :param step_split: 分割的超图数量
        :param max_seq_len: 级联序列的长度
        '''
        super().__init__()
        self.user_size = user_size
        self.emb_dim = embed_dim
        self.step_split = step_split
        self.max_seq_len = max_seq_len
        self.device = device
        self.task_num = task_num
        self.task_label = torch.LongTensor([i for i in range(self.task_num)])
        self.dycasHGNN = DynamicCasHGNN(self.user_size, self.emb_dim, self.step_split)   # HGNN
        self.relationGNN = RelationGNN(self.user_size, self.emb_dim)    # GNN
        self.relationLSTM = RelationLSTM(self.emb_dim)
        self.cascadeLSTM = CascadeLSTM(self.emb_dim)
        self.sharedLSTM = SharedLSTM(self.max_seq_len, self.emb_dim)
        self.shared_linear = LinearLayer(self.emb_dim, self.task_num)   # 判别器
        self.micro_mlp = MLP(self.emb_dim*2, self.emb_dim*4, self.user_size)
        self.macro_mlp = MLP(self.emb_dim*2, self.emb_dim*4, 1)
        # print('初始化方法')
        self.user_embedding = nn.Embedding(self.user_size, self.emb_dim)  # 用户的初始特征
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def lookup_embedding(self, examples, embeddings):
        output_embedding = []
        for example in examples:
            index = example.clone().detach()
            temp = torch.index_select(embeddings, dim=0, index=index)
            output_embedding.append(temp)
        output_embedding = torch.stack(output_embedding, 0)
        return output_embedding

    def adversarial_loss(self, shared_embedding):
        logits, loss_l2 = self.shared_linear(shared_embedding, self.device)
        # label = nn.functional.one_hot(self.task_label, self.task_num).to(self.device)
        loss_adv = torch.zeros(logits.shape[0], device=self.device)
        for task in range(self.task_num):
            label = torch.tensor([task]*logits.shape[0]).to(self.device)
            loss_adv += torch.nn.CrossEntropyLoss(reduce=False)(logits, label.long())

        loss_adv = torch.mean(loss_adv)

        return loss_adv, loss_l2

    def diff_loss(self, shared_embedding, task_embedding):
        shared_embedding -= torch.mean(shared_embedding, 0)
        task_embedding -= torch.mean(task_embedding, 0)

        # p=2时是l2正则
        shared_embedding = nn.functional.normalize(shared_embedding, dim=1, p=2)
        task_embedding = nn.functional.normalize(task_embedding, dim=1, p=2)

        correlation_matrix = task_embedding.t() @ shared_embedding
        loss_diff = torch.mean(torch.square_(correlation_matrix)) * 0.01
        loss_diff = torch.where(loss_diff > 0, loss_diff, 0)
        return loss_diff

    def forward(self, graph_list, relation_graph, examples):
        # print('执行过程')
        user_cas_embedding = self.dycasHGNN(graph_list, self.device)
        user_social_embedding = self.relationGNN(relation_graph)
        sender_social_embedding = self.relationLSTM(examples, user_social_embedding)
        sender_cas_embedding = self.cascadeLSTM(examples, user_cas_embedding)  # H^cas     (batch_size, 200, emb_dim)
        sender_cas_embedding_share = self.lookup_embedding(examples, user_cas_embedding)
        sender_social_embedding_share = self.lookup_embedding(examples, user_social_embedding)
        shared_embedding, _ = self.sharedLSTM(sender_cas_embedding_share, sender_social_embedding_share)
        example_len = torch.count_nonzero(examples, 1)  # 统计每个观察到的级联的长度，去掉用户0   (batch_size, 1)
        batch_size, seq_len, emb_dim = shared_embedding.size()
        H_user = []
        H_cas = []
        H_share = []
        for i in range(batch_size):
            H_user.append(sender_social_embedding[i, example_len[i] - 1, :])
            H_cas.append(sender_cas_embedding[i, example_len[i] - 1, :])
            H_share.append(shared_embedding[i, example_len[i]-1, :])
        H_user = torch.stack(H_user, dim=0) # (batch_size, emb_dim)
        H_cas = torch.stack(H_cas, dim=0)   # (batch_size, emb_dim)
        H_share = torch.stack(H_share, dim=0)   # (batch_size, emb_dim)
        pred_micro = self.micro_mlp(torch.concat((sender_social_embedding, shared_embedding), dim=2))   # (batch_size, 200, emb_dim)
        pred_macro = self.macro_mlp(torch.concat((H_cas, H_share), dim=1))  # 每条级联的最终长度
        loss_adv, _ = self.adversarial_loss(H_share)
        loss_diff_micro = self.diff_loss(H_share, H_user)
        loss_diff_macro = self.diff_loss(H_share, H_cas)
        loss_diff = loss_diff_micro + loss_diff_macro
        return pred_micro, pred_macro, loss_adv.item(), loss_diff.item()