import argparse
import operator

import numpy as np
import torch

import torch.optim as optim
import time
from HypergraphUtil import *
from Metrics import *
from Module import *
from DataSet import *
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_name', default='android')
parser.add_argument('-epoch', default=50)
parser.add_argument('-batch_size', default=64)
parser.add_argument('-emb_dim', default=64)
parser.add_argument('-train_rate', default=0.8)
parser.add_argument('-valid_rate', default=0.1)
parser.add_argument('-lambda_loss', default=0.3)  # 微观宏观任务平衡参数，超参数
parser.add_argument('-gamma_loss', default=0.05)  # 正交性约束平衡参数，超参数
parser.add_argument('-max_seq_length', default=200)
parser.add_argument('-step_split', default=8)  # 级联超图的个数
parser.add_argument('-lr', default=0.001)  # 学习率
parser.add_argument('-early_stop_step', default=10)  #

opt = parser.parse_args()


def MAE(y, y_predicted):
    y_predicted = y_predicted.squeeze()
    mae = torch.abs(y_predicted - y)
    # sum_sq_error = torch.sum(sq_error)
    # mse = sum_sq_error / label.size()
    mae = torch.mean(mae)
    return mae

def MSLE(y, y_predicted):
    '''
    :param y: 真实标签  tensor
    :param y_predicted: 预测值 tensor
    :return:
    '''
    predicted = y_predicted.cpu().detach().numpy()
    predicted = predicted.squeeze()
    predicted[predicted < 1] = 1
    label = y.cpu().detach().numpy()
    msle = np.square(np.log2(predicted) - np.log2(label))
    msle = np.mean(msle)
    return msle


def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    # if seq.is_cuda:
    #     PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    # if seq.is_cuda:
    #     ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    # print("masked_seq ",masked_seq.size())
    return masked_seq


def get_performance(crit, pred, gold):
    '''
    crit：损失函数，CrossEntropy
    pred：batch_size * cas_len(199) * user_size（用户个数）    表示每个用户在每个时刻(t>=2)参与级联的概率
    gold：batch_size * cas_len(199)                          表示每个时刻参与级联的用户是谁
    '''
    loss = crit(pred, gold.contiguous().view(-1))
    # torch.max(input, dim, keepdim=False) --> Tensor
    ## input：输入的Tensor
    ## dim：要压缩的维度
    ## keepdim：输出的Tensor是否保留维度
    pred = pred.max(1)[1]
    # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
    # tensor.view(-1) 转换维度为1维       tensor.view(*shape)  构建一个数据相同，但形状(形状为shape)不同的“视图”
    # data.contiguous().view(-1)    contiguous()保证一个tensor是连续的，才能被view()处理
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def train_epoch(model, train_loader, relation_graph, hypergraph_list, micro_loss_func, optimizer, lambda_loss,
                gamma_loss, user_size, device):
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_num = 0.0

    for i, batch in enumerate(train_loader):
        tgt, tgt_timestamp, tgt_idx, tgt_len = (item.to(device) for item in batch)
        gold = tgt[:, 1:]

        n_words = gold.data.ne(Constants.PAD).sum().float()  # n_words是参与一个batch(64)中的级联的用户总数（用户会重复出现）,真正的正样本个数
        n_total_words += n_words
        batch_num += tgt.size(0)
        pred_micro, pred_macro, loss_adv, loss_diff = model(hypergraph_list, relation_graph, tgt)
        mask = get_previous_user_mask(tgt[:, :-1].cpu(), user_size).to(device)
        micro_loss, n_correct = get_performance(micro_loss_func,
                                                (pred_micro[:, :-1, :] + mask).view(-1, pred_micro.size(-1)), gold)
        macro_loss = MAE(tgt_len, pred_macro)
        loss = (1 - lambda_loss) * micro_loss + lambda_loss * macro_loss
        loss += loss_adv
        loss += gamma_loss * loss_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_total_correct += n_correct
        total_loss += loss.item()

    return total_loss / n_total_words, n_total_correct / n_total_words


def test_epoch(model, data_loader, relation_graph, hypergraph_list, user_size, device,
               k_list=[10, 50, 100]):
    model.eval()

    macro_metric = {}
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
    msle = []

    n_total_words = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            tgt, tgt_timestamp, tgt_idx, tgt_len = (item.to(device) for item in batch)
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

            pred_micro, pred_macro, loss_adv, loss_diff = model(hypergraph_list, relation_graph, tgt)
            mask = get_previous_user_mask(tgt[:, :-1].cpu(), user_size).to(device)
            y_pred = (pred_micro[:, :-1, :] + mask).view(-1, pred_micro.size(-1))
            y_pred = y_pred.detach().cpu().numpy()
            scores_batch, scores_len = compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

            msle.append(MSLE(tgt_len, pred_macro))

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    macro_metric['MSLE'] = np.mean(msle)

    return scores, macro_metric


def main():
    # =============== 读取参数 ===============
    dataset = opt.dataset_name
    max_seq_length = opt.max_seq_length  # 级联序列最大长度
    batch_size = opt.batch_size
    emb_dim = opt.emb_dim
    step_split = opt.step_split
    lambda_loss = opt.lambda_loss
    gamma_loss = opt.gamma_loss
    early_stop_step = opt.early_stop_step
    patience = early_stop_step
    lr = opt.lr
    epoch = opt.epoch
    train_rate = opt.train_rate
    valid_rate = opt.valid_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ========================================

    # =============== 读取数据集 ===============
    user_size, total_cascades, timestamps, train, valid, test = SplitData(dataset, train_rate, valid_rate,
                                                                          load_dict=True)
    train_loader = DataLoader(train, batch_size, load_dict=True, cuda=False)
    valid_loader = DataLoader(valid, batch_size, load_dict=True, cuda=False)
    test_loader = DataLoader(test, batch_size, load_dict=True, cuda=False)
    # =======================================

    # =============== 准备模型 ===============
    relation_graph = RelationGraph(dataset, device)
    hypergraph_list = DynamicCasHypergraph(total_cascades, timestamps, user_size, device, step_split)
    model = Module(user_size, emb_dim, step_split, max_seq_length, 2, device).to(device)


    micro_loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    # =======================================

    # =============== 准备优化器 ===============
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # =======================================

    k_list = [10, 50, 100]
    macro_score_metrics = None  # 宏观预测分数
    micro_score_metrics = None  # 微观预测分数
    micro_score = float('-inf')  # MAP@100 分数
    macro_score = float('inf')  # MSLE 分数
    micro_best_epoch = 0
    macro_best_epoch = 0

    print(f'================ parameter detail ==================')
    print(f'Parameters: {opt}')
    print(f'====================================================')

    total_time = 0  # 训练用时
    for epoch_i in range(epoch):

        print(f'======================== Epoch {epoch_i + 1} ========================')

        # 开始训练
        start = time.time()
        loss, train_micro_accu = train_epoch(model, train_loader, relation_graph, hypergraph_list, micro_loss_func,
                                             optimizer, lambda_loss, gamma_loss, user_size, device)
        end = time.time()
        print('===== Train')
        print(f'Mean Prediction loss at epoch{epoch_i + 1}: {loss}')
        print(f'Train time at epoch{epoch_i + 1}: {end - start} second')
        total_time += end - start

        # 开始验证
        scores, macro_metric = test_epoch(model, valid_loader, relation_graph, hypergraph_list, user_size, device,
                                          k_list)
        print('===== Valid')
        print(f'Micro prediction result: {scores}')
        print(f'Macro prediction result: {macro_metric}')

        # 开始测试
        scores, macro_metric = test_epoch(model, test_loader, relation_graph, hypergraph_list, user_size, device,
                                          k_list)
        print('===== Test')
        print(f'Micro prediction result: {scores}')
        print(f'Macro prediction result: {macro_metric}')

        if scores['map@100'] > micro_score:
            micro_score_metrics = scores
            micro_score = scores['map@100']
            micro_best_epoch = epoch_i + 1

        if macro_metric['MSLE'] < macro_score:
            macro_score_metrics = macro_metric
            macro_score = macro_metric['MSLE']
            macro_best_epoch = epoch_i + 1

    print('=============== best_result ===============')
    print(f'Micro prediction epoch: {micro_best_epoch}')
    print(f'Micro result:\n{micro_score_metrics}')
    print(f'Macro prediction epoch: {macro_best_epoch}')
    print(f'Macro result:\n{macro_score_metrics}')
    print(f'Total train time: {total_time}')


if __name__ == '__main__':
    main()
