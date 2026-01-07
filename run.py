import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from HypergraphUtil import *
from Metrics import *
from Module import RLCascadeModel
from DataSet import *

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_name', default='memetracker')
parser.add_argument('-epoch', default=50, type=int)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-emb_dim', default=64, type=int)
parser.add_argument('-hidden_dim', default=64, type=int)
parser.add_argument('-train_rate', default=0.8, type=float)
parser.add_argument('-valid_rate', default=0.1, type=float)
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-rl_start_epoch', default=15, type=int, help='Epoch to start RL training')
parser.add_argument('-rollout_steps', default=5, type=int, help='Steps to rollout in RL')
parser.add_argument('-lambda_loss', default=0.3, type=float)

opt = parser.parse_args()

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

    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq


def MSLE(y, y_predicted):
    '''计算宏观预测的 MSLE'''
    predicted = y_predicted.cpu().detach().numpy().squeeze()
    # 防止 log(<=0)
    predicted[predicted < 1] = 1
    label = y.cpu().detach().numpy()
    msle = np.square(np.log2(predicted) - np.log2(label))
    return np.mean(msle)


def get_performance(crit, pred, gold):
    '''计算 Loss 和 Accuracy (原版逻辑)'''
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def get_sparse_adj_from_dhg(relation_graph, device):
    """提取稀疏邻接矩阵"""
    try:
        adj = relation_graph.A
        if not torch.is_tensor(adj):
            coo = adj.tocoo()
            indices = np.vstack((coo.row, coo.col))
            values = coo.data
            indices = torch.LongTensor(indices).to(device)
            values = torch.FloatTensor(values).to(device)
            return indices, values

        if adj.is_sparse:
            indices = adj._indices().to(device)
            values = adj._values().to(device)
            return indices, values
    except:
        edges = relation_graph.e
        num_v = relation_graph.num_v
        self_loop = torch.arange(num_v, device=device).unsqueeze(0).repeat(2, 1)

        if isinstance(edges, tuple) or isinstance(edges, list):
            src, dst = edges
            edges = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0).to(device)
        elif torch.is_tensor(edges):
            edges = edges.to(device)

        edges = torch.cat([edges, self_loop], dim=1)
        values = torch.ones(edges.shape[1]).to(device)
        return edges, values
    return None, None


# ==========================================
# 训练函数
# ==========================================

def train_supervised_epoch(model, train_loader, optimizer, device, micro_loss_func):
    model.train()
    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    for i, batch in enumerate(train_loader):
        tgt, _, _, tgt_len = (item.to(device) for item in batch)
        gold = tgt[:, 1:]
        input_seq = tgt[:, :-1]

        # Forward
        # NewModule 返回的是 (logits, pred_size)
        micro_logits, pred_log_len = model(input_seq)

        # 1. Micro Loss (Masking is Crucial)
        mask = get_previous_user_mask(input_seq.cpu(), model.user_size).to(device)
        masked_logits = micro_logits + mask

        loss_micro, n_correct = get_performance(
            micro_loss_func,
            masked_logits.view(-1, model.user_size),
            gold
        )

        # 2. Macro Loss (Size Prediction)
        # 只取最后一个时刻的预测进行监督
        pred_final_size = pred_log_len[:, -1, 0]
        true_log_len = torch.log1p(tgt_len.float())
        loss_macro = F.mse_loss(pred_final_size, true_log_len)

        # Total Loss
        loss = loss_micro + 1.0 * loss_macro  # 调整权重，让 Macro 更受重视

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n_words = gold.data.ne(Constants.PAD).sum().float()
        total_loss += loss.item()
        n_total_correct += n_correct
        n_total_words += n_words

    return total_loss / len(train_loader), n_total_correct / n_total_words


def train_rollout_epoch(model, train_loader, optimizer, device, rollout_steps=5):
    model.train()
    total_loss = 0.0
    micro_loss_func = nn.CrossEntropyLoss(ignore_index=Constants.PAD)

    for i, batch in enumerate(train_loader):
        tgt, _, _, tgt_len = (item.to(device) for item in batch)
        batch_size = tgt.size(0)

        # ==========================
        # 1. 基础监督信号 (Base Constraint)
        # ==========================
        input_seq = tgt[:, :-1]
        target_seq = tgt[:, 1:]
        micro_logits_sl, pred_size_sl = model(input_seq)

        loss_micro = micro_loss_func(micro_logits_sl.reshape(-1, model.user_size), target_seq.reshape(-1))
        loss_macro = F.mse_loss(pred_size_sl[:, -1, 0], torch.log1p(tgt_len.float()))
        loss_supervised = loss_micro + loss_macro

        # ==========================
        # 2. RL Rollout
        # ==========================
        # A. Burn-in
        context_len = max(1, tgt.size(1) // 2)
        context_seq = tgt[:, :context_len]

        # 获取上下文的初始 hidden state
        # 我们需要重新跑一遍 GRU 得到 hidden state
        social_embs = model.get_social_graph_emb()
        ctx_emb = F.embedding(context_seq, social_embs)

        # 添加位置编码 (Context部分)
        positions = torch.arange(context_len, device=device).unsqueeze(0).expand(batch_size, -1)
        ctx_input = ctx_emb + model.pos_embedding(positions)

        _, curr_hidden = model.cascade_rnn(ctx_input)

        curr_input = context_seq[:, -1].unsqueeze(1)  # [B, 1]

        # 维护一个 Mask 记录历史已激活用户
        # 初始化为 Context 中出现过的用户
        current_mask = torch.zeros(batch_size, model.user_size).to(device)
        for b in range(batch_size):
            current_mask[b, context_seq[b]] = float('-inf')

        log_probs = []
        values = []
        rewards = []
        entropy = 0

        # B. Interaction Loop
        for step in range(rollout_steps):
            abs_step = context_len + step

            # Forward Step (注意：NewModule 返回 micro, value, size, hidden)
            micro_logits, val_pred, size_pred, next_hidden = model.forward_step(curr_input, curr_hidden, abs_step)

            # --- 关键修复：加 Mask ---
            masked_logits = micro_logits + current_mask  # 加上 -inf
            dist = distributions.Categorical(logits=masked_logits)
            action = dist.sample()  # 采样下一个用户

            # 记录 Log Prob 和 Value
            log_probs.append(dist.log_prob(action))
            values.append(val_pred.squeeze())
            entropy += dist.entropy().mean()

            # 更新状态
            curr_input = action.unsqueeze(1)
            curr_hidden = next_hidden

            # 更新 Mask (将新采样的用户也 mask 掉，防止死循环重复)
            # 使用 scatter_ 填充 -inf
            action_idx = action.unsqueeze(1)  # [B, 1]
            current_mask.scatter_(1, action_idx, float('-inf'))

            # --- 计算 Reward ---
            r_step = torch.zeros(batch_size).to(device)
            future_gt = tgt[:, context_len:]  # 真实的未来序列

            for b in range(batch_size):
                if action[b] in future_gt[b]:
                    r_step[b] = 1.0  # Hit Reward
                else:
                    r_step[b] = -0.05  # 稍微降低惩罚，鼓励探索
            rewards.append(r_step)

        # C. Terminal Reward (基于 Macro 预测的准确度)
        # 使用 rollout 最后一步的 Size 预测与真实长度对比
        final_size_pred = size_pred.squeeze()
        true_log_len = torch.log1p(tgt_len.float())

        # 奖励：误差越小，奖励越大 (使用 RBF 核转换或倒数)
        r_terminal = 1.0 / (1.0 + torch.abs(final_size_pred - true_log_len))

        # 将 Terminal Reward 加到最后一步
        rewards[-1] += r_terminal

        # D. Actor-Critic Loss
        policy_loss = 0
        value_loss = 0
        R = values[-1].detach()  # Bootstrap
        gamma = 0.95

        for i in reversed(range(rollout_steps)):
            R = rewards[i] + gamma * R
            advantage = R - values[i].detach()

            policy_loss += -(log_probs[i] * advantage).mean()
            value_loss += F.mse_loss(values[i], R)

        loss_rl = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # ==========================
        # 3. Combine & Update
        # ==========================
        total_loss_batch = loss_supervised + 0.05 * loss_rl

        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += total_loss_batch.item()

    return total_loss / len(train_loader), 0.0


def test_epoch(model, data_loader, user_size, device, k_list=[10, 50, 100]):
    model.eval()

    # 初始化指标容器
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
    msle_list = []

    n_total_words = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            tgt, _, _, tgt_len = (item.to(device) for item in batch)
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

            input_seq = tgt[:, :-1]

            # Forward (注意：NewModule 返回的是 micro_logits 和 pred_size)
            micro_logits, pred_size = model(input_seq)

            # --- Micro Metrics (不变) ---
            mask = get_previous_user_mask(input_seq.cpu(), user_size).to(device)
            y_pred = (micro_logits + mask).view(-1, micro_logits.size(-1))
            y_pred = y_pred.detach().cpu().numpy()

            scores_batch, scores_len = compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len

            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len
            # 1. 获取模型输出 (这是对数尺度的预测值)
            pred_macro_log = pred_size[:, -1, 0]

            pred_macro_raw = torch.expm1(pred_macro_log)
            pred_macro_raw = torch.relu(pred_macro_raw)

            # 4. 调用原版 MSLE (此时传入的是 Raw Count，与 Baseline 保持一致)
            msle_val = MSLE(tgt_len, pred_macro_raw)
            msle_list.append(msle_val)

    # Normalize
    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    macro_metric = {'MSLE': np.mean(msle_list)}
    return scores, macro_metric


# ==========================================
# Main Loop (恢复原版打印格式)
# ==========================================

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    user_size, total_cascades, timestamps, train, valid, test = SplitData(
        opt.dataset_name, opt.train_rate, opt.valid_rate, load_dict=True
    )
    train_loader = DataLoader(train, opt.batch_size, load_dict=True, cuda=False)
    valid_loader = DataLoader(valid, opt.batch_size, load_dict=True, cuda=False)
    test_loader = DataLoader(test, opt.batch_size, load_dict=True, cuda=False)

    # 3. Model
    relation_graph = RelationGraph(opt.dataset_name, device)
    adj_indices, adj_values = get_sparse_adj_from_dhg(relation_graph, device)

    model = RLCascadeModel(
        user_size=user_size,
        embed_dim=opt.emb_dim,
        hidden_dim=opt.hidden_dim,
        adj_indices=adj_indices,
        adj_values=adj_values,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    micro_loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=Constants.PAD)  # strict match

    history = {
        'train_loss': [],
        'test_hits10': [], 'test_hits50': [], 'test_hits100': [],
        'test_map10': [], 'test_map50': [], 'test_map100': [],
        'test_msle': []
    }

    # 4. Metrics Tracking
    k_list = [10, 50, 100]
    micro_score = float('-inf')
    macro_score = float('inf')
    micro_best_epoch = 0
    macro_best_epoch = 0
    micro_score_metrics = None
    macro_score_metrics = None

    print(f'================ parameter detail ==================')
    print(f'Parameters: {opt}')
    print(f'====================================================')

    total_time = 0

    for epoch_i in range(opt.epoch):
        print(f'======================== Epoch {epoch_i + 1} ========================')

        start = time.time()

        # --- Train ---
        if epoch_i < opt.rl_start_epoch:
            loss, _ = train_supervised_epoch(model, train_loader, optimizer, device, micro_loss_func)
        else:
            loss, _ = train_rollout_epoch(model, train_loader, optimizer, device, rollout_steps=opt.rollout_steps)

        end = time.time()
        print('===== Train')
        print(f'Mean Prediction loss at epoch{epoch_i + 1}: {loss}')
        print(f'Train time at epoch{epoch_i + 1}: {end - start} second')
        total_time += end - start

        # --- Valid ---
        scores, macro_metric = test_epoch(model, valid_loader, user_size, device, k_list)
        print('===== Valid')
        print(f'Micro prediction result: {scores}')
        print(f'Macro prediction result: {macro_metric}')

        # --- Test ---
        scores, macro_metric = test_epoch(model, test_loader, user_size, device, k_list)
        print('===== Test')
        print(f'Micro prediction result: {scores}')
        print(f'Macro prediction result: {macro_metric}')

        # --- Save Best ---
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