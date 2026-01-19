import argparse
import operator

import numpy as np
import torch
import random
from collections import deque
import torch.optim as optim
import time
from HypergraphUtil import *
from Metrics import *
from Module import *
from DataSet import *
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_name', default='christianity')
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
parser.add_argument('-lr_rl', default=0.00001)
parser.add_argument('-early_stop_step', default=10)
parser.add_argument('-sl_epochs', default=45)
parser.add_argument('-rollout_steps', default=3)

opt = parser.parse_args()


# --- RL Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_micro, action, reward, next_state_micro, current_seq, done):
        # 存 CPU Tensor 以节省显存
        self.buffer.append((state_micro, action, reward, next_state_micro, current_seq, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, cur_seq, done = zip(*batch)

        # Pad cur_seq (因为 rollout 时长度不一)
        max_len = max([s.size(0) for s in cur_seq])
        padded_seqs = []
        for s in cur_seq:
            pad = torch.zeros(max_len - s.size(0), dtype=torch.long)
            padded_seqs.append(torch.cat([s, pad]))

        return (torch.stack(state), torch.tensor(action), torch.tensor(reward),
                torch.stack(next_state), torch.stack(padded_seqs), torch.tensor(done))

    def __len__(self):
        return len(self.buffer)


# --- 辅助函数 ---
def compute_reward(action, gt_set, pred_macro_size, gt_size, is_terminal,
                   action_emb=None, gt_avg_emb=None):

    reward = 0.0

    if action in gt_set:
        reward += 1.0
    else:
        # soft alignment 解决Reward稀疏问题
        if action_emb is not None and gt_avg_emb is not None:
            sim = F.cosine_similarity(action_emb.unsqueeze(0), gt_avg_emb.unsqueeze(0)).item()
            if sim > 0.5:
                reward += 0.1 * sim
            else:
                reward -= 0.05
        else:
            reward -= 0.1

    if is_terminal:
        pred = max(pred_macro_size, 1.0)
        gt = max(gt_size, 1.0)
        # 使用相对误差而不是 Log 误差，对大数更敏感
        error = abs(pred - gt) / (gt + 1.0)
        # 限制惩罚上限，防止梯度爆炸
        penalty = min(error, 2.0)
        reward -= 0.5 * penalty

    return reward


def train_sl_step(model, batch, hypergraph_list, relation_graph, optimizer, device, user_size):
    """SL 单步训练"""
    tgt, _, _, tgt_len = (item.to(device) for item in batch)

    actor_logits, pred_macro = model(hypergraph_list, relation_graph, tgt)

    # Micro Loss (Predict Next)
    # Shift targets: Input [0...T-1], Target [1...T]
    logits = actor_logits[:, :-1, :].reshape(-1, user_size)
    labels = tgt[:, 1:].reshape(-1)

    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss_micro = criterion_ce(logits, labels)

    # Macro Loss
    criterion_mse = torch.nn.MSELoss()
    loss_macro = criterion_mse(torch.log2(pred_macro.squeeze() + 1), torch.log2(tgt_len.float() + 1))

    loss = loss_micro + 0.5 * loss_macro  # 简单加权

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算准确率供打印
    pred_idx = logits.max(1)[1]
    n_correct = pred_idx.eq(labels).masked_select(labels.ne(0)).sum().float()
    n_total = labels.ne(0).sum().float()

    return loss.item(), n_correct, n_total


def train_rl_step(model, train_loader, hypergraph_list, relation_graph, optimizer, buffer, device):
    """RL 整个 Epoch 的训练 (包含 Rollout 和 Update)"""
    # ⚠️ 注意：RL 是一次性跑完整个 Loader 做 Rollout，然后 Update
    # 为了适配外层循环结构，我们在这里把逻辑写完整，外层直接调用

    model.train()
    total_q_loss = 0
    total_pi_loss = 0
    gamma = 0.99

    with torch.no_grad():
        full_graph_emb = model.relationGNN(relation_graph).detach()

    # 1. Rollout
    for batch in train_loader:
        tgt, _, _, tgt_len = (item.to(device) for item in batch)
        bs = tgt.size(0)

        start_len = random.randint(2, min(5, tgt.size(1) - 1)) # sample length
        curr_seq = tgt[:, :start_len]
        gt_avg_embs = []
        gt_sets = []
        for i in range(bs):
            gt_users = tgt[i].cpu().numpy().tolist()
            gt_users = [u for u in gt_users if u != 0]
            gt_sets.append(set(gt_users))

            if len(gt_users) > 0:
                user_idxs = torch.tensor(gt_users).to(device)
                avg_emb = full_graph_emb[user_idxs].mean(dim=0)
            else:
                avg_emb = torch.zeros(opt.emb_dim).to(device)
            gt_avg_embs.append(avg_emb)

        with torch.no_grad():
            for t in range(opt.rollout_steps):
                logits, pred_macro, s_micro, _ = model.forward_step(hypergraph_list, relation_graph, curr_seq)
                actions = Categorical(logits=logits).sample()
                next_seq = torch.cat([curr_seq, actions.unsqueeze(1)], dim=1)
                _, _, s_micro_next, _ = model.forward_step(hypergraph_list, relation_graph, next_seq)

                for i in range(bs):
                    is_terminal = (t == opt.rollout_steps - 1)
                    act_emb = full_graph_emb[actions[i]]
                    r = compute_reward(actions[i].item(), gt_sets[i],
                                       pred_macro[i].item(), tgt_len[i].item(), is_terminal,
                                       action_emb = act_emb, gt_avg_emb = gt_avg_embs[i])
                    buffer.push(s_micro[i].cpu(), actions[i].cpu(), r,
                                s_micro_next[i].cpu(), curr_seq[i].cpu(), is_terminal)
                curr_seq = next_seq

        # 2. Update (Check Buffer Size)
        if len(buffer) < opt.batch_size: continue

        b_s, b_a, b_r, b_s_next, b_cur_seq, b_d = buffer.sample(opt.batch_size)
        b_s, b_a, b_r = b_s.to(device), b_a.to(device), b_r.to(device)
        b_s_next, b_cur_seq = b_s_next.to(device), b_cur_seq.to(device)
        b_d = b_d.float().to(device)

        current_graph_emb = model.relationGNN(relation_graph)

        # Critic Update
        with torch.no_grad():
            logits_next = model.actor_head(b_s_next)
            a_next = Categorical(logits=logits_next).sample()
            target_q = model.get_q_value(b_s_next, a_next, current_graph_emb, b_cur_seq).squeeze()
            target = b_r + gamma * target_q * (1 - b_d)

        current_q = model.get_q_value(b_s, b_a, current_graph_emb, b_cur_seq).squeeze()
        loss_q = F.mse_loss(current_q, target)

        # Actor Update
        logits_pi = model.actor_head(b_s)
        dist_pi = Categorical(logits=logits_pi)
        action_pi = dist_pi.sample()
        log_prob = dist_pi.log_prob(action_pi)
        q_base = model.get_q_value(b_s, action_pi, current_graph_emb, b_cur_seq).squeeze().detach()
        entropy = dist_pi.entropy().mean()
        loss_pi = -(log_prob * q_base).mean() - 0.05 * entropy

        loss = loss_q + loss_pi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_q_loss += loss_q.item()
        total_pi_loss += loss_pi.item()
    buffer.buffer.clear()

    return total_q_loss, total_pi_loss

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


def train_epoch(model, train_loader, relation_graph, hypergraph_list, micro_loss_func,
                optimizer, lambda_loss, gamma_loss, user_size, device,
                current_epoch_idx, buffer):

    if current_epoch_idx < int(opt.sl_epochs):
        # === Phase 1: SL Mode ===
        model.train()
        total_loss = 0
        n_correct_total = 0
        n_words_total = 0

        for batch in train_loader:
            loss, n_correct, n_words = train_sl_step(model, batch, hypergraph_list, relation_graph, optimizer, device,
                                                     user_size)
            total_loss += loss
            n_correct_total += n_correct
            n_words_total += n_words

        avg_loss = total_loss / len(train_loader)
        accu = n_correct_total / (n_words_total + 1e-5)
        print(f"   [SL Phase] Epoch {current_epoch_idx + 1} | Loss: {avg_loss:.4f} | Acc: {accu:.4f}")
        return avg_loss, accu

    else:
        # === Phase 2: RL Mode ===
        # 第一次进入 RL Phase 时调整学习率 (简单的 Trick)
        if current_epoch_idx == int(opt.sl_epochs):
            print(">>> Switching to RL Phase! Reducing LR...")
            for pg in optimizer.param_groups: pg['lr'] = opt.lr_rl

        # 调用 RL 训练逻辑 (该函数内部会遍历整个 loader 做 rollout)
        q_loss, pi_loss = train_rl_step(model, train_loader, hypergraph_list, relation_graph, optimizer, buffer, device)

        # RL 阶段 loss 含义变化，返回 total loss 供打印
        total_rl_loss = q_loss + pi_loss
        print(f"   [RL Phase] Epoch {current_epoch_idx + 1} | Q-Loss: {q_loss:.4f} | Pi-Loss: {pi_loss:.4f}")

        # RL 阶段不强制计算 Micro Accuracy，返回 0 或估算值
        return total_rl_loss, 0.0


def test_epoch(model, data_loader, relation_graph, hypergraph_list, user_size, device, k_list=[10, 50, 100]):
    model.eval()
    scores = {f'hits@{k}': 0 for k in k_list}
    scores.update({f'map@{k}': 0 for k in k_list})
    msle = []
    n_total = 0

    with torch.no_grad():
        for batch in data_loader:
            tgt, _, _, tgt_len = (item.to(device) for item in batch)
            y_gold = tgt[:, 1:].contiguous().view(-1).cpu().numpy()

            pred_micro, pred_macro = model(hypergraph_list, relation_graph, tgt)

            mask = get_previous_user_mask(tgt[:, :-1].cpu(), user_size).to(device)
            y_pred = (pred_micro[:, :-1, :] + mask).view(-1, pred_micro.size(-1)).cpu().numpy()

            batch_scores, batch_len = compute_metric(y_pred, y_gold, k_list)
            n_total += batch_len
            for k in k_list:
                scores[f'hits@{k}'] += batch_scores[f'hits@{k}'] * batch_len
                scores[f'map@{k}'] += batch_scores[f'map@{k}'] * batch_len

            msle.append(MSLE(tgt_len, pred_macro))

    for k in k_list:
        scores[f'hits@{k}'] /= n_total
        scores[f'map@{k}'] /= n_total

    return scores, {'MSLE': np.mean(msle)}


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
                                                                          load_dict=False)
    train_loader = DataLoader(train, batch_size, load_dict=True, cuda=False)
    valid_loader = DataLoader(valid, batch_size, load_dict=True, cuda=False)
    test_loader = DataLoader(test, batch_size, load_dict=True, cuda=False)
    # =======================================

    # =============== 准备模型 ===============
    relation_graph = RelationGraph(dataset, device)
    hypergraph_list = DynamicCasHypergraph(total_cascades, timestamps, user_size, device, step_split)
    model = RL_MINDS_StructureAware(
        user_size=user_size,
        embed_dim=opt.emb_dim,
        step_split=opt.step_split,
        max_seq_len=200,
        device=device
    ).to(device)
    buffer = ReplayBuffer(capacity=200)


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
                                             optimizer, lambda_loss, gamma_loss, user_size, device,
                                             current_epoch_idx=epoch_i, buffer=buffer)
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
