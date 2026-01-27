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
parser.add_argument('-dataset_name', default='douban')
parser.add_argument('-epoch', default=100)
parser.add_argument('-batch_size', default=64)
parser.add_argument('-emb_dim', default=64)
parser.add_argument('-train_rate', default=0.8)
parser.add_argument('-valid_rate', default=0.1)
parser.add_argument('-lambda_loss', default=0.3)  # 微观宏观任务平衡参数，超参数
parser.add_argument('-gamma_loss', default=0.05)  # 正交性约束平衡参数，超参数
parser.add_argument('-max_seq_length', default=200)
parser.add_argument('-step_split', default=8)  # 级联超图的个数
parser.add_argument('-lr', default=0.001)  # 学习率
parser.add_argument('-lr_rl', default=0.00005)
parser.add_argument('-early_stop_step', default=10)
parser.add_argument('-sl_epochs', default=60)
parser.add_argument('-rollout_steps', default=10)
parser.add_argument('-eta', default=0.1)
parser.add_argument('-alpha_macro', default=0.5)
parser.add_argument('-gamma', default=0.99)

opt = parser.parse_args()


# --- RL Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_micro, action, reward, next_state_micro, current_seq, done, behavior_logp,
             pred_macro, pred_macro_next):
        # 存 CPU Tensor 以节省显存
        self.buffer.append((state_micro, action, reward, next_state_micro, current_seq, done,
                            behavior_logp, pred_macro, pred_macro_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        (state, action, reward, next_state, cur_seq, done
         , behavior_logp, pred_macro, pred_macro_next) = zip(*batch)

        # Pad cur_seq (因为 rollout 时长度不一)
        max_len = max([s.size(0) for s in cur_seq])
        padded_seqs = []
        for s in cur_seq:
            pad = torch.zeros(max_len - s.size(0), dtype=torch.long)
            padded_seqs.append(torch.cat([s, pad]))

        return (torch.stack(state), torch.tensor(action, dtype=torch.long), torch.tensor(reward, dtype=torch.float),
                torch.stack(next_state), torch.stack(padded_seqs), torch.tensor(done, dtype=torch.float),
                torch.tensor(behavior_logp, dtype=torch.float),
                torch.tensor(pred_macro, dtype=torch.float),
                torch.tensor(pred_macro_next, dtype=torch.float))

    def __len__(self):
        return len(self.buffer)


def compute_reward_macro_guided(action, gt_set, pred_macro_curr, pred_macro_next, gt_size, is_terminal,
                                action_emb=None, gt_avg_emb=None, current_eta=opt.eta):
    """
    重构后的奖励函数：实现宏观引导微观的协同优化。

    逻辑：
    1. 计算基础微观奖励 (Base Micro Reward)，包含命中奖励和 Embedding 软对齐奖励。
    2. 计算宏观对齐因子 (Align Factor)，基于当前预测规模与真实规模的 MSLE 距离。
    3. 耦合：Reward = Base_Micro_Reward * Align_Factor。
    4. 附加基于势能的奖励塑造 (Reward Shaping) 和终端惩罚。
    """
    is_hit = action in gt_set

    # --- Step 1: 计算基础微观奖励 (Base Micro Reward) ---
    base_reward = 5.0 if is_hit else -0.5
    msle_dist = abs(math.log2(max(float(pred_macro_next), 1.0)) - math.log2(max(gt_size, 1.0)))
    align_factor = math.exp(-msle_dist * 0.5)

    shaping_discount = 0.1
    reward = base_reward * align_factor

    # --- Step 4: 势能奖励塑造 (Potential-based Shaping) ---
    gt = max(gt_size, 1.0)

    # 定义势能函数 $\Phi(s) = - (\ln(1+p) - \ln(1+gt))^2$
    def Phi(pred):
        p = float(pred)
        return - pow(math.log1p(p) - math.log1p(gt), 2)

    phi_curr = Phi(pred_macro_curr)
    phi_next = Phi(pred_macro_next)

    # 引导奖励：鼓励 pred_macro 向真实规模靠拢
    shaping = current_eta * 0.1 * (opt.gamma * phi_next - phi_curr)
    shaping = torch.clamp(torch.tensor(shaping), -0.1, 0.1).item()

    reward += shaping * shaping_discount

    # --- Step 5: 终端宏观惩罚 ---
    if is_terminal:
        pred = max(pred_macro_next, 1.0)
        msle_val = pow(math.log2(pred) - math.log2(max(gt_size, 1.0)), 2)
        # 终端宏观惩罚，加强对最终规模的约束
        reward -= opt.alpha_macro * 0.05 * msle_val

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


# --- run.py 修改 ---
def compute_reward_minimal(action, gt_set, pred_macro_next, gt_size, is_terminal):
    if action not in gt_set:
        return -0.1

    gt = max(gt_size, 1.0)
    msle = pow(math.log2(max(float(pred_macro_next), 1.0)) - math.log2(gt), 2)

    # 核心修改：提高基础分到 5.0，并减弱 msle 的压制程度 (使用 0.2 缩放)
    align_factor = math.exp(-msle * 0.2)
    reward = 5.0 * align_factor

    if is_terminal: reward -= 0.5 * msle  # 加大对最终规模不准的惩罚
    return reward


def train_rl_step_minmal(model, train_loader, hypergraph_list, relation_graph, optimizer, buffer, device, current_eta):
    model.train()
    total_pi_loss, update_count = 0, 0
    total_phi, total_terminal_msle, step_count, terminal_count = 0.0, 0.0, 0, 0
    entropy_weight = 0.02

    for batch in train_loader:
        tgt, _, _, tgt_len = (item.to(device) for item in batch)
        bs = tgt.size(0)
        start_len = random.randint(2, 5)
        init_seq = tgt[:, :start_len]
        gt_sets = [set([u for u in t.cpu().numpy() if u != 0]) for t in tgt]

        # 1. Greedy Rollout (Baseline)
        with torch.no_grad():
            g_seq = init_seq.clone()
            g_reward = torch.zeros(bs, device=device)
            for t in range(opt.rollout_steps):
                logits, pred, _, _ = model.forward_step(hypergraph_list, relation_graph, g_seq)
                actions = logits.argmax(dim=-1)
                next_seq = torch.cat([g_seq, actions.unsqueeze(1)], dim=1)
                _, pred_next, _, _ = model.forward_step(hypergraph_list, relation_graph, next_seq)
                for i in range(bs):
                    g_reward[i] += compute_reward_minimal(actions[i].item(), gt_sets[i], pred_next[i].item(),
                                                          tgt_len[i].item(), t == opt.rollout_steps - 1)
                g_seq = next_seq
            baseline = g_reward / opt.rollout_steps

        # 2. Sampled Path (探索)
        s_seq = init_seq.clone()
        log_probs = []
        entropies = []
        s_reward = torch.zeros(bs, device=device)
        for t in range(opt.rollout_steps):
            logits, pred_curr, _, _ = model.forward_step(hypergraph_list, relation_graph, s_seq)
            mask = get_previous_user_mask(s_seq, model.user_size).to(device)
            dist = Categorical(logits=logits + mask[:, -1, :])
            actions = dist.sample()
            log_probs.append(dist.log_prob(actions))
            entropies.append(dist.entropy())

            next_seq = torch.cat([s_seq, actions.unsqueeze(1)], dim=1)
            with torch.no_grad():
                _, pred_next, _, _ = model.forward_step(hypergraph_list, relation_graph, next_seq)

            for i in range(bs):
                r = compute_reward_minimal(actions[i].item(), gt_sets[i], pred_next[i].item(), tgt_len[i].item(),
                                           t == opt.rollout_steps - 1)
                s_reward[i] += r

                # 统计宏观监控指标
                gt_i = max(float(tgt_len[i].item()), 1.0)
                total_phi += - abs(math.log1p(float(pred_curr[i].item())) - math.log1p(gt_i)) / math.log1p(gt_i)
                step_count += 1
                if t == opt.rollout_steps - 1:
                    total_terminal_msle += abs(math.log2(max(float(pred_next[i].item()), 1.0)) - math.log2(gt_i))
                    terminal_count += 1
            s_seq = next_seq

        # 3. SCST Advantage & Update
        # Advantage = (采样路径平均奖励 - 贪婪路径平均奖励) * 放大信号
        advantage = (s_reward / opt.rollout_steps - baseline).detach()

        # 诊断打印：如果这个值一直是 0，说明模型还在冷启动，需要增加 SL Epochs
        if random.random() < 0.01:
            print(f"      [Debug] Advantage Mean: {advantage.mean().item():.4f}, Std: {advantage.std().item():.4f}")

        if advantage.std() > 1e-8:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # 纯策略梯度 Loss (叠加 0.05 的熵奖励强制探索)
        pi_loss_per_batch = -(torch.stack(log_probs).sum(dim=0) * advantage)
        entropy_loss_per_batch = - (torch.stack(entropies).sum(dim=0))  # 负熵用于最大化

        pi_loss = (pi_loss_per_batch + entropy_weight * entropy_loss_per_batch).mean()

        optimizer.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_pi_loss += pi_loss.item()
        update_count += 1

    avg_phi = total_phi / (step_count + 1e-9)
    avg_msle = total_terminal_msle / (terminal_count + 1e-9)
    return 0.0, total_pi_loss / update_count, avg_phi, avg_msle

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


# def get_previous_user_mask(seq, user_size):
#     ''' Mask previous activated users.'''
#     assert seq.dim() == 2
#     prev_shape = (seq.size(0), seq.size(1), seq.size(1))
#     seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
#     previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
#     previous_mask = torch.from_numpy(previous_mask).to(seq.device)
#     if seq.is_cuda:
#         previous_mask = previous_mask.cuda()
#     masked_seq = previous_mask * seqs.float()
#
#     # force the 0th dimension (PAD) to be masked
#     PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1).to(seq.device)
#     # if seq.is_cuda:
#     #     PAD_tmp = PAD_tmp.cuda()
#     masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
#     ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size).to(seq.device)
#     # if seq.is_cuda:
#     #     ans_tmp = ans_tmp.cuda()
#     masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
#     # print("masked_seq ",masked_seq.size())
#     return masked_seq
def get_previous_user_mask(seq, user_size):
    device = seq.device
    batch_size, seq_len = seq.size()
    mask = torch.zeros(batch_size, seq_len, user_size, device=device)
    for t in range(seq_len):
        # 提取当前时刻及之前出现过的用户
        prefix_seq = seq[:, :t + 1]  # [B, t+1]
        # 在第 t 个时刻的 mask 上，将 prefix_seq 包含的 ID 位置设为 -inf
        mask[:, t, :].scatter_(1, prefix_seq.long(), float('-inf'))
    mask[:, :, 0] = float('-inf')

    return mask


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
        current_eta = opt.eta
        if current_epoch_idx < int(opt.sl_epochs) + 5:
            current_eta = 0.0
        q_loss, pi_loss, phi_mean, terminal_msle = train_rl_step_minmal(model, train_loader, hypergraph_list, relation_graph, optimizer, buffer, device, current_eta)

        # RL 阶段 loss 含义变化，返回 total loss 供打印
        total_rl_loss = q_loss + pi_loss
        print(f"   [RL Phase] Epoch {current_epoch_idx + 1} | Q-Loss: {q_loss:.4f} | Pi-Loss: {pi_loss:.4f} Phi Mean: {phi_mean:.4f} Avg Terminal MSLE: {terminal_msle:.4f}")

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
    buffer = ReplayBuffer(capacity=5000)


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
