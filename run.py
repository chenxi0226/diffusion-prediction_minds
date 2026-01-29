import argparse
import operator
import numpy as np
import torch
import random
from collections import deque
import torch.optim as optim
import time
import math
import sys

# 引入自定义模块
from HypergraphUtil import *
from Metrics import *
from Module import *
from DataSet import *
import Constants

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_name', default='memetracker')
parser.add_argument('-epoch', default=100)
parser.add_argument('-batch_size', default=64)
parser.add_argument('-emb_dim', default=64)
parser.add_argument('-train_rate', default=0.8)
parser.add_argument('-valid_rate', default=0.1)
parser.add_argument('-max_seq_length', default=200)
parser.add_argument('-step_split', default=8)  # 级联超图的个数
parser.add_argument('-lr', default=0.001)  # SL 学习率
parser.add_argument('-lr_rl', default=0.00005)  # RL 学习率 (通常比SL小)
parser.add_argument('-sl_epochs', default=1)  # SL 预训练轮数
parser.add_argument('-rollout_steps', default=10)  # RL 探索步长
parser.add_argument('-eta', default=0.1)  # 奖励相关系数
parser.add_argument('-gamma', default=0.99)  # RL 折扣因子

opt = parser.parse_args()


# --- RL Replay Buffer (占位，如果后续需要 Off-policy 可用) ---
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, log_prob, reward, entropy):
        self.buffer.append((log_prob, reward, entropy))

    def clear(self):
        self.buffer.clear()


def compute_reward_advanced(action, gt_set, pred_macro_scalar, gt_size, is_terminal):
    """
    Advanced Reward: Macro-Guided Micro Reward
    核心思想：微观的奖励受制于宏观的准确度。
    只有当 Macro 预测比较准时，Micro 的命中才会有高分。
    """
    # 1. Micro Base Reward (命中给分，未命中扣分)
    is_hit = action in gt_set
    r_micro = 2.0 if is_hit else -0.1

    # 2. Macro Guidance Factor (宏观置信度)
    # 计算预测规模与真实规模的对数距离
    gt = max(float(gt_size), 1.0)
    pred = max(float(pred_macro_scalar), 1.0)

    # MSLE 距离 (距离越大，Macro越不准)
    dist = (math.log2(pred) - math.log2(gt)) ** 2

    # 引导因子：距离越小，因子越接近 1；距离越大，因子衰减接近 0。
    # 强迫模型必须先把 Macro 预测准。
    guidance_factor = math.exp(-dist * 0.5)

    reward = r_micro * guidance_factor

    # 3. 终端修正 (Terminal Correction)
    # 在最后一步，额外奖励/惩罚最终的预测规模
    if is_terminal:
        # 如果最终预测规模很准，给予额外奖励
        reward += 1.0 * guidance_factor
        # 如果预测严重偏差，给予惩罚
        reward -= 0.5 * dist

    return reward


def train_sl_step(model, batch, hypergraph_list, relation_graph, optimizer, device, user_size):
    """
    SL 单步训练
    关键点：传入 gt_size 并设置 training_phase='SL' 以启用 Goal Injection
    """
    tgt, _, _, tgt_len = (item.to(device) for item in batch)

    # 模型前向传播
    # gt_size=tgt_len: 告诉模型真实长度，模型内部会随机决定是否用来指导 Transformer
    actor_logits, pred_macro = model(
        hypergraph_list, relation_graph, tgt,
        gt_size=tgt_len,
        training_phase='SL'
    )

    # 1. Micro Loss (Predict Next User)
    # Shift targets: Input [0...T-1], Target [1...T]
    logits = actor_logits[:, :-1, :].reshape(-1, user_size)
    labels = tgt[:, 1:].reshape(-1)

    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=Constants.PAD)
    loss_micro = criterion_ce(logits, labels)

    # 2. Macro Loss (Predict Size)
    # 使用 Log 空间计算 MSE，防止数值过大
    pred_log = torch.log2(pred_macro + 1)
    tgt_log = torch.log2(tgt_len.float() + 1)

    criterion_mse = torch.nn.MSELoss()
    loss_macro = criterion_mse(pred_log, tgt_log)

    # 联合 Loss
    loss = loss_micro + 0.5 * loss_macro  # 简单加权

    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪防止梯度爆炸 (特别是 Transformer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    # 计算准确率供打印
    pred_idx = logits.max(1)[1]
    n_correct = pred_idx.eq(labels).masked_select(labels.ne(Constants.PAD)).sum().float()
    n_total = labels.ne(Constants.PAD).sum().float()

    return loss.item(), n_correct, n_total


def train_rl_step_minmal(model, train_loader, hypergraph_list, relation_graph, optimizer, buffer, device, current_eta):
    """
    RL 训练步骤：基于 Transformer 的 On-Policy Rollout (REINFORCE)
    注意：这里使用了 batch 级更新，没有显式使用 buffer 进行多轮利用，保持策略稳定性。
    """
    model.train()  # 保持 Dropout 开启以增加随机性
    total_loss = 0
    batch_count = 0
    entropy_weight = 0.05  # 熵正则化权重，鼓励探索

    for batch in train_loader:
        tgt, _, _, tgt_len = (item.to(device) for item in batch)
        bs = tgt.size(0)

        # 随机截取一段作为初始状态 (Warm start)
        start_len = random.randint(2, 5)
        start_len = min(start_len, tgt.size(1) - 1)

        init_seq = tgt[:, :start_len]
        gt_sets = [set([u for u in t.cpu().numpy() if u != 0]) for t in tgt]

        # === 1. Sampling (Explore / Rollout) ===
        s_seq = init_seq.clone()
        saved_log_probs = []
        saved_rewards = []
        saved_entropies = []

        # Rollout Loop
        for t in range(opt.rollout_steps):
            # 单步 Forward (RL 模式：Transformer 只能靠自己的 Macro 预测来指导 Micro)
            logits, pred_scalar, _, _ = model.forward_step(hypergraph_list, relation_graph, s_seq)

            # Action Selection
            # 注意：logits 已经包含了 Topological Mask (在 forward_step 里)
            dist = Categorical(logits=logits)
            actions = dist.sample()

            saved_log_probs.append(dist.log_prob(actions))
            saved_entropies.append(dist.entropy())

            # Append action to sequence for next step
            next_seq = torch.cat([s_seq, actions.unsqueeze(1)], dim=1)

            # Get Reward based on Next State
            # 我们需要知道动作做完后，宏观预测变成了什么
            with torch.no_grad():
                _, pred_next_scalar, _, _ = model.forward_step(hypergraph_list, relation_graph, next_seq)

            step_rewards = []
            for i in range(bs):
                # 使用 Advanced Reward Function
                r = compute_reward_advanced(
                    actions[i].item(),
                    gt_sets[i],
                    pred_next_scalar[i].item(),  # 使用动作后的预测值
                    tgt_len[i].item(),
                    is_terminal=(t == opt.rollout_steps - 1)
                )
                step_rewards.append(r)

            saved_rewards.append(torch.tensor(step_rewards, device=device))
            s_seq = next_seq

        # === 2. Policy Update (REINFORCE with Baseline) ===
        # 简单使用 batch 内均值作为 Baseline 减少方差
        R = torch.zeros(bs, device=device)
        policy_loss = []

        # 反向计算累计回报
        for i in reversed(range(opt.rollout_steps)):
            R = opt.gamma * R + saved_rewards[i]
            # Advantage = R - Baseline (Mean of batch)
            advantage = R - R.mean()

            # Loss = -log_prob * advantage
            loss_step = -saved_log_probs[i] * advantage.detach()
            # Entropy Bonus (Maximize entropy -> Minimize -entropy)
            loss_ent = -entropy_weight * saved_entropies[i]

            policy_loss.append(loss_step + loss_ent)

        final_loss = torch.stack(policy_loss).sum() / bs

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += final_loss.item()
        batch_count += 1

    return total_loss / (batch_count + 1e-9), 0.0, 0.0, 0.0  # 保持返回值格式一致


def compute_metric(y_pred, y_gold, k_list):
    """辅助评估函数: Hits@K, MAP@K"""
    rec_list = {}
    for k in k_list:
        rec_list[f'hits@{k}'] = 0
        rec_list[f'map@{k}'] = 0

    batch_len = 0
    for i in range(len(y_pred)):
        label = y_gold[i]
        if label == Constants.PAD: continue  # PAD
        batch_len += 1

        # TopK: argsort 得到从小到大的索引，取最后k个并反转 -> 从大到小
        pred_top_k = y_pred[i].argsort()[-max(k_list):][::-1]

        for k in k_list:
            topk = pred_top_k[:k]
            if label in topk:
                rec_list[f'hits@{k}'] += 1
                rank = np.where(topk == label)[0][0]
                rec_list[f'map@{k}'] += 1.0 / (rank + 1)

    for k in k_list:
        if batch_len > 0:
            rec_list[f'hits@{k}'] /= batch_len
            rec_list[f'map@{k}'] /= batch_len

    return rec_list, batch_len


def MAE(y, y_predicted):
    y_predicted = y_predicted.squeeze()
    mae = torch.abs(y_predicted - y)
    mae = torch.mean(mae)
    return mae


def MSLE(y, y_predicted):
    '''
    :param y: 真实标签  tensor
    :param y_predicted: 预测值 tensor
    :return: MSLE score
    '''
    predicted = y_predicted.cpu().detach().numpy()
    predicted = predicted.squeeze()
    predicted[predicted < 1] = 1  # 防止 log(0)
    label = y.cpu().detach().numpy()
    msle = np.square(np.log2(predicted) - np.log2(label))
    msle = np.mean(msle)
    return msle


def get_previous_user_mask(seq, user_size):
    """
    Mask 掉历史序列中已经出现过的用户
    """
    device = seq.device
    batch_size, seq_len = seq.size()
    mask = torch.zeros(batch_size, seq_len, user_size, device=device)
    for t in range(seq_len):
        # 提取当前时刻及之前出现过的用户
        prefix_seq = seq[:, :t + 1]  # [B, t+1]
        # 在第 t 个时刻的 mask 上，将 prefix_seq 包含的 ID 位置设为 -inf
        mask[:, t, :].scatter_(1, prefix_seq.long(), float('-inf'))
    mask[:, :, 0] = float('-inf')  # Mask PAD/EOS

    return mask


def train_epoch(model, train_loader, relation_graph, hypergraph_list, micro_loss_func,
                optimizer, lambda_loss, gamma_loss, user_size, device,
                current_epoch_idx, buffer):
    # === 策略切换逻辑 ===
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

        # 调用 RL 训练逻辑
        current_eta = opt.eta
        if current_epoch_idx < int(opt.sl_epochs) + 5:
            current_eta = 0.0  # 预热期不使用 shaping reward

        q_loss, pi_loss, phi_mean, terminal_msle = train_rl_step_minmal(model, train_loader, hypergraph_list,
                                                                        relation_graph, optimizer, buffer, device,
                                                                        current_eta)

        # RL 阶段主要关注 Policy Loss
        print(f"   [RL Phase] Epoch {current_epoch_idx + 1} | Policy-Loss: {q_loss:.4f}")

        return q_loss, 0.0


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

            # Forward with TEST phase (GT Size is hidden)
            actor_logits, pred_macro = model(
                hypergraph_list, relation_graph, tgt,
                gt_size=None,
                training_phase='TEST'
            )

            # Masking previous users
            mask = get_previous_user_mask(tgt[:, :-1].cpu(), user_size).to(device)
            # Logits: [B, T-1, User]
            logits_pred = actor_logits[:, :-1, :] + mask

            y_pred = logits_pred.reshape(-1, user_size).cpu().numpy()

            batch_scores, batch_len = compute_metric(y_pred, y_gold, k_list)
            n_total += batch_len
            for k in k_list:
                scores[f'hits@{k}'] += batch_scores[f'hits@{k}'] * batch_len
                scores[f'map@{k}'] += batch_scores[f'map@{k}'] * batch_len

            msle.append(MSLE(tgt_len, pred_macro))

    for k in k_list:
        if n_total > 0:
            scores[f'hits@{k}'] /= n_total
            scores[f'map@{k}'] /= n_total

    return scores, {'MSLE': np.mean(msle)}


def main():
    # =============== 读取参数 ===============
    dataset = opt.dataset_name
    batch_size = opt.batch_size
    emb_dim = opt.emb_dim
    step_split = opt.step_split
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
    print(f"Building Graphs for {dataset}...")
    relation_graph = RelationGraph(dataset, device)
    hypergraph_list = DynamicCasHypergraph(total_cascades, timestamps, user_size, device, step_split)

    print(f"Initializing RL_MINDS_v2 (Macro-Guided)...")
    # === 使用新模型 RL_MINDS_v2 ===
    model = RL_MINDS_v2(
        user_size=user_size,
        embed_dim=opt.emb_dim,
        step_split=opt.step_split,
        max_seq_len=opt.max_seq_length,
        device=device
    ).to(device)

    buffer = ReplayBuffer(capacity=5000)

    # micro_loss_func 在 train_sl_step 内部定义了，这里变量保留但不使用
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
                                             optimizer, opt.lambda_loss if hasattr(opt, 'lambda_loss') else 0.5,
                                             opt.gamma_loss if hasattr(opt, 'gamma_loss') else 0.05,
                                             user_size, device,
                                             current_epoch_idx=epoch_i, buffer=buffer)
        end = time.time()
        print('===== Train')
        print(f'Mean Prediction loss at epoch {epoch_i + 1}: {loss}')
        print(f'Train time at epoch {epoch_i + 1}: {end - start:.2f} second')
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

        # 记录最佳 Micro 结果 (MAP@100)
        if scores['map@100'] > micro_score:
            micro_score_metrics = scores
            micro_score = scores['map@100']
            micro_best_epoch = epoch_i + 1

        # 记录最佳 Macro 结果 (MSLE)
        if macro_metric['MSLE'] < macro_score:
            macro_score_metrics = macro_metric
            macro_score = macro_metric['MSLE']
            macro_best_epoch = epoch_i + 1

    print('=============== best_result ===============')
    print(f'Micro prediction epoch: {micro_best_epoch}')
    print(f'Micro result:\n{micro_score_metrics}')
    print(f'Macro prediction epoch: {macro_best_epoch}')
    print(f'Macro result:\n{macro_score_metrics}')
    print(f'Total train time: {total_time:.2f}s')


if __name__ == '__main__':
    main()