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
parser.add_argument('-dataset_name', default='christianity')
parser.add_argument('-epoch', default=50)
parser.add_argument('-batch_size', default=64)
parser.add_argument('-emb_dim', default=128)
parser.add_argument('-train_rate', default=0.8)
parser.add_argument('-valid_rate', default=0.1)
parser.add_argument('-max_seq_length', default=200)
parser.add_argument('-step_split', default=8)  # 级联超图的个数
parser.add_argument('-lr', default=0.001)  # SL 学习率
parser.add_argument('-lr_rl', default=0.00001)  # RL 学习率 (通常比SL小)
parser.add_argument('-sl_epochs', default=20)  # SL 预训练轮数
parser.add_argument('-rollout_steps', default=15)  # RL 探索步长
parser.add_argument('-eta', default=0.2)  # 奖励相关系数
parser.add_argument('-gamma', default=0.99)  # RL 折扣因子

opt = parser.parse_args()

def compute_reward_delta(action, gt_set, curr_size, goal_scalar, is_terminal, prev_dist):
    """
    Reward = R_micro + eta * (Phi_{t-1} - Phi_t)
    """
    # 1. Micro Reward (不再连坐)
    is_hit = action in gt_set
    r_micro = 2.0 if is_hit else -0.1

    # 2. Macro Progress Reward
    # 目标是常数，curr_size 是当前真实大小
    goal = max(float(goal_scalar), 1.0)
    curr = max(float(curr_size), 1.0)

    # 当前状态距离目标的“势能”
    curr_dist = abs(math.log2(curr) - math.log2(goal))

    r_macro = 0.0
    if prev_dist is not None:
        # 如果距离变小了 (prev > curr)，奖励正分
        r_macro = (prev_dist - curr_dist)

    # 组合
    eta = 0.5  # 调节因子
    reward = r_micro + eta * r_macro

    # 3. Terminal Check (防止 Agent 走太远)
    if is_terminal:
        # 如果最终停下的位置和目标很接近，给大奖
        if curr_dist < 0.5:
            reward += 5.0
        elif curr_dist > 2.0:
            reward -= 2.0

    return reward, curr_dist


def train_sl_step(model, batch, hypergraph_list, relation_graph, optimizer, device, user_size):
    tgt, _, _, tgt_len = (item.to(device) for item in batch)

    # Forward SL (同时训练 Oracle 和 Policy)
    actor_logits, pred_macro = model.forward_sl(
        hypergraph_list, relation_graph, tgt, gt_size=tgt_len
    )

    # 1. Micro Loss
    logits = actor_logits[:, :-1, :].reshape(-1, user_size)
    labels = tgt[:, 1:].reshape(-1)
    loss_micro = F.cross_entropy(logits, labels, ignore_index=Constants.PAD)

    # 2. Macro Loss (Critical for Phase 1)
    pred_log = torch.log2(pred_macro + 1)  # pred_macro 已经是 scalar
    tgt_log = torch.log2(tgt_len.float() + 1).unsqueeze(1)  # [B, 1]

    # 注意：这里的 pred_log 来源于 Oracle，labels 来源于 GT
    loss_macro = F.mse_loss(pred_log, tgt_log)

    loss = loss_micro + 1.0 * loss_macro

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    # Metrics
    pred_idx = logits.max(1)[1]
    n_correct = pred_idx.eq(labels).masked_select(labels.ne(Constants.PAD)).sum().float()
    n_total = labels.ne(Constants.PAD).sum().float()

    return loss.item(), n_correct, n_total


def train_rl_step_frozen_goal(model, train_loader, hypergraph_list, relation_graph, optimizer, device):
    model.eval()  # 确保 Dropout 关闭 (RL通常在eval模式下采样，或者开着也可以，但这里为了稳定先关)
    # 注意：model.macro_oracle 应该已经被 freeze_oracle() 处理过 requires_grad=False

    total_loss = 0
    batch_count = 0

    for batch in train_loader:
        tgt, _, _, tgt_len = (item.to(device) for item in batch)
        bs = tgt.size(0)

        # Warm Start 长度
        start_len = random.randint(2, 5)
        start_len = min(start_len, tgt.size(1) - 1)
        init_seq = tgt[:, :start_len]
        gt_sets = [set([u for u in t.cpu().numpy() if u != 0]) for t in tgt]

        # === 1. Generate FROZEN Goal (Once per episode) ===
        # 这个 goal_log 在接下来的 rollout 中绝对不变
        fixed_goal_log = model.get_initial_goal(hypergraph_list, relation_graph, init_seq)
        goal_scalars = torch.pow(2, fixed_goal_log).view(-1).cpu().numpy()  # 用于计算 Reward

        # Init RL State
        curr_seq = init_seq.clone()

        # 计算初始距离 (Phi_0)
        curr_sizes = [torch.count_nonzero(seq).item() for seq in curr_seq]
        prev_dists = [abs(math.log2(max(c, 1)) - math.log2(max(g, 1))) for c, g in zip(curr_sizes, goal_scalars)]

        saved_log_probs = []
        saved_rewards = []

        # === 2. Rollout Loop ===
        for t in range(opt.rollout_steps):
            # Policy Forward (Input: Seq + Fixed Goal)
            logits, _ = model.forward_policy_step(hypergraph_list, relation_graph, curr_seq, fixed_goal_log)

            dist = Categorical(logits=logits)
            actions = dist.sample()

            saved_log_probs.append(dist.log_prob(actions))

            # Env Step
            next_seq = torch.cat([curr_seq, actions.unsqueeze(1)], dim=1)

            # Compute Reward
            step_rewards = []
            new_dists = []

            for i in range(bs):
                # 真实演化的 Size
                real_size = torch.count_nonzero(next_seq[i]).item()

                r, new_d = compute_reward_delta(
                    actions[i].item(),
                    gt_sets[i],
                    real_size,
                    goal_scalars[i],
                    is_terminal=(t == opt.rollout_steps - 1),
                    prev_dist=prev_dists[i]
                )
                step_rewards.append(r)
                new_dists.append(new_d)

            saved_rewards.append(torch.tensor(step_rewards, device=device))

            # Update State
            curr_seq = next_seq
            prev_dists = new_dists

        # === 3. Update Policy (REINFORCE) ===
        R = torch.zeros(bs, device=device)
        policy_loss = []

        for i in reversed(range(opt.rollout_steps)):
            R = opt.gamma * R + saved_rewards[i]
            advantage = R - R.mean()  # Simple Baseline
            loss_step = -saved_log_probs[i] * advantage.detach()
            policy_loss.append(loss_step)

        final_loss = torch.stack(policy_loss).sum() / bs

        optimizer.zero_grad()
        final_loss.backward()
        # 这里的梯度只会更新 MicroPolicy，因为 MacroOracle 被锁住了
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += final_loss.item()
        batch_count += 1

    return total_loss / (batch_count + 1e-9)


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
                current_epoch_idx):
    sl_epochs = int(opt.sl_epochs)

    if current_epoch_idx < sl_epochs:
        # Phase 1: SL (Train Oracle + Policy Warmup)
        model.train()
        total_loss = 0
        n_correct = 0
        n_total = 0
        for batch in train_loader:
            l, c, t = train_sl_step(model, batch, hypergraph_list, relation_graph, optimizer, device, user_size)
            total_loss += l
            n_correct += c
            n_total += t
        print(
            f"   [SL Phase] Epoch {current_epoch_idx + 1} | Loss: {total_loss:.4f} | Acc: {n_correct / (n_total + 1e-5):.4f}")
        return total_loss, 0

    else:
        # Phase 2: RL (Frozen Oracle)

        # ★★★ One-time Freeze Trigger ★★★
        if current_epoch_idx == sl_epochs:
            print(">>> [System] Transitioning to RL Phase...")
            model.freeze_oracle()  # 冻结 Macro
            # 降低学习率，只微调 Policy
            for pg in optimizer.param_groups: pg['lr'] = opt.lr_rl

        loss = train_rl_step_frozen_goal(model, train_loader, hypergraph_list, relation_graph, optimizer, device)
        print(f"   [RL Phase] Epoch {current_epoch_idx + 1} | Policy Loss: {loss:.4f}")
        return loss, 0


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

            # [MODIFY] 适配新接口：
            # 使用 forward_sl，并且不传入 gt_size (设为 None)。
            # 这样内部会自动使用 Oracle 的预测值作为 target_signal，模拟真实测试场景。
            actor_logits, pred_macro = model.forward_sl(
                hypergraph_list, relation_graph, tgt,
                gt_size=None
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

            # 计算 Macro MSLE 指标
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

    print(f"Initializing MyModule (Macro-Guided)...")
    # === 使用新模型 MyModule ===
    model = MyModule(
        user_size=user_size,
        embed_dim=opt.emb_dim,
        step_split=opt.step_split,
        max_seq_len=opt.max_seq_length,
        device=device
    ).to(device)

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
                                             current_epoch_idx=epoch_i)
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