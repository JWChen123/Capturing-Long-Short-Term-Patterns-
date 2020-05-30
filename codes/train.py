# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch

import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg
from collections import deque, Counter
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve


class ModelParameterData(object):
    def __init__(self,
                 loc_emb_size=300,
                 hidden_size=300,
                 tim_emb_size=20,
                 loc_cat_emb_size=50,
                 pool_size=10,
                 lr=1e-4,
                 lr_step=3,
                 lr_decay=0.5,
                 dropout_p=0.6,
                 L2=1e-7,
                 clip=3.0,
                 epoch_max=20,
                 optim='Adam',
                 rnn_type='LSTM',
                 data_path='./data/',
                 save_path='./results/',
                 data_name='NYC'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        data = pickle.load(
            open(self.data_path + self.data_name + '.pk', 'rb'),
            encoding='iso-8859-1')
        self.vid_list = data['vid_list']
        self.vid_list_cat = data['vid_list_cat']  # NYC:376 TKY:338
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']

        self.loc_size = len(self.vid_list)
        self.loc_cat_size = len(self.vid_list_cat)
        self.uid_size = len(self.uid_list)
        self.tim_size = 48
        self.pool_size = pool_size
        self.loc_cat_emb_size = loc_cat_emb_size
        self.tim_emb_size = tim_emb_size
        self.loc_emb_size = loc_emb_size
        self.hidden_size = hidden_size

        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip
        self.rnn_type = rnn_type


def data_statistics(parameters):
    data = parameters.data_neural
    user_set = data.keys()
    idx = 0
    for u in user_set:
        sessions = data[u]['sessions']
        sub_len = len(sessions)
        idx += sub_len
    return idx


def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])
            if len(target) == 1:
                pass

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1], s[2])
                                    for s in sessions[tt]])  # pid,tid,p_catid
            for j in range(c):
                history.extend(
                    [(s[0], s[1], s[2]) for s in sessions[train_id[j]]])

            loc_tim = history
            loc_tim.extend([(s[0], s[1], s[2]) for s in session[:-1]])
            loc_np = np.reshape(
                np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(
                np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            p_cat_np = np.reshape(
                np.array([s[2] for s in loc_tim]), (len(loc_tim), 1))

            trace['loc'] = torch.LongTensor(loc_np)
            trace['tim'] = torch.LongTensor(tim_np)
            trace['target'] = torch.LongTensor(target)
            trace['p_cat'] = torch.LongTensor(p_cat_np)
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = train_idx.keys()
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        list_user = list(user)
        while queue_left > 0:
            # 打乱用户数据
            np.random.shuffle(list_user)
            for j, u in enumerate(list_user):
                if len(initial_queue[u]) > 0:
                    # train_queue=([(uid,initial_queue.popoleft())])
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum(
                [1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def markov(parameters, candidate):
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        test_id = parameters.data_neural[u]['test']
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])
        locations_train = []
        for t in trace_train:
            locations_train.extend(t)
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]
    acc = 0
    acc1 = 0
    count = 0
    user_acc = {}
    user_acc_top5 = {}
    for u in validation.keys():
        # 去除重复地点
        topk = list(set(validation[u][0]))
        # 状态转移概率矩阵 所有状态即为topk
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                if loc in topk and target in topk:
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):
            # 状态转移概率矩阵每行和为1
            # 行代表当前状态，列代表下一状态
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum

        # validation
        user_count = 0
        user_acc[u] = 0
        user_acc_top5[u] = 0
        test_id = parameters.data_neural[u]['test']
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1
                user_count += 1
                # 如果这个地点从未去过，什么也不做
                if loc in topk:
                    # 获取loc所在状态转移概率矩阵中所在行中转移到下一状
                    # 态的最大值所在索引
                    temp_count = 0
                    tt = []
                    pred = np.argmax(transfer[topk.index(loc), :])
                    # 获取前5个可能去的地方
                    # the reason of - is argsort function
                    tmp = np.argsort(-transfer[topk.index(loc), :])
                    for m in tmp:
                        if transfer[topk.index(loc), m] > 0:
                            temp_count += 1
                            if m >= len(topk) - 1:
                                tt.append(np.random.randint(len(topk)))
                            else:
                                tt.append(m)
                        if temp_count > 5:
                            break
                    if len(tt) < 5:
                        for _ in range(5 - len(tt)):
                            tt.append(np.random.randint(len(topk)))
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))
                    tt1 = []
                    for n in tt:
                        tt1.append(topk[n])
                    if target in tt1:
                        acc1 += 1
                        user_acc_top5[u] += 1

                    pred2 = topk[pred]
                    if pred2 == target:
                        acc += 1
                        user_acc[u] += 1
        # 当前用户准确率
        user_acc[u] = user_acc[u] / user_count
        user_acc_top5[u] = user_acc_top5[u] / user_count
    # 整个样本平均准确率
    avg_acc_top5 = np.mean([user_acc_top5[u] for u in user_acc_top5])
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    return avg_acc, avg_acc_top5, user_acc
