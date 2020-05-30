import torch
from model import Model
from train import generate_input_long_history, generate_queue
from train import ModelParameterData, data_statistics
from sklearn.metrics import precision_score, recall_score
import torch.nn as nn
import torch.optim as optim
import math
import json
import time
import random
import math
import numpy as np
import argparse
import os


def run(args):
    parameters = ModelParameterData(
        loc_emb_size=args.loc_emb_size,
        loc_cat_emb_size=args.cat_emb_size,
        hidden_size=args.hidden_size,
        pool_size=args.pool_size,
        dropout_p=args.dropout_p,
        data_name=args.data_name,
        lr=args.learning_rate,
        lr_step=args.lr_step,
        lr_decay=args.lr_decay,
        L2=args.L2,
        optim=args.optim,
        clip=args.clip,
        epoch_max=args.epoch_max,
        data_path=args.data_path,
        save_path=args.save_path)
    # sub_len = data_statistics(parameters)

    candidate = parameters.data_neural.keys()
    print('prepare the data')
    data_train, train_idx = generate_input_long_history(
        parameters.data_neural, 'train', candidate)
    data_test, test_idx = generate_input_long_history(parameters.data_neural,
                                                      'test', candidate)

    print('set the parameters')
    # initial model
    model = Model(parameters)
    # Move models to GPU
    if args.USE_CUDA:
        model.cuda()
    SAVE_PATH = args.save_path
    try:
        os.mkdir(SAVE_PATH)
    except FileExistsError:
        pass
    # 度量标准
    metrics = {
        'train_loss': [],
        'valid_loss': [],
        'accuracy': [],
        'accuracy_top5': [],
        'accuracy_top10': []
    }
    lr = parameters.lr  # 学习速率

    # Initialize optimizers and criterion

    model_optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        model_optimizer,
        'max',
        patience=parameters.lr_step,
        factor=parameters.lr_decay,
        threshold=1e-3)  # 动态学习率
    criterion = nn.NLLLoss().cuda()
    # weight_mask = torch.ones(parameters.loc_size).cuda()
    # weight_mask[parameters.vid_list['pad']] = 0
    # perplexity = nn.CrossEntropyLoss(weight_mask).cuda()

    print('begin the train')
    if args.pretrain == 0:
        # Keep track of time elapsed and running averages
        start = time.time()
        for epoch in range(1, args.epoch_max + 1):
            # Run the train function
            loss, model = run_new(args, data_train, train_idx, 'train', lr,
                                  parameters.clip, model, model_optimizer,
                                  criterion)

            print_summary = '%s (%d %d%%) %.4f %g' % (
                time_since(start, epoch / args.epoch_max), epoch,
                epoch / args.epoch_max * 100, loss, lr)
            print(print_summary)
            metrics['train_loss'].append(loss)

            valid_loss, avg_acc, avg_acc_top5, avg_acc_top10, recall_5, precision_5, f1_5 = run_new(
                args, data_test, test_idx, 'test', lr, parameters.clip, model,
                model_optimizer, criterion)
            print(
                'loss: %.3f acc@1: %.3f. acc@5: %.3f acc@10: %.3f recall@5:%.3f precison@5:%.3f f1@5: %.3f'
                % (valid_loss, avg_acc, avg_acc_top5, avg_acc_top10, recall_5,
                   precision_5, f1_5))
            metrics['valid_loss'].append(valid_loss)
            metrics['accuracy'].append(avg_acc)
            metrics['accuracy_top5'].append(avg_acc_top5)
            metrics['accuracy_top10'].append(avg_acc_top10)
            save_name = 'ep_' + str(epoch) + 'model.m'
            torch.save(model.state_dict(), args.save_path + save_name)
            scheduler.step(avg_acc)
            lr_last = lr
            lr = model_optimizer.param_groups[0]['lr']
            if lr_last > lr:
                load_epoch = np.argmax(metrics['accuracy'])
                load_name = 'ep_' + str(load_epoch + 1) + 'model.m'
                model.load_state_dict(torch.load(args.save_path + load_name))
                print('load epoch={} model state'.format(load_epoch + 1))
            if lr <= 0.5 * 1e-7:
                break

        metrics_view = {
            'train_loss': [],
            'valid_loss': [],
            'accuracy': [],
            'accuracy_top5': [],
            'accuracy_top10': []
        }
        for key in metrics_view:
            metrics_view[key] = metrics[key]
        json.dump({
            'metrics': metrics_view,
            'param': {
                'hidden_size': parameters.hidden_size,
                'L2': parameters.L2,
                'lr': parameters.lr,
                'loc_emb': parameters.loc_emb_size,
                'cat_emb': parameters.loc_cat_emb_size,
                'dropout': parameters.dropout_p,
                'clip': parameters.clip,
                'lr_step': parameters.lr_step,
                'lr_decay': parameters.lr_decay
            }
        },
                  fp=open('./results/' + 'tmp_res' + '.txt', 'w'),
                  indent=4)
    elif args.pretrain == 1:
        model.load_state_dict(torch.load('./results/TKY_10_2.m'))
        valid_loss, avg_acc, avg_acc_top5, avg_acc_top10, f1_5 = run_new(
            args, data_test, test_idx, 'test', lr, parameters.clip, model,
            model_optimizer, criterion)
        print('loss: %.3f acc@1: %.3f. acc@5: %.3f acc@10: %.3f f1@5:%.3f' %
              (valid_loss, avg_acc, avg_acc_top5, avg_acc_top10, f1_5))


def getacc(score, target):
    target = target.data.cpu().numpy()
    _, topi = score.data.topk(10, 1)
    acc = np.zeros((3, 1))
    index = topi.cpu().numpy()
    y_pred = np.array([pred[0] for pred in index])
    y_pred_5 = [pred[:5] for pred in index]

    for i, p in enumerate(index):
        t = target[i]
        if t == p[0] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t in p[:10] and t > 0:
            acc[2] += 1
    return acc, y_pred, y_pred_5, target


def run_new(args,
            data,
            run_idx,
            mode,
            lr,
            clip,
            model,
            model_optimizer,
            criterion,
            mode2=None):
    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)
    pred_1 = {}
    pred_5 = {}
    ground_target = {}

    users_acc = {}

    for _ in range(queue_len):
        model_optimizer.zero_grad()
        loss = 0
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0, 0, 0]
            pred_1[u] = []
            pred_5[u] = []
            ground_target[u] = []
        loc = data[u][i]['loc'].cuda()
        tim = data[u][i]['tim'].cuda()
        activity = data[u][i]['p_cat'].cuda()
        target = data[u][i]['target'].cuda()
        target_len = target.data.size()[0]
        score = model(loc, tim, activity, target_len)
        loss = criterion(score, target)

        if mode == 'train':
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            model_optimizer.step()

        if mode == 'test':
            acc, y_pred, y_pred_5, target = getacc(score, target)
            pred_1[u].extend(y_pred)
            pred_5[u].extend(y_pred_5)
            ground_target[u].extend(target)
            # acc@k
            users_acc[u][0] += target_len
            users_acc[u][1] += acc[0]
            users_acc[u][2] += acc[1]
            users_acc[u][3] += acc[2]
        total_loss.append(loss.item())
    epoch_loss = np.mean(total_loss)
    if mode == 'train':
        return epoch_loss, model
    elif mode == 'test':
        users_rnn_acc = {}
        users_rnn_acc_top5 = {}
        users_rnn_acc_top10 = {}
        users_recall1 = {}
        users_recall5 = {}
        users_precision1 = {}
        users_precision5 = {}

        for u in users_acc:
            prec5 = []
            reca5 = []
            # Top@k
            tmp_acc = users_acc[u][1] / users_acc[u][0]
            top5_acc = users_acc[u][2] / users_acc[u][0]
            top10_acc = users_acc[u][3] / users_acc[u][0]
            users_rnn_acc[u] = tmp_acc
            users_rnn_acc_top5[u] = top5_acc
            users_rnn_acc_top10[u] = top10_acc
            # recall@1|precision@1
            users_recall1[u] = recall_score(
                pred_1[u], ground_target[u], average='micro')
            users_precision1[u] = precision_score(
                pred_1[u], ground_target[u], average='micro')
            # recall@5|precision@5
            for j in range(np.array(pred_5[u]).shape[0]):
                prec = len(set(ground_target[u][:j + 1]) & set(pred_5[u][j])
                           ) / float(5)
                reca = len(set(ground_target[u][:j + 1]) & set(pred_5[u][j])
                           ) / float(len(set(ground_target[u][:j + 1])))
                prec5.append(prec)
                reca5.append(reca)
            users_recall5[u] = np.mean(reca5)
            users_precision5[u] = np.mean(prec5)
        avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
        avg_acc_top5 = np.mean(
            [users_rnn_acc_top5[x] for x in users_rnn_acc_top5])
        avg_acc_top10 = np.mean(
            [users_rnn_acc_top10[x] for x in users_rnn_acc_top10])
        avg_recall1 = np.mean([users_recall1[x] for x in users_recall1])
        avg_recall5 = np.mean([users_recall5[x] for x in users_recall5])
        avg_precision1 = np.mean(
            [users_precision1[x] for x in users_precision1])
        avg_precision5 = np.mean(
            [users_precision5[x] for x in users_precision5])
        f1 = 2 * avg_recall1 * avg_precision1 / (avg_recall1 + avg_precision1)
        f1_5 = 2 * avg_recall5 * avg_precision5 / (
            avg_recall5 + avg_precision5)
        return epoch_loss, avg_acc, avg_acc_top5, avg_acc_top10, avg_recall5, avg_precision5, f1_5


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    np.random.seed(1) 
    torch.manual_seed(1) 
    parser.add_argument(
        '--loc_emb_size',
        type=int,
        default=400,
        help="location embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=20)
    parser.add_argument('--cat_emb_size', type=int, default=50)
    parser.add_argument('--USE_CUDA', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--pool_size', type=int, default=10)
    parser.add_argument('--dropout_p', type=float, default=0.7)
    parser.add_argument(
        '--data_name',
        type=str,
        default='NYC_10',
        choices=['NYC_10', 'TKY_10'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_step', type=int, default=3)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument(
        '--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument(
        '--L2', type=float, default=1e-7, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=6.0)
    parser.add_argument('--epoch_max', type=int, default=50)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument(
        '--save_path', type=str, default='./results/checkpoint/')
    args = parser.parse_args()
    run(args)