from options import args
import tqdm
import random
import copy
import numpy as np
import torch
import sys
import csv
from utils import load_lookups, prepare_instance, prepare_instance_bert, MyDataset, my_collate, my_collate_bert, \
    early_stop, save_everything, llprint
from models import pick_model
import torch.optim as optim
from collections import defaultdict #  默认字典类
from torch.utils.data import DataLoader
import os
import time
from train_test import train, test
from pytorch_pretrained_bert import BertAdam
import ctypes as ct

def drop_rate(epoch):
    max_epoch = 30
    drop = np.linspace(0,0.15, max_epoch)
    if epoch < max_epoch:
        return drop[epoch]
    else:
        return 0.15


if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)

    csv.field_size_limit(int(ct.c_ulong(-1).value // 2)) #sys.maxsize

    # load vocab and other lookups
    print("loading lookups...")
    dicts = load_lookups(args)  # look-up 方法 return dicts

    model = pick_model(args, dicts)
    print(model)

    if not args.test_model:
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = None

    if args.tune_wordemb == False:
        model.freeze_net()

    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: []) # 生成了三个字典


    if args.model.find("bert") != -1:
        prepare_instance_func = prepare_instance_bert
    else:
        prepare_instance_func = prepare_instance # 是否使用bert

    train_instances = prepare_instance_func(dicts, args.data_path, args, args.MAX_LENGTH)  # 统计训练的实例个数
    print("train_instances {}".format(len(train_instances)))
    if args.version != 'mimic2':
        dev_instances = prepare_instance_func(dicts, args.data_path.replace('train','dev'), args, args.MAX_LENGTH)
        print("dev_instances {}".format(len(dev_instances)))
    else:
        dev_instances = None
    test_instances = prepare_instance_func(dicts, args.data_path.replace('train','test'), args, args.MAX_LENGTH)
    print("test_instances {}".format(len(test_instances)))

    if args.model.find("bert") != -1:
        collate_func = my_collate_bert
    else:
        collate_func = my_collate

    train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=collate_func)
    if args.version != 'mimic2':
        dev_loader = DataLoader(MyDataset(dev_instances), 1, shuffle=False, collate_fn=collate_func)
    else:
        dev_loader = None
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=collate_func) # collate_fn：如何取样本的，自定义函数来实现想要的功能

    if not args.test_model and args.model.find("bert") != -1:
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_train_optimization_steps = int(
            len(train_instances) / args.batch_size + 1) * args.n_epochs

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=0.1,
                             t_total=num_train_optimization_steps)

    test_only = args.test_model is not None


    best_epoch = 0
    best_f1 = 0
    for epoch in range(args.n_epochs):
        start_time = time.time()




        loss_everyR = []
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(args.MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H_%M_%S', time.localtime())]))
            os.makedirs(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))

        if not test_only:
            for e in range(3):


                epoch_start = time.time()
                losses = train(args, model, optimizer, epoch, args.gpu, train_loader)
                if e==0:
                    # loss_everyR.append(losses.item())
                    loss_everyR = copy.deepcopy(losses)
                    continue
                elif e == 1:
                    loss_everyR = sorted(loss_everyR)
                    break
                jiezhi_loss = int(len(loss_everyR)* (1- drop_rate(epoch)))
                if losses > loss_everyR[jiezhi_loss - 1]:
                    continue



                # loss = np.mean(losses) # 对所有元素求均值 ,注释此处，将下面的loss更改为losses
                epoch_finish = time.time()
                print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, losses)) ############
        else:
            loss = np.nan

        fold = 'test' if args.version == 'mimic2' else 'dev'
        dev_instances = test_instances if args.version == 'mimic2' else dev_instances
        dev_loader = test_loader if args.version == 'mimic2' else dev_loader
        if epoch == args.n_epochs - 1:
            print("last epoch: testing on dev and test sets")
            test_only = True

        # test on dev
        evaluation_start = time.time()
        metrics = test(args, model, args.data_path, fold, args.gpu, dicts, dev_loader)
        evaluation_finish = time.time()
        print("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))
        if test_only or epoch == args.n_epochs - 1:
            metrics_te = test(args, model, args.data_path, "test", args.gpu, dicts, test_loader)
        else:
            metrics_te = defaultdict(float)
        metrics_tr = {'loss': losses} ##################
        metrics_all = (metrics, metrics_te, metrics_tr)

        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        a = metrics_te['f1_macro']
        if epoch != 0 and best_f1 < a:
            best_epoch = epoch
            best_f1 = a

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                              best_epoch,
                                                                                              elapsed_time,
                                                                                              elapsed_time * (
                                                                                                      args.n_epochs - epoch - 1) / 60))

        save_everything(args, metrics_hist_all, model, model_dir, None, args.criterion, test_only)

        sys.stdout.flush()




        if test_only:
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = pick_model(args, dicts)
    print('best_epoch:', best_epoch)




