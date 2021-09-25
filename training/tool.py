import os
import torch
import torch.nn.functional as F
import time
import numpy as np
from training.load_data import get_batch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    return loss

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')

    return loss

def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_count = 0
    prediction = []
    label = []

    for step, (eod, gt) in enumerate(training_data):

        # prepare data
        Eod, Gt = eod.to(device), gt.to(device)

        # forward
        optimizer.zero_grad()
        pred = model(Eod)

        # backward
        loss = cal_performance(pred, Gt, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()
        # optimizer.step()

        # note keeping
        total_loss += loss.item()
        n_count += Eod.size(0) * Eod.size(1)

        pred = pred.max(1)[1]
        gold = Gt.contiguous().view(-1)

        pred = np.array(pred.cpu())
        gold = np.array(gold.cpu())

        length = len(pred)
        for i in range(length):
            prediction.append(pred[i])
            label.append(gold[i])

    epoch_loss = total_loss / n_count
    accuracy = 100 * accuracy_score(label, prediction)
    precision = 100 * precision_score(label, prediction, average='macro')
    recall = 100 * recall_score(label, prediction, average='macro')
    f1 = 100 * f1_score(label, prediction, average='macro')

    return epoch_loss, accuracy, precision, recall, f1

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_count = 0
    prediction = []
    label = []

    with torch.no_grad():
        for step, (eod, gt) in enumerate(validation_data):

            # prepare data
            Eod, Gt = eod.to(device), gt.to(device)

            # forward
            pred = model(Eod)
            loss = cal_performance(pred, Gt, smoothing=False)

            # note keeping
            total_loss += loss.item()
            n_count += Eod.size(0) * Eod.size(1)

            pred = pred.max(1)[1]
            gold = Gt.contiguous().view(-1)

            pred = np.array(pred.cpu())
            gold = np.array(gold.cpu())

            length = len(pred)
            for i in range(length):
                prediction.append(pred[i])
                label.append(gold[i])

    epoch_loss = total_loss / n_count
    accuracy = 100 * accuracy_score(label, prediction)
    precision = 100 * precision_score(label, prediction, average='macro')
    recall = 100 * recall_score(label, prediction, average='macro')
    f1 = 100 * f1_score(label, prediction, average='macro')

    return epoch_loss, accuracy, precision, recall, f1


def train(model, training_data, validation_data, optimizer, device, args):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if args.log:
        log_train_file = args.log + '.train.log'
        log_valid_file = args.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch, loss, accuracy, precision, recall, f1\n')
            log_vf.write('epoch, loss, accuracy, precision, recall, f1\n')

    writer = SummaryWriter(comment=args.log[7:])

    valid_accus = []
    for epoch_i in range(args.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, train_mp, train_mr, train_f1 = train_epoch(
            model, training_data, optimizer, device, smoothing=args.label_smoothing)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.2f} %, '
              'precision: {mp: 3.2f} %, recall: {mr: 3.2f} %, f1: {f1: 3.2f} % ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=train_accu, mp=train_mp, mr=train_mr, f1=train_f1,
            elapse=(time.time() - start) / 60))

        writer.add_scalar('scalar/loss_train', train_loss, epoch_i + 1)
        writer.add_scalar('scalar/acc_train', train_accu, epoch_i + 1)
        writer.add_scalar('scalar/mp_train', train_mp, epoch_i + 1)
        writer.add_scalar('scalar/mr_train', train_mr, epoch_i + 1)
        writer.add_scalar('scalar/f1_train', train_f1, epoch_i + 1)


        start = time.time()
        valid_loss, valid_accu, valid_mp, valid_mr, valid_f1 = eval_epoch(model, validation_data, device)
        print('  - (validation)   loss: {loss: 8.5f}, accuracy: {accu:3.2f} %, '
              'precision: {mp: 3.2f} %, recall: {mr: 3.2f} %, f1: {f1: 3.2f} % ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=valid_loss, accu=valid_accu, mp=valid_mp, mr=valid_mr, f1=valid_f1,
            elapse=(time.time() - start) / 60))

        writer.add_scalar('scalar/loss_valid', valid_loss, epoch_i + 1)
        writer.add_scalar('scalar/acc_valid', valid_accu, epoch_i + 1)
        writer.add_scalar('scalar/mp_valid', valid_mp, epoch_i + 1)
        writer.add_scalar('scalar/mr_valid', valid_mr, epoch_i + 1)
        writer.add_scalar('scalar/f1_valid', valid_f1, epoch_i + 1)

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch_i}

        if args.save_model:
            if args.save_mode == 'all':
                model_name = args.save_model + '_accu_{accu:3.2f}.chkpt'.format(accu=valid_accu)
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = args.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch: 4.0f},{loss: 8.5f},{accu: 3.2f},{mp: 3.2f},{mr: 3.2f},{f1: 3.2f}\n'.format(
                    epoch=epoch_i, loss=train_loss, accu=train_accu, mp=train_mp, mr=train_mr, f1=train_f1))
                log_vf.write('{epoch: 4.0f},{loss: 8.5f},{accu: 3.2f},{mp: 3.2f},{mr: 3.2f},{f1: 3.2f}\n'.format(
                    epoch=epoch_i, loss=valid_loss, accu=valid_accu, mp=valid_mp, mr=valid_mr, f1=valid_f1))

    writer.close()

    if log_valid_file:
        with open(log_valid_file, 'a') as log_vf:
            log_vf.write('{Best:3.2f}\n'.format(Best=max(valid_accus)))

def prepare_dataloaders(eod_data, gt_data, args):
    # ========= Preparing DataLoader =========#
    EOD, GT = [], []
    for i in range(eod_data.shape[1] - args.length - 1):
        eod, gt = get_batch(eod_data, gt_data, i, args.length)
        EOD.append(eod)
        GT.append(gt)

    train_eod, train_gt = EOD[:args.train_index], GT[:args.train_index]
    valid_eod, valid_gt = EOD[args.train_index:], GT[args.train_index:]

    # ========= debug =========#
    # train1, train2, train3 = 0,0,0
    # test1, test2, test3 = 0,0,0
    # for i in train_gt:
    #     for j in i:
    #         if j==0:
    #             train1+=1
    #         elif j==1:
    #             train2+=1
    #         else:
    #             train3+=1
    #
    # for i in valid_gt:
    #     for j in i:
    #         if j==0:
    #             test1+=1
    #         elif j==1:
    #             test2+=1
    #         else:
    #             test3+=1
    #
    # trains = train1+train2+train3
    # tests = test1+test2+test3
    # print(train1/trains, train2/trains, train3/trains)
    # print(test1/tests, test2/tests, test3/tests)
    # exit()

    train_eod, valid_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod)
    train_gt, valid_gt = torch.LongTensor(train_gt), torch.LongTensor(valid_gt)

    print(train_eod.shape, valid_eod.shape)
    print(train_gt.shape, valid_gt.shape)

    train_dataset = Data.TensorDataset(train_eod, train_gt)
    valid_dataset = Data.TensorDataset(valid_eod, valid_gt)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    return train_loader, valid_loader

def prepare_dataloaders_(eod_data, gt_data, args):
    # ========= Preparing DataLoader =========#
    EOD, GT = [], []
    for i in range(eod_data.shape[1] - args.length - args.steps - 1):
        eod, gt = get_batch(eod_data, gt_data, i, args.length)
        EOD.append(eod)
        GT.append(gt)

    train_eod, train_gt = EOD[:args.train_index], GT[:args.train_index]
    valid_eod, valid_gt = EOD[args.train_index:], GT[args.train_index:]

    # ========= debug =========#
    # train1, train2, train3 = 0,0,0
    # test1, test2, test3 = 0,0,0
    # for i in train_gt:
    #     for j in i:
    #         if j==0:
    #             train1+=1
    #         elif j==1:
    #             train2+=1
    #         else:
    #             train3+=1
    #
    # for i in valid_gt:
    #     for j in i:
    #         if j==0:
    #             test1+=1
    #         elif j==1:
    #             test2+=1
    #         else:
    #             test3+=1
    #
    # trains = train1+train2+train3
    # tests = test1+test2+test3
    # print(train1/trains, train2/trains, train3/trains)
    # print(test1/tests, test2/tests, test3/tests)
    # exit()

    train_eod, valid_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod)
    train_gt, valid_gt = torch.LongTensor(train_gt), torch.LongTensor(valid_gt)

    print(train_eod.shape, valid_eod.shape)
    print(train_gt.shape, valid_gt.shape)

    train_dataset = Data.TensorDataset(train_eod, train_gt)
    valid_dataset = Data.TensorDataset(valid_eod, valid_gt)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    return train_loader, valid_loader