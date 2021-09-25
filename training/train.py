import argparse
import numpy as np
import os
from training.load_data import load_EOD_data, get_batch
import torch.utils.data as Data
import torch
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
import torch.optim as optim
import time
import torch.nn.functional as F
import math
from tqdm import tqdm

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.eq(gold)
    n_correct = n_correct.sum().item()

    return loss, n_correct

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
    total_accu = 0
    n_count = 0
    for step, (eod, gt) in enumerate(training_data):

        # prepare data
        Eod, Gt = eod.to(device), gt.to(device)

        # forward
        optimizer.zero_grad()
        pred = model(Eod)

        # backward
        loss, n_correct = cal_performance(pred, Gt, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()
        total_accu += n_correct
        n_count += Eod.size(0) * Eod.size(1)

    epoch_loss = total_loss / n_count
    accuracy = total_accu / n_count
    return epoch_loss, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total_accu = 0
    n_count = 0

    with torch.no_grad():
        for step, (eod, gt) in enumerate(validation_data):

            # prepare data
            Eod, Gt = eod.to(device), gt.to(device)

            # forward
            pred = model(Eod)
            loss, n_correct = cal_performance(pred, Gt, smoothing=False)

            # note keeping
            total_loss += loss.item()
            total_accu += n_correct
            n_count += Eod.size(0) * Eod.size(1)

    epoch_loss = total_loss / n_count
    accuracy = total_accu / n_count
    return epoch_loss, accuracy


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
            log_tf.write('epoch,loss,accuracy\n')
            log_vf.write('epoch,loss,accuracy\n')

    valid_accus = []
    for epoch_i in range(args.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=args.label_smoothing)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=valid_loss, accu=100 * valid_accu,
            elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch_i}

        if args.save_model:
            if args.save_mode == 'all':
                model_name = args.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = args.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss, accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss, accu=100*valid_accu))



def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-market', help='market name', default='NASDAQ')
    parser.add_argument('-tickers', help='fname for selected tickers')
    parser.add_argument('-length', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-steps', default=1,
                        help='steps to make prediction')
    parser.add_argument('-threshold', type=float, default=0.005,
                        help='NASDAQ threshold')
    # parser.add_argument('-threshold', type=float, default=0.005,
    #                     help='NYSE threshold')

    parser.add_argument('-train_index', type=int, default=1008)

    # parser.add_argument('-gpu', '--gpu', type=int, default=0, help='use gpu')

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-d_model', type=int, default=9)
    parser.add_argument('-d_inner_hid', type=int, default=64)
    parser.add_argument('-d_k', type=int, default=32)
    parser.add_argument('-d_v', type=int, default=32)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=3)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-proj_share_weight', default='True')
    # parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default='../saveModel/transformer')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', default='True')
    # parser.add_argument('-label_smoothing', action='store_true')

    args = parser.parse_args()

    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model

    if args.tickers is None:
        args.tickers = args.market + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    # args.gpu = (args.gpu == 1)

    # ========= Loading Dataset =========#
    tickers = np.genfromtxt(os.path.join(args.path, '..', args.tickers),
                            dtype=str, delimiter='\t', skip_header=False)
    args.stocks_num = len(tickers)
    print(args.market, 'stock nums:', args.stocks_num)
    eod_data, ground_truth = load_EOD_data(data_path=args.path, market_name=args.market, tickers=tickers,
                                                  steps=args.steps, threshold=args.threshold)

    train_loader, valid_loader = prepare_dataloaders(eod_data, ground_truth, args)

    # ========= Preparing Model =========#
    print(args)
    device = torch.device('cuda' if args.cuda else 'cpu')

    transformer = Transformer(
        args.length,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        d_inner=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    train(transformer, train_loader, valid_loader, optimizer, device, args)

def prepare_dataloaders(eod_data, gt_data, args):
    # ========= Preparing DataLoader =========#
    EOD, GT = [], []
    for i in range(eod_data.shape[1] - args.length):
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

    train_eod, valid_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod)
    train_gt, valid_gt = torch.LongTensor(train_gt), torch.LongTensor(valid_gt)

    train_dataset = Data.TensorDataset(train_eod, train_gt)
    valid_dataset = Data.TensorDataset(valid_eod, valid_gt)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    return train_loader, valid_loader

if __name__ == '__main__':
    main()