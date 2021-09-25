import sys
sys.path.insert(0, '../')
import argparse
import numpy as np
from training.load_data import load_EOD_data, load_air_data
import torch
from Net.Models_SAMA import SAMA
from transformer.Optim import ScheduledOptim
from training.tool import prepare_dataloaders, train
import torch.optim as optim
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-market', help='market name', default='NASDAQ')
    # parser.add_argument('-market', help='market name', default='NYSE')
    parser.add_argument('-tickers', help='fname for selected tickers')
    parser.add_argument('-length', default=5,
                        help='length of historical sequence for feature')
    parser.add_argument('-nums', default=1)
    parser.add_argument('-t1', default=5)
    parser.add_argument('-t2', default=10)
    parser.add_argument('-t3', default=20)
    parser.add_argument('-steps', default=1,
                        help='steps to make prediction')
    parser.add_argument('-threshold', type=float, default=0.005,
                        help='NASDAQ threshold')
    # parser.add_argument('-threshold', type=float, default=0.005,
    #                     help='NYSE threshold')

    parser.add_argument('-train_index', type=int, default=1008)
    # parser.add_argument('-train_index', type=int, default=1457)
    # parser.add_argument('-gpu', '--gpu', type=int, default=0, help='use gpu')

    parser.add_argument('-epoch', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-d_model', type=int, default=16) #d_k=d_v=d_model/n_head
    parser.add_argument('-d_k', type=int, default=4)
    parser.add_argument('-d_v', type=int, default=4)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=3)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-proj_share_weight', default='True')
    # parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default='../log/SAMA5days_1')
    parser.add_argument('-save_model', default='../saveModel/SAMA5days_softmax_2')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', default='True')
    # parser.add_argument('-label_smoothing', action='store_true')

    args = parser.parse_args()

    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model

    if args.tickers is None:
        args.tickers = args.market + '_tickers_qualify_dr-0.98_min-5_smooth.csv'


    # ========= Loading Dataset =========#
    tickers = np.genfromtxt(os.path.join(args.path, '..', args.tickers),
                            dtype=str, delimiter='\t', skip_header=False)
    args.stocks_num = len(tickers)
    print(args.market, 'stock nums:', args.stocks_num)
    eod_data, ground_truth = load_EOD_data(data_path=args.path, market_name=args.market, tickers=tickers,
                                                  steps=args.steps, threshold=args.threshold)

    train_loader, valid_loader = prepare_dataloaders(eod_data, ground_truth, args)

    # eod_data, ground_truth = load_air_data('./AirQuality_new.csv')
    #
    # train_loader, valid_loader = prepare_dataloaders(eod_data, ground_truth, args)

    # ========= Preparing Model =========#
    print(args)
    device = torch.device('cuda' if args.cuda else 'cpu')

    model = SAMA(
        len_max_seq=args.length,
        nums=args.nums,
        t1=args.t1,
        t2=args.t2,
        t3=args.t3,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout
    ).to(device)
    model = nn.DataParallel(model, device_ids=[0])
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    # optimizer = optim.Adam(Hgru.parameters(), lr=args.r)

    train(model, train_loader, valid_loader, optimizer, device, args)

if __name__ == '__main__':
    main()