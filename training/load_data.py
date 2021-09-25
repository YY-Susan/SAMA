import argparse
import numpy as np
import os
import pandas as pd

def load_air_data(data_path, steps=1, threshold=0.005):
    air_csv = np.genfromtxt(
        data_path, dtype=np.float32, delimiter=',',
        skip_header=1, usecols=(1, 2, 3, 4, 5, 6),
    )
    print('air_csv data shape:', air_csv.shape)
    # air_data = np.zeros([1, air_csv.shape[0], air_csv.shape[1]], dtype=np.float32)
    ground_truth = np.zeros([1, air_csv.shape[0]], dtype=np.float32)

    for row in range(air_csv.shape[0]):
        if row > steps - 1:
            after = air_csv[row][-1]
            before = air_csv[row - steps][-1]
            rate = (after - before) / before
            # if rate > 0:
            #     ground_truth[index][row] = 1
            # else:
            #     ground_truth[index][row] = 0
            if rate < -threshold:
                ground_truth[0][row] = 0
            elif rate > threshold:
                ground_truth[0][row] = 2
            else:
                ground_truth[0][row] = 1

    air_data = air_csv[np.newaxis, :, :]
    print('air_data', air_data.shape, 'ground_truth', ground_truth.shape)
    return air_data, ground_truth


def load_EOD_data(data_path, market_name, tickers, steps=1, threshold=0.005):
    eod_data = []
    ground_truth = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 5], dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)

        for row in range(single_EOD.shape[0]):
            if row > steps - 1:
                after = single_EOD[row][-1]
                before = single_EOD[row - steps][-1]
                rate = (after - before) / before
                # if rate > 0:
                #     ground_truth[index][row] = 1
                # else:
                #     ground_truth[index][row] = 0
                if rate < -threshold:
                    ground_truth[index][row] = 0
                elif rate > threshold:
                    ground_truth[index][row] = 2
                else:
                    ground_truth[index][row] = 1

        eod_data[index, :, :] = single_EOD[:, 5:]
    return eod_data, ground_truth

def load_EOD_data_(data_path, market_name, tickers, steps=1, threshold=0.005):
    eod_data = []
    ground_truth = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 5], dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)

        for row in range(single_EOD.shape[0] - steps):
            if row > steps - 1:
                after = 0
                for i in range(row, row+steps):
                    after += single_EOD[i][-1]
                after /= steps
                before = single_EOD[row-1][-1]
                rate = (after - before) / before
                # if rate > 0:
                #     ground_truth[index][row] = 1
                # else:
                #     ground_truth[index][row] = 0
                if rate < -threshold:
                    ground_truth[index][row] = 0
                elif rate > threshold:
                    ground_truth[index][row] = 2
                else:
                    ground_truth[index][row] = 1

        eod_data[index, :, :] = single_EOD[:, 5:]
    return eod_data, ground_truth

def get_batch(eod_data, gt_data, offset, seq_len):
    return eod_data[:, offset:offset+seq_len, :], \
           gt_data[:, offset+seq_len]


if __name__ == '__main__':
    desc = 'train a rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)
    tickers = np.genfromtxt(os.path.join(args.p, '..', args.t),
                            dtype=str, delimiter='\t', skip_header=False)

    eod_data, ground_truth = load_EOD_data(data_path=args.p, market_name=args.m, tickers=tickers, steps=1)
    print(eod_data.shape, ground_truth.shape)
