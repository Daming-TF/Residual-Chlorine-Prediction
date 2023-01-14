import sys

import pandas
import numpy
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.utils.data
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
import os

search_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(search_path)
from lib.config import argparser
import lib.models
import lib.dataset.residual_chlorine as rc


def main():
    args = argparser.parser_args()
    res_cl = pandas.read_excel(r'E:\residual chlorine\HongmushanResidualChlorine.xlsx')   # 将excel数据读出，用values存储93*2的数据
    n = args.train_data_num
    window_size = args.window_size

    # 把余氯数据读出，并拆分为80个训练集，13个测试集
    res_cl_train = res_cl.values[0:n, 1]
    res_cl_test = res_cl.values[n:len(res_cl), 1]

    # 将数据标准化转化为（-1,1）范围内
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # res_cl_train_norm = scaler.fit_transform(res_cl_train.reshape(-1, 1))    # 将数据变为列向量并归一化
    res_cl_train_norm = torch.FloatTensor(scaler.fit_transform(res_cl_train.reshape(-1, 1))).view(-1)
    # residualChlorineTest_norm = scaler.fit_transform(res_cl_test.reshape(-1, 1))
    res_cl_test_norm = torch.FloatTensor(scaler.fit_transform(res_cl_test.reshape(-1, 1))).view(-1)
    if torch.cuda.is_available() and args.gpu_enable:
        res_cl_train_norm = res_cl_train_norm.cuda()
        res_cl_test_norm = res_cl_test_norm.cuda()

    # def input_data(seq, ws):
    #     out = []
    #     L = len(seq)
    #     for i in range(L-ws):
    #         window = seq[i:i+ws]
    #         label = seq[i+ws:i+ws+1]
    #         out.append((window, label))
    #     return out

    # train_data = input_data(residualChlorineTrain_norm_tenser, window_size)
    seq = res_cl_train_norm
    train_dataset = rc.ResidualChlorineDataset(copy.deepcopy(seq), window_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.workers,
    )

    torch.manual_seed(args.seed)      # 设置生成随机数的种子
    model = getattr(lib.models, 'base').get_net()
    criterion = nn.MSELoss()
    if torch.cuda.is_available() and args.gpu_enable:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    start_time = time.time()
    for epoch in range(args.epochs):
        for i, (seq, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            a = seq.reshape(-1, 1, window_size)       # {1*1*window_size}
            if torch.cuda.is_available() and args.gpu_enable:
                a = a.cuda()
                y_train = y_train.cuda()
            y_pred = model(a)
            print(y_pred.shape)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    trainNum = len(res_cl_train_norm)
    preds = res_cl_train_norm[-window_size:].tolist()       # 获取训练数据序列最后5个数据作为模型预测的初始数据

    model.eval()    # 不启用 Batch Normalization 和 Dropout


    def mape(actual, pred):
        actual, pred = numpy.array(actual), numpy.array(pred)
        return numpy.mean(numpy.abs((actual - pred) / actual)) * 100


    for i in range(trainNum):
        seq = torch.FloatTensor(preds[-window_size:])
        if torch.cuda.is_available() and args.gpu_enable:
            seq = seq.cuda()
        with torch.no_grad():
            preds.append(model(seq.reshape(1, 1, -1)).item())   # item()将零维tensor元素转换为浮点数

    trainPredictions = scaler.inverse_transform(numpy.array(preds[window_size:]).reshape(-1, 1))    # 将标准化后的数据转换为原始数据。
    MAE_train = mean_absolute_error(res_cl_train, trainPredictions, multioutput='raw_values')
    MSE_train = mean_squared_error(res_cl_train, trainPredictions, multioutput='raw_values')
    MAPE_train = mape(res_cl_train, trainPredictions)

    futureValues = len(res_cl_test_norm)

    preds = res_cl_test_norm[-window_size:].tolist()

    model.eval()

    for i in range(futureValues):
        seq = torch.FloatTensor(preds[-window_size:])
        if torch.cuda.is_available() and args.gpu_enable:
            seq = seq.cuda()
        with torch.no_grad():
            preds.append(model(seq.reshape(1, 1, -1)).item())

    testPredictions = scaler.inverse_transform(numpy.array(preds[window_size:]).reshape(-1, 1))
    MAE_test = mean_absolute_error(res_cl_test, testPredictions, multioutput='raw_values')
    MSE_test = mean_squared_error(res_cl_test, testPredictions, multioutput='raw_values')
    MAPE_test = mape(res_cl_test, testPredictions)

    pyplot.figure(figsize=(12, 4))
    pyplot.grid(True)
    pyplot.plot(res_cl['date'], res_cl['ResidualChlorine'])
    x = res_cl.values[n:len(res_cl) + 1, 0]
    # x = numpy.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]').astype('datetime64[D]')
    pyplot.plot(x, testPredictions, color='red')
    pyplot.show()


if __name__ == '__main__':
    main()
