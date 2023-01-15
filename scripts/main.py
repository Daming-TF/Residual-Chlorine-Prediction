import sys
import pandas
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.utils.data
import time
import os

search_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(search_path)

from lib.config import argparser
import lib.models
import lib.dataset.residual_chlorine as rc
from val import val
from lib.utils import plot_res


def main():
    args = argparser.parser_args()
    n = args.train_data_num
    window_size = args.window_size

    # Step1：把余氯数据读出，并拆分成n个训练集，其余为测试集
    res_cl = pandas.read_excel(r'E:\residual chlorine\HongmushanResidualChlorine.xlsx')
    res_cl_train = res_cl.values[0:n, 1]
    res_cl_test = res_cl.values[n-window_size:len(res_cl), 1]

    # Step2：将数据标准化转化为（-1,1）范围内
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # res_cl_train_norm = scaler.fit_transform(res_cl_train.reshape(-1, 1))    # 将数据变为列向量并归一化
    res_cl_train_norm = torch.FloatTensor(scaler.fit_transform(res_cl_train.reshape(-1, 1))).view(-1)
    # residualChlorineTest_norm = scaler.fit_transform(res_cl_test.reshape(-1, 1))
    res_cl_test_norm = torch.FloatTensor(scaler.fit_transform(res_cl_test.reshape(-1, 1))).view(-1)
    if torch.cuda.is_available() and args.gpu_enable:
        res_cl_train_norm = res_cl_train_norm.cuda()
        res_cl_test_norm = res_cl_test_norm.cuda()

    # Step3：网络设置（dataset，model，loss，opt）
    seq = res_cl_train_norm
    train_dataset = rc.ResidualChlorineDataset(seq, window_size)
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

    # Step4：网络训练(自监督)
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

    # Step5：模型验证
    """
    训练集和测试集验证逻辑有细微差异，以93个样本为例，我们想划分80个训练样本，13个测试样本
    考虑到初始序列，实际上我们需要划分80个训练样本和18个测试样本
    模型验证前输入模型的初始序列对应样本位置分别为[0:window_size]和[n-window_size:n]
    模型验证时去除初始序列样本，实际上有效验证的训练样本有75个，有效验证的测试样本有13个
    """
    model.eval()    # 不启用 Batch Normalization 和 Dropout
    train_num = len(res_cl_train_norm)
    train_init_seq = res_cl_train_norm[0:window_size]
    train_pred = val(args, model, train_num, "train", scaler, train_init_seq, res_cl_train)

    val_num = len(res_cl_test_norm)
    val_init_seq = res_cl_test_norm[0:window_size]
    test_pred = val(args, model, val_num, "test", scaler, val_init_seq, res_cl_test)

    # Step6:可视化预测结果
    plot_res(res_cl, args, train_pred, test_pred)


if __name__ == '__main__':
    main()
