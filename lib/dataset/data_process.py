import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
import torch
from collections import defaultdict
from tqdm import tqdm
import time


def data_seg2(args):
    """
    根据args设定的dividing_time作为关键字划分训练集和测试集
    :param
        args: 设置超参的存储变量
    :return:
        res_cl：存储excel的dataframe
        scaler：数据标准化用的转换方法
        res_cl_train_norm， res_cl_test_norm：标准化的训练序列和测试序列，行向量(tensor)
    """
    data_path = args.data_path
    # n = args.train_data_num
    window_size = args.window_size
    dividing_time = args.dividing_time

    res_cl = pandas.read_excel(data_path)
    # res_cl = res_cl.drop(res_cl[res_cl.values[:, 1] == 0].index).dropna()     # 旧方法

    # 数据加载慢主要在于res_cl.values的遍历，所以使用下面的方法：
    index_list = res_cl[res_cl.values[:, 0] == "val"].index
    if len(index_list) == 1:
        i = index_list[0]
    else:
        error_string = "Something error occurs!! It seems that code doesn't find the dividing time!"
        print(f'\033[1;33;40m{error_string}\033[0m]')
        exit(1)

    train_dataframe = res_cl.iloc[:i, :]
    train_dataframe = train_dataframe.drop(train_dataframe[train_dataframe.values[:, 1] == 0].index).dropna()
    res_cl_train = train_dataframe.values[:, 1]

    test_dataframe = res_cl.iloc[i+1:, :]
    test_dataframe = test_dataframe.drop(test_dataframe[test_dataframe.values[:, 1] == 0].index).dropna()
    res_cl_test = test_dataframe.values[:, 1]

    info = '{:40}{:<40}{:<40}'.format(f'Train Data:{res_cl_train.shape[0]}', f'Test Data:{res_cl_test.shape[0]}',
                                      f'Dividing Time:{dividing_time}')
    print(f"\033[1;33;40m {info} \033[0m]")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # res_cl_train_norm = scaler.fit_transform(res_cl_train.reshape(-1, 1))    # 将数据变为列向量并归一化
    res_cl_train_norm = torch.FloatTensor(scaler.fit_transform(res_cl_train.reshape(-1, 1))).view(-1)
    # residualChlorineTest_norm = scaler.fit_transform(res_cl_test.reshape(-1, 1))
    res_cl_test_norm = torch.FloatTensor(scaler.fit_transform(res_cl_test.reshape(-1, 1))).view(-1)
    if torch.cuda.is_available() and args.gpu_enable:
        res_cl_train_norm = res_cl_train_norm.cuda()
        res_cl_test_norm = res_cl_test_norm.cuda()

    return res_cl, scaler, res_cl_train_norm, res_cl_test_norm, i


def data_seg1(args):
    """
    根据args设定的train_data_num超参设定训练集和测试集
    :param
        args: 设置超参的存储变量
    :return:
        res_cl：存储excel的dataframe
        scaler：数据标准化用的转换方法
        res_cl_train_norm， res_cl_test_norm：标准化的训练序列和测试序列，行向量(tensor)
    """
    data_path = args.data_path
    n = args.train_data_num

    res_cl = pandas.read_excel(data_path)
    res_cl = res_cl.drop(res_cl[res_cl.values[:, 1] == 0].index).dropna()

    res_cl_train = res_cl.values[0:n, 1]
    res_cl_test = res_cl.values[n:len(res_cl), 1]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    res_cl_train_norm = torch.FloatTensor(scaler.fit_transform(res_cl_train.reshape(-1, 1))).view(-1)
    res_cl_test_norm = torch.FloatTensor(scaler.fit_transform(res_cl_test.reshape(-1, 1))).view(-1)
    if torch.cuda.is_available() and args.gpu_enable:
        res_cl_train_norm = res_cl_train_norm.cuda()
        res_cl_test_norm = res_cl_test_norm.cuda()

    return res_cl, scaler, res_cl_train_norm, res_cl_test_norm


def data_inverse_transform(scaler, data):
    if isinstance(data, torch.Tensor):
        return scaler.inverse_transform(data.cpu().numpy().reshape(-1, 1))
    elif isinstance(data, list):
        return scaler.inverse_transform(np.array(data).reshape(-1, 1))
    elif isinstance(data, np.ndarray):
        return scaler.inverse_transform(data.reshape(-1, 1))
