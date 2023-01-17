# from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np


def plot_res(ws, train_pred, test_pred, res_cl_train, res_cl_test):
    x1 = np.arange(0, train_pred.shape[0], 1)
    x2 = np.arange(0, test_pred.shape[0], 1)
    print(x1.shape)
    plt.subplot(1, 2, 1)
    plt.plot(x1, res_cl_train[ws:, ], c='b')
    plt.plot(x1, train_pred, c='c', marker='.')
    plt.subplot(1, 2, 2)
    plt.plot(x2, res_cl_test[ws:, ], c='b')
    plt.plot(x2, test_pred, c='c', marker='.')
    plt.show()
    # pyplot.figure(figsize=(12, 4))
    # pyplot.grid(True)
    # pyplot.plot(res_cl['date'], res_cl['ResidualChlorine'])
    # x_1 = res_cl.values[args.window_size: args.train_data_num, 0]
    # x_2 = res_cl.values[args.train_data_num: len(res_cl) + 1, 0]
    # # x = numpy.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]').astype('datetime64[D]')
    # pyplot.plot(x_1, train_pred, color='blue')
    # pyplot.plot(x_2, test_pred, color='red')
    # pyplot.show()
