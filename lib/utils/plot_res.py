# from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np


def plot_res(label_length, ws, once_pred,
             train_pred=None, test_pred=None, res_cl_train=None, res_cl_test=None,
             size=None, interval=1):
    x1 = np.arange(0, test_pred.shape[0], 1)
    x2 = np.arange(0, once_pred.shape[0], 1)

    plt.subplot(size[0], size[1], 1)
    plt.plot(x1[::interval], res_cl_test[ws::interval, ], c='b', marker='+')
    plt.plot(x1[::interval], test_pred[::interval], c='c', marker='.')
    plt.subplot(size[0], size[1], 2)
    plt.plot(x2[::interval], res_cl_test[ws:ws + label_length:interval, ], c='b', marker='+')
    plt.plot(x2[::interval], once_pred[::interval], c='c', marker='.')

    if train_pred is not None:
        x3 = np.arange(0, train_pred.shape[0], 1)
        plt.subplot(1, 3, 3)
        plt.plot(x3[::interval], res_cl_train[ws::interval, ], c='b', marker='+')
        plt.plot(x3[::interval], train_pred[::interval], c='c', marker='.')

    plt.show()


# def plot2(label_length, ws, once_pred, test_pred, res_cl_test, interval=1):
#     x1 = np.arange(0, once_pred.shape[0], 1)
#     x2 = np.arange(0, test_pred.shape[0], 1)
#
#     plt.subplot(1, 2, 1)
#     plt.plot(x1[::interval], res_cl_test[ws:ws+label_length:interval, ], c='b', marker='+')
#     plt.plot(x1[::interval], once_pred[::interval], c='c', marker='.')
#     plt.subplot(1, 2, 2)
#     plt.plot(x2[::interval], res_cl_test[ws::interval, ], c='b', marker='+')
#     plt.plot(x2[::interval], test_pred[::interval], c='c', marker='.')
#     plt.show()
