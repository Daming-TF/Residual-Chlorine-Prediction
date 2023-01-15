from matplotlib import pyplot


def plot_res(res_cl, args, train_pred, test_pred):
    pyplot.figure(figsize=(12, 4))
    pyplot.grid(True)
    pyplot.plot(res_cl['date'], res_cl['ResidualChlorine'])
    x_1 = res_cl.values[args.window_size: args.train_data_num, 0]
    x_2 = res_cl.values[args.train_data_num: len(res_cl) + 1, 0]
    # x = numpy.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]').astype('datetime64[D]')
    pyplot.plot(x_1, train_pred, color='blue')
    pyplot.plot(x_2, test_pred, color='red')
    pyplot.show()
