from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from .rec_exp_info import rec_exp_info


def map_error(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def cal_err(args, gt, pred):
    mae = mean_absolute_error(gt, pred, multioutput='raw_values')
    mse = mean_squared_error(gt, pred, multioutput='raw_values')
    mape = map_error(gt, pred)
    print('{:40}{:<40}{:<40}'.format(f"MAE: {mae}", f"MSE: {mse}", f"MAPE: {mape}"))
    rec_exp_info(args, mae, mse, mape)
