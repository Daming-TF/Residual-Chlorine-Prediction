from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def map_error(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def cal_err(gt, pred):
    mae = mean_absolute_error(gt, pred, multioutput='raw_values')
    mse = mean_squared_error(gt, pred, multioutput='raw_values')
    mape = map_error(gt, pred)
    print('{:40}{:<40}{:<40}'.format(f"MAE[Train]: {mae}", f"MSE[Train]: {mse}",
                                     f"MAPE[Train]: {mape}"))
