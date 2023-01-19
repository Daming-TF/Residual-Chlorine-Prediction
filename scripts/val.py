import torch
import numpy as np
from tqdm import tqdm

from lib.utils.calculation_errors import cal_err
from lib.dataset import data_inverse_transform


def val(args, model, scaler, raw_data_norm):
    window_size = args.window_size
    input_seq = raw_data_norm[0:window_size].tolist()       # 存放归一化预测数据
    preds_norm = []      # 用于存储输入模型的序列数据

    for i in tqdm(range(len(raw_data_norm)-window_size)):
        input_seq[i+window_size-1] = raw_data_norm[i+window_size-1]
        seq = torch.FloatTensor(input_seq[-window_size:])
        if torch.cuda.is_available() and args.gpu_enable:
            seq = seq.cuda()
        with torch.no_grad():
            out = model(seq.reshape(1, 1, -1)).item()
            input_seq.append(out)
            preds_norm.append(out)

    preds = data_inverse_transform(scaler, preds_norm)
    raw_data = data_inverse_transform(scaler, raw_data_norm)
    cal_err(raw_data[window_size:len(raw_data_norm)], preds)
    return preds, raw_data


def val_old(args, model, data_num, scaler, init_seq, raw_data):
    if not isinstance(init_seq, list):
        init_seq = init_seq.tolist()
    preds_norm = init_seq

    window_size = args.window_size

    for i in range(data_num-window_size):
        seq = torch.FloatTensor(preds_norm[-window_size:])
        if torch.cuda.is_available() and args.gpu_enable:
            seq = seq.cuda()
        with torch.no_grad():
            preds_norm.append(model(seq.reshape(1, 1, -1)).item())

    preds = scaler.inverse_transform(np.array(preds_norm[window_size:data_num]).reshape(-1, 1))
    cal_err(raw_data[window_size:data_num], preds)
    return preds
