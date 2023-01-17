import torch
import numpy as np
from lib.utils.calculation_errors import cal_err


def val(args, model, data_num, scaler, init_seq, raw_data):
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
