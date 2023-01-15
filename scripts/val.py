import torch
import numpy as np
from lib.utils.calculation_errors import cal_err


def val(args, model, data_num, data_type, scaler, init_seq, raw_data):
    if not isinstance(init_seq, list):
        init_seq = init_seq.tolist()
    preds = init_seq

    window_size = args.window_size

    for i in range(data_num):
        seq = torch.FloatTensor(preds[-window_size:])
        if torch.cuda.is_available() and args.gpu_enable:
            seq = seq.cuda()
        with torch.no_grad():
            preds.append(model(seq.reshape(1, 1, -1)).item())

    if data_type == "train":
        train_predictions = scaler.inverse_transform(np.array(preds[window_size:data_num]).reshape(-1, 1))
        cal_err(raw_data[window_size:data_num], train_predictions)
        return train_predictions

    elif data_type == "test":
        test_predictions = scaler.inverse_transform(np.array(preds[window_size:data_num]).reshape(-1, 1))
        cal_err(raw_data[window_size:data_num], test_predictions)
        return test_predictions
