def rec_exp_info(args, mae, mse, mape):
    txt_path = args.exp_info_path
    with open(txt_path, 'r')as f:
        data = f.read()
        # print(data)
    m = args.model
    ws = args.window_size
    l = args.label_length
    info = f'Model:{m}\tWindow_size:{ws}\tLabel_length:{l}\tMAE:{mae}\tMSE:{mse}\tMAPE:{mape}\n'
    data += info
    with open(txt_path, 'w')as f:
        f.write(data)
