import sys
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data
import time
import os

search_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(search_path)

from lib.config import argparser
import lib.models
from lib.dataset import ResidualChlorineDataset, data_seg1, data_seg2
from val import val, val_once
from lib.utils import plot_res, init_seeds


def main():
    args = argparser.parser_args()
    window_size = args.window_size
    log_writer = SummaryWriter()

    # Step1：把余氯数据读出，并拆分成训练集和测试集
    res_cl, scaler, res_cl_train_norm, res_cl_test_norm, dividing_time_index = data_seg2(args)

    # Step2：网络设置（dataset，model，loss，opt）
    seq = res_cl_train_norm
    init_seeds(args.seed)
    train_dataset = ResidualChlorineDataset(seq, window_size, args.label_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.workers,
    )

    model = getattr(lib.models, args.model).get_net(args)
    criterion = nn.MSELoss()
    if torch.cuda.is_available() and args.gpu_enable:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Step3：网络训练
    model.train()
    start_time = time.time()
    for epoch in range(args.epochs):
        for i, (seq, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            a = seq.reshape(-1, 1, window_size)       # {batch_size， 数据维度（余氯数据是一维的）， 窗口长度}
            if torch.cuda.is_available() and args.gpu_enable:
                a = a.cuda()
                y_train = y_train.cuda()
            y_pred = model(a)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

        log_writer.add_scalar('loss/train', float(loss), epoch)
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    # Step4：模型验证
    model.eval()    # 不启用 Batch Normalization 和 Dropout
    # train_pred, res_cl_train = val(args, model, scaler, res_cl_train_norm)
    test_pred, res_cl_test = val(args, model, scaler, res_cl_test_norm)
    once_pred = val_once(args, model, scaler, res_cl_test_norm)

    # Step5:可视化预测结果
    info = '{:20}{:<20}{:<20}'.format(f'model:{args.model}', f'window_size:{args.window_size}',
                                      f'label_length:{args.label_length}')
    print(f"\033[1;33;40m {info} \033[0m]")
    # plot_res(args.window_size, train_pred, test_pred, res_cl_train, res_cl_test)
    # plot2(args.label_length, args.window_size, once_pred, test_pred, res_cl_test)
    plot_res(args.label_length, args.window_size, once_pred, size=(1, 2),
             train_pred=None, res_cl_train=None,
             test_pred=test_pred, res_cl_test=res_cl_test)


if __name__ == '__main__':
    main()
