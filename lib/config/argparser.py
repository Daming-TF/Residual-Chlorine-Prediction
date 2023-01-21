import argparse


def parser_args():
    """
    注意所有训练用的excel文件规定：
    1. 只有两列——————第一列名为：‘date’，第二列名为：‘ResidualChlorine’，
    2. 若用dividing_time方式划分训练集记得在划分时间位置添加一行并标注好划定关键字‘val’
    """
    parser = argparse.ArgumentParser(description="Train residual chlorine prediction")
    # parser.add_argument("--train_data_num", default=150)
    parser.add_argument("--window_size", default=10, type=int)
    parser.add_argument("--epochs", default=350, type=int)
    parser.add_argument("--gpu_enable", default=True)
    parser.add_argument("--seed", default=101, type=int)
    parser.add_argument("--batch_size", default=2650, type=int)
    # 数据集快捷关键字：ResidualChlorine-update       ResidualChlorine-test-debug     HongmushanResidualChlorine-update
    parser.add_argument("--data_path", default=r'E:\residual chlorine\ResidualChlorine-update.xlsx')
    parser.add_argument("--exp_info_path", default=r'E:\Project\ResidualChlorinePrediction\res\exp_rec.txt')
    parser.add_argument("--pic_save_dir", default=r'E:\Project\ResidualChlorinePrediction\res\exp')
    parser.add_argument("--dividing_time", default='val')
    parser.add_argument("--label_length", default=1, type=int)
    parser.add_argument("--model", default='base2')
    parser.add_argument("--pred_num", default=20, type=int)
    # parser.add_argument("--workers", default=8)
    args = parser.parse_args()

    return args
