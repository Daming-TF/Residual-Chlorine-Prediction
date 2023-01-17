import argparse


def parser_args():
    """
    注意所有训练用的excel文件规定：
    1. 只有两列——————第一列名为：‘date’，第二列名为：‘ResidualChlorine’，
    2. 若用dividing_time方式划分训练集记得在划分时间位置添加一行并标注好划定关键字‘val’
    """
    parser = argparse.ArgumentParser(description="Train residual chlorine prediction")
    # parser.add_argument("--train_data_num", default=150)
    parser.add_argument("--window_size", default=5)
    parser.add_argument("--epochs", default=500)
    parser.add_argument("--gpu_enable", default=True)
    parser.add_argument("--seed", default=101)
    parser.add_argument("--batch_size", default=100)
    # ResidualChlorine-update       ResidualChlorine-test-debug     HongmushanResidualChlorine-update
    parser.add_argument("--data_path", default=r'E:\residual chlorine\HongmushanResidualChlorine-update.xlsx')
    parser.add_argument("--dividing_time", default='val')
    # parser.add_argument("--workers", default=8)
    args = parser.parse_args()
    return args
