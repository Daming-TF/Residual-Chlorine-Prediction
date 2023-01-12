import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="Train residual chlorine prediction")
    parser.add_argument("--train_data_num", default=80)
    parser.add_argument("--window_size", default=5)
    parser.add_argument("--epochs", default=2)
    parser.add_argument("--gpu_enable", default=True)
    parser.add_argument("--seed", default=101)
    args = parser.parse_args()
    return args
