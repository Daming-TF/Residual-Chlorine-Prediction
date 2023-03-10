import torch.nn as nn


class CNNnetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ws = args.window_size
        self.label_length = args.label_length
        self.module_1 = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, padding=1), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 16, 3, 1, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 3, 1, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, 1, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            # nn.Dropout(),
            nn.Flatten()
        )
        self.pridict = nn.Sequential(
            nn.Linear(64*self.ws, 1024), nn.ReLU(),
            # nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, self.label_length)
        )

    def forward(self, x):
        x = self.module_1(x)
        # x = x.view(-1)
        x = self.pridict(x)
        return x


def get_net(args):
    return CNNnetwork(args)


if __name__ == '__main__':
    model = CNNnetwork()
    print("test@mingjiahui")
