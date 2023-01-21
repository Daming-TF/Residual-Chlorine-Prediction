import torch.nn as nn


class CNNnetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ws = args.window_size
        self.label_length = args.label_length
        self.conv_size = 3
        self.padding_size = (self.conv_size - 1) // 2
        self.module_1 = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, padding=1, bias=False), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 16, 3, 1, padding=1, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            # nn.Dropout(),
        )
        self.module_2 = nn.Sequential(
            nn.Conv1d(16, 16, self.conv_size, 1, padding=self.padding_size, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 16, self.conv_size, 1, padding=self.padding_size, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            # nn.Dropout(),
        )
        self.module_3 = nn.Sequential(
            nn.Conv1d(16, 32, self.conv_size, 1, padding=self.padding_size, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.module_4 = nn.Sequential(
            nn.Conv1d(32, 32, self.conv_size, 1, padding=self.padding_size, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, self.conv_size, 1, padding=self.padding_size, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            # nn.Dropout(),
        )
        self.module_5 = nn.Sequential(
            nn.Conv1d(32, 64, self.conv_size, 1, padding=self.padding_size, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.pridict = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*self.ws, 512), nn.ReLU(),
            # nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(512, self.label_length)
        )

    def forward(self, x):
        x1 = self.module_1(x)
        x2 = x1 + self.module_2(x1)
        x3 = self.module_3(x2)
        x4 = x3 + self.module_4(x3)
        x5 = self.module_5(x4)
        out = self.pridict(x5)
        return out


def get_net(args):
    return CNNnetwork(args)


if __name__ == '__main__':
    model = CNNnetwork()
    print("test@mingjiahui")
