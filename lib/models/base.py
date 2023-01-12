import torch.nn as nn


class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, 2, 1), nn.ReLU(),
            nn.Conv1d(8, 16, 2, 1), nn.ReLU(),
            nn.Conv1d(16, 32, 2, 1), nn.ReLU(),
            nn.Conv1d(32, 64, 2, 1), nn.ReLU(),
            # nn.Dropout(),
            nn.Flatten()
        )
        self.pridict = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1)
        x = self.pridict(x)
        return x


def get_net():
    return CNNnetwork()


if __name__ == '__main__':
    model = CNNnetwork()
    print("test@mingjiahui")
