import torch.utils.hipify.constants
from torch.utils.data import Dataset
import copy


class ResidualChlorineDataset(Dataset):
    def __init__(self, seq, ws):
        self.seq = seq
        self.ws = ws
        self.x, self.y = self._get_db()

    def _get_db(self):
        x = []
        y = []
        l = len(self.seq)
        for i in range(l - self.ws):
            window = self.seq[i:i + self.ws]
            label = self.seq[i + self.ws:i + self.ws + 1]
            x.append(window)
            y.append(label)
        return x, y

    def __getitem__(self, index):
        # db_rec = copy.deepcopy(self.db[index])
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
