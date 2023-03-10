from torch.utils.data import Dataset


class ResidualChlorineDataset(Dataset):
    def __init__(self, seq, ws, label_length):
        self.seq = seq
        self.ws = ws
        self.label_length = label_length
        self.x, self.y = self._get_db()

    def _get_db(self):
        x = []
        y = []
        l = len(self.seq)
        for i in range(l - self.ws - self.label_length + 1):
            window = self.seq[i:i + self.ws]
            label = self.seq[i + self.ws:i + self.ws + self.label_length]
            x.append(window)
            y.append(label)
        return x, y

    def __getitem__(self, index):
        # db_rec = copy.deepcopy(self.db[index])
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
