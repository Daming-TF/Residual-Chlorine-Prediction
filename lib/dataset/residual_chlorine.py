from torch.utils.data import Dataset

class ResidualChlorine(Dataset):
    def __init__(self):
        self.db = self.get_db()
    def __getitem__(self, item):
    def __len__(self):