import numpy as np
import torch
import torch.nn as nn

a = np.arange(0, 5, 1).reshape(1, 1, -1)
a = torch.FloatTensor(a)
print(a)
conv1 = nn.Conv1d(1, 8, 2, 1)
b = conv1(a)
print(b)
