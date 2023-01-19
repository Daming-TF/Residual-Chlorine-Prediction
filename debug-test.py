import numpy as np
import torch
import torch.nn as nn

# a = np.arange(0, 5, 1).reshape(1, 1, -1)
# a = torch.FloatTensor(a)
# print(a)
# conv1 = nn.Conv1d(1, 8, 3, 1, padding=1)
# b = conv1(a)
# print(b)
out = [12, 12]
label_length = 10
a_tensor = torch.FloatTensor(np.arange(0, label_length, 1)).contiguous().view(1, -1)[0]
a_np = a_tensor.numpy().reshape(-1, 1)
a_list = a_tensor.tolist()
print(a_list)
