# import torch
# import torch.nn as nn
# import numpy as np
# a = np.arange(0, 10, 1).reshape(2, -1)
# a = torch.from_numpy(a).float()
# print(a)
# cov = nn.Conv1d(2, 8, 2, 1)
# flatten = nn.Flatten()
# x = cov(a)
# print(x.shape)
# y = flatten(x)
# print(y.shape)
import os
print(__file__)
a = os.path.dirname(os.path.abspath(__file__))
b = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(b)
