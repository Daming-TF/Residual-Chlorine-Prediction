import os
import pandas
import numpy
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time
from torch.nn.functional import relu
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

gpu_enable = False
# 将excel数据读出，用values存储93*2的数据
residualChlorine = pandas.read_excel(r'E:\residual chlorine\HongmushanResidualChlorine.xlsx')
n = 80
# 把余氯数据读出，并拆分为80个训练集，13个测试集
residualChlorineTrain = residualChlorine.values[0:n, 1]
residualChlorineTest = residualChlorine.values[n:len(residualChlorine), 1]

# 将数据标准化转化为（-1,1）范围内
scaler = MinMaxScaler(feature_range=(-1, 1))
residualChlorineTrain_norm = scaler.fit_transform(residualChlorineTrain.reshape(-1, 1))    # 将数据变为列向量并归一化
residualChlorineTrain_norm_tenser = torch.FloatTensor(residualChlorineTrain_norm).view(-1)      # 将数组类型转化为张量并变化为行向量
residualChlorineTest_norm = scaler.fit_transform(residualChlorineTest.reshape(-1, 1))
residualChlorineTest_norm_tenser = torch.FloatTensor(residualChlorineTest_norm).view(-1)
if torch.cuda.is_available() and gpu_enable:
    residualChlorineTrain_norm_tenser = residualChlorineTrain_norm_tenser.cuda()
    residualChlorineTest_norm_tenser = residualChlorineTest_norm_tenser.cuda()

window_size = 5


def input_data(seq, ws):
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    return out


train_data = input_data(residualChlorineTrain_norm_tenser, window_size)


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


torch.manual_seed(101)      # 设置生成随机数的种子
model = CNNnetwork()
criterion = nn.MSELoss()
if torch.cuda.is_available() and gpu_enable:
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 2
model.train()
start_time = time.time()
for epoch in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        a = seq.reshape(1, 1, -1)       # {1*1*window_size}
        if torch.cuda.is_available() and gpu_enable:
            a = a.cuda()
            y_train = y_train.cuda()
        y_pred = model(a)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

trainNum = len(residualChlorineTrain_norm_tenser)
preds = residualChlorineTrain_norm_tenser[-window_size:].tolist()       # 获取训练数据序列最后5个数据作为模型预测的初始数据

model.eval()    # 不启用 Batch Normalization 和 Dropout


def mape(actual, pred):
    actual, pred = numpy.array(actual), numpy.array(pred)
    return numpy.mean(numpy.abs((actual - pred) / actual)) * 100


for i in range(trainNum):
    seq = torch.FloatTensor(preds[-window_size:])
    if torch.cuda.is_available() and gpu_enable:
        seq = seq.cuda()
    with torch.no_grad():
        preds.append(model(seq.reshape(1, 1, -1)).item())   # item()将零维tensor元素转换为浮点数

trainPredictions = scaler.inverse_transform(numpy.array(preds[window_size:]).reshape(-1, 1))    # 将标准化后的数据转换为原始数据。
MAE_train = mean_absolute_error(residualChlorineTrain, trainPredictions, multioutput='raw_values')
MSE_train = mean_squared_error(residualChlorineTrain, trainPredictions, multioutput='raw_values')
MAPE_train = mape(residualChlorineTrain, trainPredictions)

futureValues = len(residualChlorineTest_norm_tenser)

preds = residualChlorineTest_norm_tenser[-window_size:].tolist()

model.eval()

for i in range(futureValues):
    seq = torch.FloatTensor(preds[-window_size:])
    if torch.cuda.is_available() and gpu_enable:
        seq = seq.cuda()
    with torch.no_grad():
        preds.append(model(seq.reshape(1, 1, -1)).item())

testPredictions = scaler.inverse_transform(numpy.array(preds[window_size:]).reshape(-1, 1))
MAE_test = mean_absolute_error(residualChlorineTest, testPredictions, multioutput='raw_values')
MSE_test = mean_squared_error(residualChlorineTest, testPredictions, multioutput='raw_values')
MAPE_test = mape(residualChlorineTest, testPredictions)

pyplot.figure(figsize=(12, 4))
pyplot.grid(True)
pyplot.plot(residualChlorine['date'], residualChlorine['ResidualChlorine'])
x = residualChlorine.values[n:len(residualChlorine)+1,0]
# x = numpy.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]').astype('datetime64[D]')
pyplot.plot(x, testPredictions, color='red')
pyplot.show()
