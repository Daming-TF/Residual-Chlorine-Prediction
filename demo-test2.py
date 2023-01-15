import os
import pandas
import numpy
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time
from torch.nn.functional import relu
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

residualChlorine = pandas.read_excel(r'E:\residual chlorine\HongmushanResidualChlorine.xlsx')
n = 80
window_size = 5
residualChlorineTrain = residualChlorine.values[0:n,1]
residualChlorineTest = residualChlorine.values[n-window_size : len(residualChlorine),1]

scaler = MinMaxScaler(feature_range=(-1, 1))
residualChlorineTrain_norm = scaler.fit_transform(residualChlorineTrain.reshape(-1, 1))
residualChlorineTrain_norm_tenser = torch.FloatTensor(residualChlorineTrain_norm).view(-1)
residualChlorineTest_norm = scaler.fit_transform(residualChlorineTest.reshape(-1, 1))
residualChlorineTest_norm_tenser = torch.FloatTensor(residualChlorineTest_norm).view(-1)
window_size = 5


def input_data(seq,ws):
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    return out
train_data = input_data(residualChlorineTrain_norm_tenser,window_size)

class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,8,2,1), nn.ReLU(),
            nn.Conv1d(8,16,2,1), nn.ReLU(),
            nn.Conv1d(16,32,2,1), nn.ReLU(),
            nn.Conv1d(32,64,2,1), nn.ReLU(),
            # nn.Dropout(),
            nn.Flatten()
        )
        self.pridict = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1)
        x = self.pridict(x)
        return x

torch.manual_seed(101)
model = CNNnetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 200
model.train()
start_time = time.time()
for epoch in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        a = seq.reshape(1,1,-1)
        y_pred = model(a)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

trainNum = len(residualChlorineTrain_norm_tenser)
# preds = residualChlorineTrain_norm_tenser[-window_size:].tolist()
preds = residualChlorineTrain_norm_tenser[0:window_size].tolist()

model.eval()

def mape(actual, pred):
    actual, pred = numpy.array(actual), numpy.array(pred)
    return numpy.mean(numpy.abs((actual - pred) / actual)) * 100

for i in range(trainNum):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        preds.append(model(seq.reshape(1,1,-1)).item())

trainPredictions = scaler.inverse_transform(numpy.array(preds[window_size:n]).reshape(-1, 1))

MAE_train = mean_absolute_error(residualChlorineTrain[window_size:n], trainPredictions, multioutput = 'raw_values')
MSE_train = mean_squared_error(residualChlorineTrain[window_size:n], trainPredictions, multioutput = 'raw_values')
MAPE_train = mape(residualChlorineTrain[window_size:n], trainPredictions)
# print(f"MAE[Train]: {MAE_train}\tMSE[Train]: {MSE_train}\tMAPE[Train]: {MAPE_train}")
print('{:40}{:<40}{:<40}'.format(f"MAE[Train]: {MAE_train}", f"MSE[Train]: {MSE_train}", f"MAPE[Train]: {MAPE_train}"))

futureValues = len(residualChlorineTest_norm_tenser)

preds = residualChlorineTest_norm_tenser[0:window_size].tolist()

model.eval()

for i in range(futureValues):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        preds.append(model(seq.reshape(1,1,-1)).item())

testPredictions = scaler.inverse_transform(numpy.array(preds[window_size:len(residualChlorineTest)]).reshape(-1, 1))
MAE_test = mean_absolute_error(residualChlorineTest[window_size:len(residualChlorineTest)], testPredictions, multioutput = 'raw_values')
MSE_test = mean_squared_error(residualChlorineTest[window_size:len(residualChlorineTest)], testPredictions, multioutput = 'raw_values')
MAPE_test = mape(residualChlorineTest[window_size:len(residualChlorineTest)], testPredictions)
# print(f"MAE[Test]: {MAE_test}\tMSE[Test]: {MSE_test}\tMAPE[Test]: {MAPE_test}")
print('{:<40}{:<40}{:<40}'.format(f"MAE[Test]: {MAE_test}", f"MSE[Test]: {MSE_test}", f"MAPE[Test]: {MAPE_test}"))

pyplot.figure(figsize=(12,4))
pyplot.grid(True)
pyplot.plot(residualChlorine['date'], residualChlorine['ResidualChlorine'])
x = residualChlorine.values[n:len(residualChlorine)+1,0]
# x = numpy.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]').astype('datetime64[D]')
pyplot.plot(x, testPredictions,color='red')
pyplot.show()