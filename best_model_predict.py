import pandas as pd
import xlrd
import openpyxl
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import matplotlib as mpl
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks
import glob
import os.path
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, GRU
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K


def import_press(data_csv):
    csv_data = pd.read_csv(data_csv, header=0, usecols=[95,96])
    csv_data.loc[csv_data['flag1.4'] >= 12, 'local_pressure'] = np.nan
    csv_data['local_pressure'] *= 0.1
    data = csv_data['local_pressure'].values
    data.tolist()
    return data

def binning(dataset, bins):
    rmax = dataset.shape[0]
    bdata = np.zeros_like(dataset, dtype='float32')
    bdata = bdata[0:int(rmax/bins)]
    for r in np.arange(0, rmax-bins, bins):
        bdata[int(r/bins)] = np.mean(dataset[r:r+bins])
    return bdata

def create_dataset(dataset, history, future):
    X,Y = [], []
    step = 1
    nancount = 0
    for i in range(0, len(dataset) - history - future, step):
        if np.isnan(dataset[i : i + history]).any() or \
           np.isnan(dataset[i + history + future - 1]).any():
            nancount += 1
        else:
            X.append(dataset[i : i + history])
            Y.append(dataset[i + history + future - 1])
    X = np.reshape(X, [-1, history, 1])
    Y = np.reshape(Y, [-1, 1])
    return X, Y

def split_data(x, y, test_size):
    pos = round(len(x) * (1 - test_size))
    trainX, trainY = x[:pos], y[:pos]
    testX, testY   = x[pos:], y[pos:]
    return trainX, trainY, testX, testY

bins = 30
#24,96,336
history = 96
#2,3,7,19
future = 2
validation_size = 0.10


filename_list = sorted(glob.glob("csv/*.csv"))
testfilename_list = sorted(glob.glob("csv/test/*.csv"))
traindata = []
testdata = []

for filelist in filename_list:
    data = import_press(filelist)
    traindata.extend(data)

for filelist in testfilename_list:
    data = import_press(filelist)
    testdata.extend(data)

train_data = np.array(traindata)
test_data = np.array(testdata)
    
#binning
datasetb = binning(train_data, bins)
testdatab = binning(test_data, bins)

#create_dataset
trainX, trainY = create_dataset(datasetb, history, future)
testX, testY = create_dataset(testdatab, history, future)

ori_testY = testY

#normalization
def off(data):
    offs = np.min(data)
    da = data - offs
    return offs, da

def facter(data):
    fact = np.max(data)
    da = data / fact
    return fact, da

def normalize(data, offs, fact):
    da = data - offs
    dat = da / fact
    return dat

offs, off_trainX = off(trainX)
fact, trainX = facter(off_trainX)

testX = normalize(testX, offs, fact)
trainY = normalize(trainY, offs, fact)
testY = normalize(testY, offs, fact)

#create_validation
trainX, trainY, validationX, validationY = split_data(trainX, trainY, validation_size)


#learning
data_length = len(trainX)
num_input = history
num_output = 1
#int(future goto ni motometakazu/int(data_length/1000))
epochs = int(18000/int(data_length/1000))
batch_size = 1000
n_hide_1 = 100
n_hide_2 = 100
learning_rate = 0.001

#function loss
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

#optimizer
Adams = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

#NNcreate
model = Sequential()
#hide1
model.add(GRU(n_hide_1, activation='linear', input_shape=(num_input, 1), return_sequences=True))
#hide2
model.add(GRU(n_hide_2, activation='linear'))
#out
model.add(Dense(num_output, activation='linear'))
#compile
model.compile(loss=rmse,
              optimizer=Adams)

#load_model
f_model = './storage/model/atmo_by_atmo'

model.load_weights(os.path.join(f_model, '*.hdf5'))

test_score = model.evaluate(testX, testY, verbose=0)
test_loss = test_score*fact

print("test_loss: {:.5f}, real_test_loss: {:.5f}".format(test_score, test_loss))

#test_persistense
n_persistense = np.zeros_like(testY)
n_persistense_loss = np.zeros(len(testY)-1)
for i in range(0, len(testY)-future):
    n_persistense[i+future] = testY[i]
for i in range(1, len(testY)):
    n_persistense_loss[i-1] = np.sqrt(np.mean(np.square(n_persistense[i] - testY[i])))
persis = np.mean(n_persistense_loss)

persistense = np.zeros_like(ori_testY)
persistense_loss = np.zeros(len(ori_testY)-1)
for i in range(0, len(ori_testY)-future):
    persistense[i+future] = ori_testY[i]
for i in range(1, len(ori_testY)):
    persistense_loss[i-1] = np.sqrt(np.mean(np.square(persistense[i] - ori_testY[i])))
real_persis = np.mean(persistense_loss)

print("persistense_loss: {:.5f}, real_persis_loss: {:.5f}".format(persis, real_persis))
