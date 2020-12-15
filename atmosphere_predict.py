"""-----version-----
python-apt           1.6.5+ubuntu0.2
pandas               1.0.3
xlrd                 1.2.0
openpyxl             3.0.3
numpy                1.18.3
matplotlib           3.2.1
tensorflow           2.1.0
"""-----------------

import pandas as pd
import xlrd
import openpyxl
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import matplotlib as mpl
import glob
import os.path
import math
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, GRU
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K


def import_pressure(data_csv):
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
history = 24
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

#best_save
f_model = './storage/model'
cp = keras.callbacks.ModelCheckpoint(filepath = os.path.join(f_model, 'gru_model_ep{epoch:02d}-loss{loss:.4f}-vloss{val_loss:.4f}.hdf5'), save_best_only=True, period=10)

#learn_main
hist = model.fit(trainX, trainY,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(validationX, validationY),
                 callbacks=[cp])

predicted = model.predict(trainX)
val_pred = model.predict(validationX)
test_pred = model.predict(testX)

#test_evaluate
score = model.evaluate(testX, testY, verbose=0)
test_loss = score

#persistense_loss
persistense = np.zeros_like(trainY)
persistense_loss = np.zeros(len(trainY)-1)
val_persistense = np.zeros_like(validationY)
for i in range(0, len(trainY)-future):
    persistense[i+future] = trainY[i]
for i in range(0, len(validationY)-future):
    val_persistense[i+future] = validationY[i]
for i in range(1, len(trainY)):
    persistense_loss[i-1] = np.sqrt(np.mean(np.square(persistense[i] - trainY[i])))
persis = np.mean(persistense_loss)
persi = np.zeros_like(hist.history['loss'])
persi += persis
real_persis = persis*fact

print("persistense_loss: {:.4f}, real_persis_loss: {:.4f}".format(persis, real_persis))
#result
lastnumber = len(hist.history['loss'])-1
print("last_train_loss : {:.4f}, last_val_loss : {:.4f}"
      .format(hist.history['loss'][lastnumber], hist.history['val_loss'][lastnumber]))
print("test_loss: {:.4f}, real_test_loss: {:.4f}".format(test_loss, test_loss*fact))

#glaph
plt.subplot(3, 1, 1)
plt.rcParams["font.size"] = 20
plt.plot(hist.history['loss'], color = "blue", label = "training_loss")
plt.plot(hist.history['val_loss'], color = "green", label = "validation_loss")
plt.plot(persi, color = "red", label = "persistense_loss")
plt.title("epoch={}, batch_size={}, lerning_rate={}, hide_node={}, history={}".
          format(epochs, batch_size, learning_rate, n_hide_1, history))
plt.xlabel("epochs", size=20)
plt.ylabel("loss", size=20)
plt.ylim(0, 1)
plt.legend(fontsize=18,
           loc='best')
plt.tick_params(labelsize=20)

#minutes split
minutes = 3*48
predicted = predicted[len(predicted)-1-minutes:]
real = trainY[len(trainY)-1-minutes:]
persistense_cut = persistense[len(persistense)-1-minutes:]

plt.subplot(3, 1, 2)
plt.rcParams["font.size"] = 20
plt.plot(predicted*fact+offs, color="green", label="tra_pred")
plt.plot(real*fact+offs, color="blue", label="tra_real")
plt.plot(persistense_cut*fact+offs, color="red", label="tra_persis")
plt.xlabel("time[min]", size=20)
plt.ylabel("temperature[degrees]", size=20)
plt.legend(fontsize=18,
           loc='best')

#minutes split
val_pred = val_pred[len(val_pred)-1-minutes:]
val_real = validationY[len(validationY)-1-minutes:]
val_persistense_cut = val_persistense[len(val_persistense)-1-minutes:]

plt.subplot(3, 1, 3)
plt.rcParams["font.size"] = 20
plt.plot(val_pred*fact+offs, color="green", label="val_pred")
plt.plot(val_real*fact+offs, color="blue", label="val_real")
plt.plot(val_persistense_cut*fact+offs, color="red", label="val_persis")
plt.xlabel("time[min]", size=20)
plt.ylabel("temperature[degrees]", size=20)
plt.legend(fontsize=18,
           loc='best')
plt.show()
