# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np

# Data from exercise file

# Load the dataset
with open("data/LSTM_keras_data.pkl", "rb") as fp:
    all_sequences=pickle.load(fp, encoding='latin-1')
print('Dataset sample:', all_sequences[0],  ' having ', len(all_sequences), ' `entries')

# create dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)

look_back = 5

# split into train and test sets
train_size = int(len(all_sequences) * 0.80)
test_size = len(all_sequences) - train_size
train = all_sequences[0:train_size]
test = all_sequences[train_size:len(all_sequences)]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print('train shape: ', train.shape)
print('trainX shape: ', trainX.shape)
print('trainY shape: ', trainY.shape)

print('Creating and fitting the model...')
#%%time
# create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(LSTM(32,input_shape=(None, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=2, batch_size=batch_size, verbose=2)

print("Evaluating the model...")
#Evaluate
score, acc = model.evaluate(testX, testY, show_accuracy=True)

print('Test score:', score)
print('Test accuracy:', acc)