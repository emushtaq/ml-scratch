# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np

# Data from exercise file

# Load the dataset
with open("data/LSTM_keras_data.pkl", "rb") as fp:
    all_sequences=pickle.load(fp, encoding='latin-1')
print('Dataset sample:', all_sequences[0],  ' having ', len(all_sequences), ' `entries')

X = np.array([i[0] for i in all_sequences])
y = np.array([i[1] for i in all_sequences])

print('Sample dataset: X', len(X), " Y", len(y))
print('X[0]:', X[0])
print('y[0]:', y[0])

#split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

print('Sample dataset(train): X', len(X_train), " Y", len(y_train))
print('X_train[0]:', X_train[0])
print('y_train[0]:', y_train[0])

X_train = X_train.reshape(8000, 1, 20)
y_train = y_train.reshape(8000, 1, 1)

print('Sample dataset reshaped(train): X', len(X_train), " Y", len(y_train))
print('X_train[0]:', X_train[0])
print('y_train[0]:', y_train[0])

print("SHAPE of training data" + str(np.shape(X_train)))

print("Creating model...")
# Create LSTM model
model = Sequential()
#model.add(LSTM(20, input_dim=20, input_length=8000, return_sequences=True))
model.add(LSTM(20, input_shape=(1, 20), return_sequences=True))
#model.add(Dense(2))

print("Compiling the model...")
# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

print("Fitting the data...")
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=1)

print("Evaluating the model...")
#Evaluate
score, acc = model.evaluate(X_test, y_test, show_accuracy=True)

print('Test score:', score)
print('Test accuracy:', acc)