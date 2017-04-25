from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np

# Load the dataset
with open("data/LSTM_keras_data.pkl", "rb") as fp:
    all_sequences=pickle.load(fp, encoding='latin-1')
print('Dataset sample:', all_sequences[0][0],  ' having ', len(all_sequences), ' `entries')

X = np.array([i[0] for i in all_sequences], dtype=float)
y = np.array([i[1] for i in all_sequences], dtype=float)

#split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

print('old SHAPEs: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train = X_train.reshape((1, len(X_train), 20))
X_test = X_test.reshape((1, len(X_test), 20))
y_test = y_test.reshape((1, len(y_test), 1))
y_train = y_train.reshape((1, len(y_train), 1))

print('new SHAPEs: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print("Creating model...")
# Create LSTM model
model = Sequential()
model.add(LSTM(20, input_shape=(None, 20)))
model.add(Dense(20))

print("Compiling the model...")
# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])


print("Fitting the data...")
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=10, batch_size=128)