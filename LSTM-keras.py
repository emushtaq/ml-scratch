from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import LSTM
import numpy as np

# Data from exercise file

# Import dataset
with open("data/data.pkl", "rb") as fp:
    all_sequences = pickle.load(fp)

X = np.array([i[0] for i in all_sequences])
y = np.array([i[1] for i in all_sequences])


#split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

print "SHAPE " + str(np.shape(X))

# TODO: NEED to fix model initialisation (shape/dim)

# Create LSTM model
model = Sequential()
model.add(LSTM(64,  input_dim=8000, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=50, batch_size=10)
