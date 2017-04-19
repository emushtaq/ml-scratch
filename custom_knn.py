# Euclidean distance
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

import random
# custom KNN classifier
class customKNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions = []

        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        # initialize the best distance with distance of the first point and store its index
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        # return the label of the closest datapoint (k is assumed to be 1)
        return self.Y_train[best_index]

# import dataset
from sklearn.datasets import load_iris

iris = load_iris()

# Correalted to the concept that the classifier is a function on the data( f(x) ) producing an output(y)
X = iris.data
Y = iris.target

# sklearn tool to split dataset. Half is used to train and the remaining half to test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)


# Commenting the sklearn KNN classifier and implementing a custom K nearest neighbors classifier
# from sklearn import neighbors
# knn_classifier = neighbors.KNeighborsClassifier()

# initialize the custom classifier
knn_classifier = customKNN()

knn_classifier.fit(X_train, Y_train)
knn_prediction = knn_classifier.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy: %f" % (accuracy_score(Y_test, knn_prediction)))

# So the idea is start with a model, and optimize it with the training data to minimize the error
# http://playground.tensorflow.org