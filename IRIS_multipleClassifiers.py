# import dataset
from sklearn.datasets import load_iris

iris = load_iris()

# Correalted to the concept that the classifier is a function on the data( f(x) ) producing an output(y)
X = iris.data
Y = iris.target

# sklearn tool to split dataset. Half is used to train and the remaining half to test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

# Decision tree classifier
from sklearn import tree
tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(X_train, Y_train)
tree_prediction = tree_classifier.predict(X_test)

# K nearest neighbors classifier
from sklearn import neighbors
knn_classifier = neighbors.KNeighborsClassifier()
knn_classifier.fit(X_train, Y_train)
knn_prediction = knn_classifier.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print("TREE: %f \n KNN: %f" % (accuracy_score(Y_test, tree_prediction), accuracy_score(Y_test, knn_prediction)))

# So the idea is start with a model, and optimize it with the training data to minimize the error
# http://playground.tensorflow.org