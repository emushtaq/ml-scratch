# IRIS flower classification using a Decision tree. Features include length and width of sepals and petals

## GOALS
# - Import dataset
# - Train a classifier
# - Predict the label for a new flower
# - Optional (Copy pasted) visualisation of the tree
##

import numpy as np
from sklearn import tree
# Use the sklearn inbuilt dataset for iris
from sklearn.datasets import load_iris

# import the dataset
iris = load_iris()

# display the features and labels of the dataset
print("Features: ", iris.feature_names)
print("Labels: %s(0), %s(1), %s(2)" % (iris.target_names[0], iris.target_names[1], iris.target_names[2]))

# Seperate testing and training data from the dataset
test_index = [0,50,100,10,50,110]

# training data
train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis=0)

# testing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

# create and train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# predict test data
print("Test data labels are: ", test_target)
print("Predicted labels are: ", clf.predict(test_data))
