from sklearn import tree

# Hardcoding feature vector of the fruits. Using feature weight and texture (0 ~ bumpy, 1 ~ smooth)
features = [[140, 1], [150, 1], [160, 0], [170, 0], [180, 0]]

# Hardcoding the training labels (0 ~ apples, 1 ~ oranges)
labels = [0, 0, 1, 1, 1]

# Initializing a classifier
clf = tree.DecisionTreeClassifier()

# Training the classifier with training data
clf.fit(features, labels)

# Predicting using the classifier
input_weight = input("Enter the weight of the fruit: ")
input_texture = input("Enter 0 for bumpy texture and 1 for smooth texture: ")

# Function that predicts the type of fruit
def predict_fruit(weight, texture, classifier) :
    if classifier.predict([[int(input_weight), int(input_texture)]]) == 0:
        return "Apple"
    else: return "Orange"

print("According to your inputs the fruit could be an " + predict_fruit(input_weight, input_texture, clf))

