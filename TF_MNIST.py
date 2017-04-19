# import tflearn wrapper

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# load the MNIST training dataset
mnist = learn.datasets.load_dataset('mnist')

# training dataset (55000 images)
data = mnist.train.images

# training labels
labels = np.asarray(mnist.train.labels, dtype=np.int32)

# test dataset (10000 images)
test_data = mnist.test.images

# test labels
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# use to restrict size
max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]

# function to display digits
def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    # reshaping required since the images are flattened
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)
    plt.show()

print("Number of features in 1 image: %d (logically the total pixel i.e., 28*28)" %(len(data[0])))

# Create a linear classifier using the tflearn wrapper
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(feature_columns=feature_columns, n_classes=10)
classifier.fit(data, labels, batch_size=100, steps=1000)

# Evaluating the classifier with the test dataset (Takes time. Be patient)
# classifier.evaluate(test_data, test_labels)
# print("Classifier accuracy: %f " %(classifier.evaluate(test_data, test_labels)["accuracy"]))

# Make a prediction
input_index = int(input("Enter the index of the test image"))
print("Predicted %d, Label: %d" % (classifier.predict(test_data[input_index]), test_labels[input_index]))
display(input_index)