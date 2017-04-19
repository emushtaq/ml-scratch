# training with image data. Classifier that classifies the types of flowers
# images from : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#2
# ~ about 100 images in each training directory


# Download the image dataset from : http://download.tensorflow.org/example_images/flower_photos.tgz

# Retrain the final layer of the "Inception" neural net with the downloaded training data with the command mentioned in the codelab : # In Docker
# python tensorflow/examples/image_retraining/retrain.py \
# --bottleneck_dir=/tf_files/bottlenecks \
# --how_many_training_steps 500 \
# --model_dir=/tf_files/inception \
# --output_graph=/tf_files/retrained_graph.pb \
# --output_labels=/tf_files/retrained_labels.txt \
# --image_dir /tf_files/flower_photos

# Use the TF wrapper learn to create a classifier for predicting any test flower image. The result should provide an image with confidence scores
