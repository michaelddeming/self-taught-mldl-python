

# import the MNIST dataset from keras.datasets and load the data
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print the training images 'shape' property (60000, 28, 28) -> len=60000 each frame or index is a multidimensional array of 28x28 (pixels)
print(train_images.shape)

#print the length of training lables, this should match the length of training images. the labels are the expected results of the training images.
print(len(train_labels))
print(train_labels)


# print the test images 'shape' property (10000, 28, 28) -> len=10000 each frame or index is a multidimensional array of 28x28 (pixels)
print(test_images.shape)

#print the length of test lables, this should match the length of test images. the labels are the expected results of the test images.
print(len(test_labels))
print(test_labels)


import keras
from keras import layers

# adding 2 layers to our model, these are Dense layers which are densely connected (also called fully connected) neural layers.

# the second layer (last) is a 10 way "softmax" classification layer which will return an array of 10 probability scores. Each score will be the probability that the current digit image belongs to one of our 10 digit classes.
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax"),
])


# add a loss function, optimizer, and metrics to monitor during training and testing. (do not need to know this yet...)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# preprocess the data or reshape the data into a type the that model is expecting and scaling the values to a interval [0, 1].

# previously the images were in a shape of (60000, 28, 28) of type uint8 w/ values in interval [0, 255]. We transform them into float32 array shape (600000, 28 * 28) with values between 0 and 1.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((60000, 28 * 28))
test_images = test_images.astype("float32") / 255

# train the model by calling Keras fit() we fit the model to its training data

model.fit(train_images, train_labels, epochs=5, batch_size=128)