import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

"""
What to do?
• Consider what activation function you want to use for the output layer.
• Report your accuracy, is this satisfactory? Why / why not?
• Plot the learning history from the history element.
"""
# Loads the data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Plots a single digit from the data
plt.imshow(train_data[1])
plt.show()

# Reshapes the data to work in a FFN
train_data = train_data.reshape((60000, 28*28))
test_data = test_data.reshape((10000, 28*28))

num_classes = len(np.unique(train_labels))

train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)