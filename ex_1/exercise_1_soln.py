
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

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

# Exercise 1

from keras.layers import Dense
from keras.models import Sequential

# Softmax for multi-neuron output layers (classification with multiple options)
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(28*28,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_data=(test_data, test_labels))

# Exercise 2
print('Model accuracy: ' + str(model.evaluate(test_data, test_labels)[1]))

# Exercise 3
plt.plot(range(10), history.history['loss'], '-', color='r', label='Training loss')
plt.plot(range(10), history.history['val_loss'], '--', color='r', label='Validation loss')
plt.plot(range(10), history.history['accuracy'], '-', color='b', label='Training accuracy')
plt.plot(range(10), history.history['val_accuracy'], '--', color='b', label='Validation accuracy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()