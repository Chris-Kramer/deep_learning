import tensorflow as tf
from matplotlib import pyplot as plt

def plot_perf(history: "tf.model"):
    """_summary_
    
    Args:
        history (tf.model): _description_
    """
    plt.plot(range(10), history.history['loss'], '-', color='r', label='Training loss')
    plt.plot(range(10), history.history['val_loss'], '--', color='r', label='Validation loss')
    plt.plot(range(10), history.history['accuracy'], '-', color='b', label='Training accuracy')
    plt.plot(range(10), history.history['val_accuracy'], '--', color='b', label='Validation accuracy')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()