import tensorflow as tf
from tensorflow.keras import layers

class NeuralNetwork:
    def __init__(self):
        pass

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(784,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        return model
