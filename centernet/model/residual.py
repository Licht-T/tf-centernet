import tensorflow as tf

from .convolution import Convolution


class ResidualBlock(tf.keras.Model):
    def __init__(self, input_channels: int, output_channels: int, strides=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = Convolution(output_channels, 3, strides)

        self.conv2 = Convolution(output_channels, 3, activation=False)

        self.shortcut = tf.keras.Sequential()

        if strides > 1 or input_channels != output_channels:
            self.shortcut.add(Convolution(output_channels, 1, strides, activation=False))

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)

        y = self.shortcut(inputs)

        return tf.keras.activations.relu(x + y)
