import tensorflow as tf


class Convolution(tf.keras.Model):
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides: int = 1,
                 batch_normalization: bool = True,
                 activation: bool = True
                 ):
        super(Convolution, self).__init__()

        self.sequential = tf.keras.Sequential()

        if kernel_size > 1:
            padding_size = kernel_size // 2
            self.sequential.add(tf.keras.layers.ZeroPadding2D((padding_size, padding_size)))

        self.sequential.add(tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides,
            padding='VALID', use_bias=not batch_normalization,
            kernel_initializer='he_normal'
        ))

        if batch_normalization:
            self.sequential.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))

        if activation:
            self.sequential.add(tf.keras.layers.ReLU())

    def call(self, x, **kwargs):
        return self.sequential(x)
