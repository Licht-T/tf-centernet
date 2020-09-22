import tensorflow as tf

from .convolution import Convolution


class CenterNetHeadPart(tf.keras.Model):
    def __init__(self, n: int):
        super(CenterNetHeadPart, self).__init__()

        self.conv1 = Convolution(256, 3, batch_normalization=False)
        self.conv2 = Convolution(n, 1, batch_normalization=False, activation=False)

    def call(self, inputs, training=None, mask=None):
        return self.conv2(self.conv1(inputs))


class CenterNetHead(tf.keras.Model):
    def __init__(self, num_classes: int):
        super(CenterNetHead, self).__init__()

        self.class_heatmap_predictor = CenterNetHeadPart(num_classes)
        self.offset_predictor = CenterNetHeadPart(2)
        self.wh_predictor = CenterNetHeadPart(2)

    def call(self, inputs, training=None, mask=None):
        return [
            tf.sigmoid(self.class_heatmap_predictor(inputs)),
            self.offset_predictor(inputs),
            self.wh_predictor(inputs)
        ]
