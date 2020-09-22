import tensorflow as tf

from .hourglass import Hourglass104
from .head import ObjectDetectionHead


class ObjectDetectionModel(tf.keras.Model):
    def __init__(self, num_classes: int):
        super(ObjectDetectionModel, self).__init__()

        self.hourglass104 = Hourglass104()

        self.head1 = ObjectDetectionHead(num_classes)
        self.head2 = ObjectDetectionHead(num_classes)

    def call(self, inputs, training=None, mask=None):
        hourglass2_output, hourglass1_output = self.hourglass104(inputs)

        output1 = self.head1(hourglass1_output)
        output2 = self.head2(hourglass2_output)

        return [output2, output1]
