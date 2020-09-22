import tensorflow as tf

from .hourglass import Hourglass104
from .head import CenterNetHead


class CenterNetModel(tf.keras.Model):
    def __init__(self, num_classes: int):
        super(CenterNetModel, self).__init__()

        self.hourglass104 = Hourglass104()

        self.head1 = CenterNetHead(num_classes)
        self.head2 = CenterNetHead(num_classes)

    def call(self, inputs, training=None, mask=None):
        hourglass2_output, hourglass1_output = self.hourglass104(inputs)

        output1 = self.head1(hourglass1_output)
        output2 = self.head2(hourglass2_output)

        return [output2, output1]
