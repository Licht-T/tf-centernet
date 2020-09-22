"""
MIT License

Copyright (c) 2020 Licht Takeuchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import tensorflow as tf
import numpy as np
import PIL.Image

from .model.object_detection import ObjectDetectionModel
from .util import image_util

VERSION = 'v1.0.5'


class ObjectDetection:
    def __init__(self, num_classes: int = 80):
        self.mean = np.array([[[0.408, 0.447, 0.470]]], dtype=np.float32)
        self.std = np.array([[[0.289, 0.274, 0.278]]], dtype=np.float32)
        self.k = 100
        self.score_threshold = 0.3

        self.num_classes = num_classes

        self.model = None

        self.init_model()

    def init_model(self):
        self.model = ObjectDetectionModel(self.num_classes)
        self.model(tf.keras.Input((512, 512, 3)))

    def load_weights(self, weights_path: str = None):
        if weights_path is None:
            base_url = f'https://github.com/Licht-T/tf-centernet/releases/download/{VERSION}'
            if self.num_classes == 80:
                weights_path = tf.keras.utils.get_file(
                    f'centernet_pretrained_coco_{VERSION}.h5',
                    f'{base_url}/centernet_pretrained_coco.h5',
                    cache_subdir='tf-centernet'
                )
            else:
                raise RuntimeError('weights_path should not be None.')

        self.model.load_weights(weights_path)

    def predict(self, img: np.ndarray, debug=False):
        orig_wh = np.array(img.shape[:2])[::-1]
        resize_factor = 512.0 / orig_wh.max()
        centering = (512.0 - orig_wh * resize_factor) / 2

        input_img = tf.image.resize_with_pad(img, 512, 512)
        input_img = (tf.dtypes.cast(input_img, tf.float32) / tf.constant(255, tf.float32) - self.mean) / self.std
        input_img = input_img[tf.newaxis, ...]

        predicted, _ = self.model(input_img)

        heatmap, offsets, whs = predicted

        heatmap = tf.dtypes.cast(heatmap == tf.nn.max_pool2d(heatmap, 3, 1, 'SAME'), tf.float32) * heatmap

        heatmap = np.squeeze(heatmap.numpy())
        offsets = np.squeeze(offsets.numpy())
        whs = np.squeeze(whs.numpy())

        idx = heatmap.flatten().argsort()[::-1][:self.k]
        scores = heatmap.flatten()[idx]
        idx = idx[scores > self.score_threshold]
        scores = scores[scores > self.score_threshold]

        rows, cols, classes = np.unravel_index(idx, heatmap.shape)

        xys = np.concatenate([cols[..., np.newaxis], rows[..., np.newaxis]], axis=-1) + offsets[rows, cols]
        boxes = np.concatenate([xys - whs[rows, cols]/2, xys + whs[rows, cols]/2], axis=1).reshape((-1, 2, 2))

        boxes = ((512 / heatmap.shape[0]) * boxes - centering) / resize_factor
        boxes = boxes.reshape((-1, 4))

        if debug:
            im = PIL.Image.fromarray(img[..., ::-1])
            im = image_util.draw_bounding_boxes(im, boxes, classes, scores)
            im.save('./output.png')

        return boxes, classes, scores
