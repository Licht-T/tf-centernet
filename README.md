# tf-centernet
[![PyPI version](https://badge.fury.io/py/tf-centernet.svg)](https://badge.fury.io/py/tf-centernet)
![Upload Python Package](https://github.com/Licht-T/tf-centernet/workflows/Upload%20Python%20Package/badge.svg)

[CenterNet](https://arxiv.org/abs/1904.07850) implementation with Tensorflow 2.

## Install
```bash
pip instal tf-centernet
```

## Example
### CenterNet object detection
```python
import numpy as np
import PIL.Image
import centernet

# Default: num_classes=80
obj = centernet.ObjectDetection(num_classes=80)

# Default: weights_path=None
# num_classes=80 and weights_path=None: Pre-trained COCO model will be loaded.
# Otherwise: User-defined weight file will be loaded.
obj.load_weights(weights_path=None)

img = np.array(PIL.Image.open('./data/sf.jpg'))[..., ::-1]

# The image with predicted bounding-boxes is created if `debug=True`
boxes, classes, scores = obj.predict(img, debug=True)
```
![output](https://raw.githubusercontent.com/Licht-T/tf-centernet/master/data/output.png)

## TODO
* [x] Object detection
* [x] Pre-trained model for object detection
* [ ] Pose estimation
* [ ] Pre-trained model for pose estimation
* [ ] Training function and Loss definition
* [ ] Training data augmentation
