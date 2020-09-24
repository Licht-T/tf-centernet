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
import PIL.Image
import PIL.ImageDraw
import PIL.ImageColor
import numpy as np


def draw_bounding_boxes(im: PIL.Image, bboxes: np.ndarray, classes: np.ndarray,
                        scores: np.ndarray) -> PIL.Image:
    im = im.copy()
    num_classes = len(set(classes))
    class_to_color_id = {cls: i for i, cls in enumerate(set(classes))}

    colors = [PIL.ImageColor.getrgb(f'hsv({int(360 * x / num_classes)},100%,100%)') for x in range(num_classes)]

    draw = PIL.ImageDraw.Draw(im)

    for bbox, cls, score in zip(bboxes, classes, scores):
        color = colors[class_to_color_id[cls]]
        draw.rectangle((*bbox.astype(np.int64),), outline=color)

        text = f'{cls}: {int(100 * score)}%'
        text_w, text_h = draw.textsize(text)
        draw.rectangle((bbox[0], bbox[1], bbox[0] + text_w, bbox[1] + text_h), fill=color, outline=color)
        draw.text((bbox[0], bbox[1]), text, fill=(0, 0, 0))

    return im


def draw_keypoints(im: PIL.Image, bboxes: np.ndarray, keypoints: np.ndarray,
                        scores: np.ndarray) -> PIL.Image:
    im = im.copy()
    num_joints = keypoints.shape[1]

    colors = [PIL.ImageColor.getrgb(f'hsv({int(360 * x / num_joints)},100%,100%)') for x in range(num_joints)]

    draw = PIL.ImageDraw.Draw(im)
    r = 5

    for joints in keypoints:
        for i, joint in enumerate(joints):
            color = colors[i]
            draw.ellipse((*(joint - r), *(joint + r)), fill=color, outline=color)

    # for bbox, cls, score in zip(bboxes, classes, scores):
    #     color = colors[class_to_color_id[cls]]
    #     draw.rectangle((*bbox.astype(np.int64),), outline=color)
    #
    #     text = f'{cls}: {int(100 * score)}%'
    #     text_w, text_h = draw.textsize(text)
    #     draw.rectangle((bbox[0], bbox[1], bbox[0] + text_w, bbox[1] + text_h), fill=color, outline=color)
    #     draw.text((bbox[0], bbox[1]), text, fill=(0, 0, 0))

    return im


def apply_exif_orientation(img: PIL.Image) -> PIL.Image:
    methods = {
        1: tuple(),
        2: (PIL.Image.FLIP_LEFT_RIGHT,),
        3: (PIL.Image.ROTATE_180,),
        4: (PIL.Image.FLIP_TOP_BOTTOM,),
        5: (PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.ROTATE_90),
        6: (PIL.Image.ROTATE_270,),
        7: (PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.ROTATE_270),
        8: (PIL.Image.ROTATE_90,),
    }

    exif = img._getexif()

    if exif is None:
        return img

    for method in methods[exif.get(0x112, 1)]:
        img = img.transpose(method)

    return img
