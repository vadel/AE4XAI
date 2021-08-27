import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops



def load_image_into_numpy_array(path, img_width=None, img_height=None):
    image = None
    if(path.startswith('http')):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image  = Image.open(path)
        image  = np.array(image)

    if img_width is None and img_height is None: 
        (im_width, im_height) = image.size
        out_img = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)
    else: 
        image = tf.image.resize(image, size=(img_height, img_width))
        out_img = np.array(image).reshape((1, img_height, img_height, 3)).astype(np.uint8)
    return out_img


model_display_name = "Mask R-CNN Inception ResNet V2 1024x1024"; img_width,img_height=1024,1024
model_handle = "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"

print('Selected model:'+ model_display_name)
print('Model Handle at TensorFlow Hub: {}'.format(model_handle))

print('loading model...')
hub_model = hub.load(model_handle)
print('model loaded!')


def segment_image(image_path):

    flip_image_horizontally = False
    convert_image_to_grayscale = False

    image_np = load_image_into_numpy_array(image_path, img_height, img_width)

    # Flip horizontally
    if(flip_image_horizontally):
        image_np[0] = np.fliplr(image_np[0]).copy()

    # Convert image to grayscale
    if(convert_image_to_grayscale):
        image_np[0] = np.tile(np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    results = hub_model(image_np)
    result = {key:value.numpy() for key,value in results.items()}
    
    if 'detection_masks' in result:
        # we need to convert np.arrays to tensors
        detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
        detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])

        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, 
                                                                            detection_boxes,
                                                                            image_np.shape[1],
                                                                            image_np.shape[2])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
        result['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return result
