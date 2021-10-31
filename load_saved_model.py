import os
import glob
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from model_v2 import MobileNetV2

model_tf = tf.keras.models.load_model("tf_CMN_b16e24_onnx", compile=False)
print("saved_model")
model_tf.summary()

im_height = 224
im_width = 224
num_classes = 7

feature = MobileNetV2(include_top=False)

model_ckpt = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(keepdims=True),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])
weights_path = './save_weights/CMNv2_b16_e24.ckpt'
assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
model_ckpt.load_weights(weights_path)
print("saved_weights")
model_ckpt.summary()