import cv2
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.keras.preprocessing import image as keras_image_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing

def load_image(path, image_size = (331, 331), num_channels = 3, interpolation = 'bilinear', smart_resize=False):
  """Load an image from a path and resize it."""
  interpolation = image_preprocessing.get_interpolation(interpolation)
  img = io_ops.read_file(path)
  img = image_ops.decode_image(
      img, channels=num_channels, expand_animations=False)
  if smart_resize:
    img = keras_image_ops.smart_resize(img, image_size, interpolation=interpolation)
  else:
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
  img.set_shape((image_size[0], image_size[1], num_channels))
  return img
#load the model


IMG_HEIGHT, IMG_WIDTH = 331, 331
names = []
with open('labels.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        names.append(x)

model = load_model("models/model_comb")
img_path = 'sample/rottweiler.jpg' #'sample/pug.jpg'
img = load_image(img_path)
imgList = np.asarray([img])

pred = model.predict(imgList)
#display the image of dog
#cv2.imshow("Dog Breed", cv2.resize(cv2.imread(img_path,cv2.IMREAD_COLOR),((IMG_HEIGHT,IMG_WIDTH)))) 
#display the predicted breed of dog
pred_breed = names[np.argmax(pred)]
print("Predicted Breed for this Dog is :",pred_breed)