import numpy as np
import base64
import cv2
from tensorflow.keras.models import load_model
from tensorflow.python.ops import image_ops
from tensorflow.python.keras.preprocessing import image as keras_image_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing

labels = []

model = None

def process_image(img, image_size = (331, 331), num_channels = 3, interpolation = 'bilinear', smart_resize=False):
  """Load an image from a path and resize it."""
  interpolation = image_preprocessing.get_interpolation(interpolation)
  #img = image_ops.decode_image(img, channels=num_channels, expand_animations=False)
  #if smart_resize:
  #  img = keras_image_ops.smart_resize(img, image_size, interpolation=interpolation)
  #else:
  img = image_ops.resize_images_v2(img, image_size, method=interpolation)
  img.set_shape((image_size[0], image_size[1], num_channels))
  return img

def classify_image(image_base64_data, file_path=None):

    img = get_image(file_path, image_base64_data)
    print(img.shape)
    img = process_image(img)
    imgList = np.asarray([img])
    global model, labels
    #load_saved_artifacts()
    prediction = model.predict(imgList)

    idx = (-prediction).argsort()[0][0:3]
    class_probs = []
    class_names = []
    for x in idx:
        class_probs.append(np.around(prediction[0][x]*100,15))
        class_names.append(labels[x])
    #print(class_probs)
    result = {
        'bool' : 1,
        'class': labels[np.argmax(prediction)],
        'class_names': class_names,
        'class_probability': class_probs
    }
    alt = {'bool' : 1, 'class': labels[np.argmax(prediction)]}

    return result

def load_saved_artifacts():
    #print("loading saved artifacts...start")
    global model
    global labels
    labels = []
    with open("./artifacts/labels.txt", "r") as f:
        for line in f:
        # remove linebreak from a current name
        # linebreak is the last character of each line
            x = line[:-1]
        # add current item to the list
            labels.append(x)
    model = load_model("./artifacts/model")
    #print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_image(image_path, image_base64_data):
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    return img

def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()