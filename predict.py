# import the necessary packages
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import argparse
from keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras

from keras.models import load_model

if __name__ == '__main__':
    # dimensions of our images
    img_width, img_height = 150, 150

    # load the model we saved
    model = load_model('train_car_model_09_11.keras')

    img = tf.keras.utils.load_img(
        'image_test/car/1.jpeg', target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    print(predictions[0])

    im_class = tf.argmax(predictions[0], axis=-1)
    print(im_class)
