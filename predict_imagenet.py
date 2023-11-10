import kwargs
import numpy as np
import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.applications.xception import decode_predictions
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow_datasets.core import DatasetCollectionLoader
from tqdm import tqdm


if __name__ == '__main__':
    # dimensions of our images
    img_width, img_height = 299, 299

    # load the model we saved
    model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(299, 299, 3),
        include_top=True,
    )

    img = tf.keras.utils.load_img(
        'image_test/car/4.jpeg', target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    print(predictions[0])

    print('Predicted:', decode_predictions(predictions, top=3)[0])

