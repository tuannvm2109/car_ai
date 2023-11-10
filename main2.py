import kwargs
import numpy as np
import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow_datasets.core import DatasetCollectionLoader
from tqdm import tqdm

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

if __name__ == '__main__':
    tfds.disable_progress_bar()

    train_ds, validation_ds = tfds.load(
        "cars196",
        split=["train+test[:50%]", 'test[50%:]'],
        as_supervised=True,  # Include labels
        download=True,
    )

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
    print(
        "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
    )
    # print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))

    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(train_ds.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title(int(label).__str__())
        plt.axis("off")

    size = (150, 150)

    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    # test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    batch_size = 32

    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    # test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    # x = base_model(x, training=False)
    # x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    # outputs = keras.layers.Dense(200, activation='softmax')(x)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # outputs1 = keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))(x)
    # outputs2 = keras.layers.MaxPooling2D(2, 2)(outputs1)
    # outputs3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(outputs2)
    # outputs4 = keras.layers.MaxPooling2D(2, 2)(outputs3)
    # outputs5 = keras.layers.Conv2D(64, (3, 3), activation='relu')(outputs4)
    outputs6 = keras.layers.Flatten()(x)
    outputs7 = keras.layers.Dense(64, activation='relu')(outputs6)
    outputs8 = keras.layers.Dense(196)(outputs7)
    model = keras.Model(inputs, outputs8)

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    epochs = 30
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    epochs = 10
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    model.save('train_car_model.keras')
