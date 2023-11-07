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

    train_ds, validation_ds, test_ds = tfds.load(
        "cars196",
        # Reserve 10% for validation and 10% for test
        split=["train", 'test[:70%]', 'test[70%:]'],
        as_supervised=True,  # Include labels
        download=True,
    )

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
    print(
        "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
    )
    print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))

    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(train_ds.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title(int(label).__str__())
        plt.axis("off")

    size = (150, 150)

    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    batch_size = 32

    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

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
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    epochs = 20
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(1e-5),  # Low learning rate
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    epochs = 10
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    model.save('train_model.keras')
