import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def train_model(image_directory="images", training_epochs=1, fine_tune_epochs=1):
    train_dir = "train"
    validation_dir = "test"
    BATCH_SIZE = 32
    IMG_SIZE = (70, 70)
    total_epochs = training_epochs + fine_tune_epochs
    train_dataset = image_dataset_from_directory(
        train_dir, label_mode="categorical", shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    validation_dataset = image_dataset_from_directory(
        validation_dir, label_mode="categorical", shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    class_names = train_dataset.class_names
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.5),
        ]
    )
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 127.5, offset=-1)

    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(len(class_names))

    inputs = tf.keras.Input(shape=(70, 70, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    train_dataset = image_dataset_from_directory(
        train_dir, label_mode="categorical", shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    validation_dataset = image_dataset_from_directory(
        validation_dir, label_mode="categorical", shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    class_names = train_dataset.class_names

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    mdl = model.fit(train_dataset, epochs=total_epochs, validation_data=validation_dataset)

    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
        metrics=["accuracy"],
    )
    mdl_fine = model.fit(
        train_dataset, epochs=total_epochs, initial_epoch=mdl.epoch[-1], validation_data=validation_dataset
    )
    model.save("trained_model")

