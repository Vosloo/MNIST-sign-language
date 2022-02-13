import math
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":

    ROOT_PATH = Path(__name__).resolve().parent

    DATA_PATH = ROOT_PATH.joinpath("data")
    CHECKPOINT_PATH = ROOT_PATH.joinpath("clf_checkpoints")

    MNIST_TRAIN = DATA_PATH.joinpath("sign_mnist_train.csv")
    MNIST_TEST = DATA_PATH.joinpath("sign_mnist_test.csv")

    train_df = pd.read_csv(MNIST_TRAIN)
    test_df = pd.read_csv(MNIST_TEST)

    test = pd.read_csv(MNIST_TEST)
    y = test["label"]

    y_train, X_train_df = train_df["label"], train_df.drop("label", axis=1)
    y_test, X_test_df = test_df["label"], test_df.drop("label", axis=1)

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    x_train = X_train_df.values
    x_test = X_test_df.values

    # Normalize the data
    x_train = x_train / 255
    x_test = x_test / 255

    # Reshaping the to an image with single chanel
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
    )

    datagen.fit(x_train)

    model = Sequential()
    model.add(
        Conv2D(
            75,
            (3, 3),
            strides=1,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))

    model.add(Conv2D(50, (3, 3), padding="same", activation="relu"))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))

    model.add(Conv2D(25, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(24, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    checkpoint_name = "classificator-{epoch:03d}.ckpt"

    batch_size = 64
    epochs = 30

    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.00001
    )

    # Number of training examples / batch size = number of steps in each epoch
    save_freq = math.ceil(train_df.shape[0] / batch_size)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH.joinpath(checkpoint_name),
        verbose=1,
        save_weights_only=True,
        save_freq=5 * save_freq,  # Save weights every 5 epochs
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.005, patience=5, verbose=1, mode="auto"
    )

    model.save_weights(CHECKPOINT_PATH.joinpath(checkpoint_name.format(epoch=0)))

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[
            learning_rate_reduction,
            checkpoint_callback,
            early_stopping_callback,
        ],
    )

    model.save_weights(
        CHECKPOINT_PATH.joinpath(checkpoint_name.format(epoch=history.epoch[-1]))
    )

