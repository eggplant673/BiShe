import os
import numpy as np
import keras
from keras import models, layers, optimizers
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

mode = "train"
# mode = "test"

# CUB_200_2011 dataset
train_dir = "./data/cub-200-2011/train"
val_dir = "./data/cub-200-2011/val"
test_dir = "./data/cub-200-2011/test"
classes_count = 200

# Load pre-trained models
image_size = 224

history = None

if mode == "train":
    # VGG16 base
    vgg_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    base_model = vgg16.VGG16
    trainable_layers = 4

    base_model = base_model(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

    # Freeze all but the last 4 layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in base_model.layers:
        print(layer, layer.trainable)

    # Create our new model
    bird_model = models.Sequential()

    # Add the vgg convolutional base model
    bird_model.add(base_model)

    # Add new layers
    bird_model.add(layers.Flatten())
    bird_model.add(layers.Dense(1024, activation="relu"))
    bird_model.add(layers.Dropout(0.5))
    bird_model.add(layers.Dense(classes_count, activation="softmax"))

    # Show a summary of the model
    bird_model.summary()

    # Set up data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_batchsize = 20
    validation_batchsize = 5

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode="categorical"
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=validation_batchsize,
        class_mode="categorical",
        shuffle=False
    )

    # Set up early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto"
    )

    # Compile the model
    bird_model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = bird_model.fit_generator(
        train_generator,
        callbacks=[early_stop],
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_batchsize,
        verbose=1
    )

    # Save the model
    bird_model.save("bird_model_vgg16_224_cub-200-2011_last4.h5")

    exit(0)
elif mode == "test":
    bird_model = models.load_model("bird_model_vgg16_224_cub-200-2011_last4.h5")
    bird_model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    test_batchsize = 10

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=test_batchsize,
        class_mode="categorical"
    )

    history = bird_model.evaluate_generator(
        test_generator,
        steps=test_generator.samples / test_generator.batch_size,
        verbose=1
    )

    print(history)

    exit(0)



