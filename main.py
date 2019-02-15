from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import os


import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.engine import InputLayer
from tensorflow.python.keras.layers import Conv2D, UpSampling2D
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import save_model

import h5py

print(tf.__version__)


def main2():
    # Get images
    image = img_to_array(load_img('woman.png'))
    image = np.array(image, dtype=float)
    # Import map images into the lab colorspace
    X = rgb2lab(1.0 / 255 * image)[:, :, 0]
    Y = rgb2lab(1.0 / 255 * image)[:, :, 1:]
    Y = Y / 128
    X = X.reshape(1, 400, 400, 1)
    Y = Y.reshape(1, 400, 400, 2)
    # Building the neural network
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), input_shape=(None, None, 1), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    # Finish model
    model.compile(optimizer='rmsprop', loss='mse')
    # Train the neural network
    model.fit(x=X, y=Y, batch_size=1, epochs=3000)
    print(model.evaluate(X, Y, batch_size=1))
    # Output colorizations
    output = model.predict(X)
    output = output * 128
    canvas = np.zeros((400, 400, 3))
    canvas[:, :, 0] = X[0][:, :, 0]
    canvas[:, :, 1:] = output[0]
    plt.imsave("img_result.png", lab2rgb(canvas))
    plt.imsave("img_gray_scale.png", rgb2gray(lab2rgb(canvas)))


def main():
    # Get images
    X = []
    for filename in os.listdir('./color_images/Small/'):
        X.append(img_to_array(load_img('./color_images/Small/' + filename)))
        print(filename)
    X = np.array(X, dtype=float)
    # Set up training and test data
    split = int(0.95 * len(X))
    Xtrain = X[:split]
    Xtrain = 1.0 / 255 * Xtrain
    # Design the neural network
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), input_shape=(None, None, 1), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    # Finish model
    model.compile(optimizer='rmsprop', loss='mse')
    # Image transformer
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
    # Generate training data
    batch_size = 50

    def image_a_b_gen(batch_size):
        for batch in datagen.flow(Xtrain, batch_size=batch_size):
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:, :, :, 0]
            Y_batch = lab_batch[:, :, :, 1:] / 128
            yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)

    # Train model
    TensorBoard(log_dir='/output')
    model.fit_generator(image_a_b_gen(batch_size), steps_per_epoch=100, epochs=5)
    # Test images
    Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
    Xtest = Xtest.reshape(Xtest.shape + (1,))
    Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
    Ytest = Ytest / 128
    print(model.evaluate(Xtest, Ytest, batch_size=batch_size))
    # Load black and white images
    color_me = []
    for filename in os.listdir('./gray_images/Test/'):
        color_me.append(img_to_array(load_img('./gray_images/Test/' + filename)))
    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape + (1,))
    model.save('./result/network.h5')
    del model

def testNet(color_me):
    # Test model
    
    model = load_model('./result/network.h5')
    
    
    color_me = []
    for filename in os.listdir('./gray_images/Test/'):
        color_me.append(img_to_array(load_img('./gray_images/Test/' + filename)))
    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape + (1,))
    
    
    output = model.predict(color_me)
    output = output * 128
    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((32, 32, 3))
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        plt.imsave("./result/img_" + str(i) + ".png", lab2rgb(cur))


if __name__ == '__main__':
    #main()
    testNet('./gray_images/Test/gray_image7.png')
