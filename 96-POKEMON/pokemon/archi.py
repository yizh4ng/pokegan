import tensorflow.keras as keras

from tframe.nets.net import Net


def disc_v1():
  net = Net('Dis')
  net.add(keras.layers.Conv2D(16, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(16, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(16, 3, 2))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(32, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(32, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(32, 3, 2))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(64, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(64, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(64, 3, 2))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Flatten())
  net.add(keras.layers.Dropout(0.5))
  net.add(keras.layers.Dense(2, activation='sigmoid'))
  return net

def disc_v2():
  net = Net('Dis')

  net.add(keras.layers.Conv2D(32, 7, 2))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(32, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(32, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(64, 3, 2))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(64, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(64, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(128, 3, 2))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(128, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(128, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(1024, 3, 1))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Flatten())
  net.add(keras.layers.Dropout(0.5))
  net.add(keras.layers.Dense(2, activation='sigmoid'))
  return net