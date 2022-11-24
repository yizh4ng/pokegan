import tensorflow.keras as keras
import tensorflow as tf
import pm_core as core

from tframe import console
from pm_core import th
from tframe.nets.image2scalar.resnet import ResNet
from tframe.nets.net import Net
from tframe.models.model import Model
from tframe.quantity import *
from tframe.quantity import Quantity






# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
class generator_loss(Quantity):
  def __init__(self):
    super(generator_loss, self).__init__('genloss', True)
    
  def __call__(self, fake_output):
    return self.function(fake_output)
  
  def function(self, fake_output):
    # return keras.losses.BinaryCrossentropy()(tf.ones_like(fake_output), fake_output)
    # return keras.losses.BinaryCrossentropy()(
    #   tf.clip_by_value(tf.random.normal(tf.shape(fake_output), 0.75, 1),0, 999),
    #   tf.clip_by_value(fake_output, 0.0001, 0.9999))
    # return keras.losses.BinaryCrossentropy()(
    #   tf.random.normal(tf.shape(fake_output), 0.7, 1.2), fake_output)
    return -tf.reduce_mean(fake_output)

class discriminator_loss(Quantity):
  def __init__(self):
    super(discriminator_loss, self).__init__('disloss', True)

  def __call__(self, real_output, fake_output):
    return self.function(real_output, fake_output)

  def function(self, real_output, fake_output):
    # tf.random.uniform(tf.shape(real_output), 0.7, 1)
    real_loss = keras.losses.BinaryCrossentropy()(
      tf.random.normal(tf.shape(real_output), 0.7, 0.9999) , tf.clip_by_value(real_output, 0.0001, 0.9999))
    fake_loss = keras.losses.BinaryCrossentropy()(
      tf.random.normal(tf.shape(fake_output), 0.0001, 0.3), tf.clip_by_value(fake_output, 0.0001, 0.9999))
    return real_loss + fake_loss


class disc_real_loss(Quantity):
  def __init__(self):
    super(disc_real_loss, self).__init__('dis_real_loss', True)

  def __call__(self, real_output):
    return self.function(real_output)

  def function(self, real_output):
    # real_loss = keras.losses.BinaryCrossentropy()(
    #   tf.clip_by_value(tf.random.normal(tf.shape(real_output), 0.7, 1) ,0,1),
    #   tf.clip_by_value(real_output, 0.0001, 0.9999))
    # real_loss = keras.losses.BinaryCrossentropy()(
    #   tf.random.normal(tf.shape(real_output), 0.7, 1.2) , real_output)
    real_loss = -tf.reduce_mean(real_output)
    return real_loss

class disc_fake_loss(Quantity):
  def __init__(self):
    super(disc_fake_loss, self).__init__('dis_fake_loss', True)

  def __call__(self,  fake_output):
    return self.function(fake_output)

  def function(self, fake_output):
    # fake_loss = keras.losses.BinaryCrossentropy()(
    #   tf.clip_by_value(tf.random.normal(tf.shape(fake_output), 0.15, 0.3),0, 1),
    #   tf.clip_by_value(fake_output, 0.0001, 0.9999))
    # fake_loss = keras.losses.BinaryCrossentropy()(
    #   tf.random.normal(tf.shape(fake_output), 0.0, 0.3), fake_output)
    fake_loss = tf.reduce_mean(fake_output)
    return fake_loss

th.task_name ='Gan'
def generator():
  # net = Net(name='gen')
  # net.add(keras.layers.Dense(24 * 24 * 128))
  # net.add(keras.layers.Reshape([24, 24, 128]))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Convolution2DTranspose(64, 3, 1, 'same' ))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Dropout(0.2))
  # net.add(keras.layers.Convolution2DTranspose(64, 3, 1, 'same' ))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Dropout(0.2))
  # net.add(keras.layers.Convolution2DTranspose(64, 3, 2, 'same'))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Dropout(0.2))
  # net.add(keras.layers.Convolution2DTranspose(32, 3, 1, 'same'))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Dropout(0.2))
  # net.add(keras.layers.Convolution2DTranspose(32, 3, 1, 'same'))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Dropout(0.2))
  # net.add(keras.layers.Convolution2DTranspose(32, 3, 2, 'same'))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Convolution2DTranspose(16, 3, 1, 'same'))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Dropout(0.2))
  # net.add(keras.layers.Convolution2DTranspose(16, 3, 1, 'same'))
  # net.add(keras.layers.BatchNormalization())
  # net.add(keras.layers.LeakyReLU())
  # net.add(keras.layers.Dropout(0.2))
  # net.add(keras.layers.Conv2D(3, 1, 1, 'same'))
  # net.add(keras.layers.Activation('tanh'))
  # model = Model(loss=generator_loss(), metrics=None, net=net)

  net = Net(name='gen')
  net.add(keras.layers.Dense(24 * 24 * 128))
  net.add(keras.layers.Reshape([24, 24, 128]))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Convolution2DTranspose(64, 3, 1, 'same'))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Dropout(0.5))
  net.add(keras.layers.Convolution2DTranspose(64, 3, 2, 'same'))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Dropout(0.5))
  net.add(keras.layers.Convolution2DTranspose(32, 3, 1, 'same'))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Dropout(0.5))
  net.add(keras.layers.Convolution2DTranspose(32, 3, 2, 'same'))
  net.add(keras.layers.BatchNormalization())
  net.add(keras.layers.LeakyReLU())
  net.add(keras.layers.Conv2D(3, 1, 1, 'same'))
  net.add(keras.layers.Activation('tanh'))
  model = Model(loss=generator_loss(), metrics=None, net=net)
  return model

def discriminator():
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
  # net.add(ResNet(filters=16, block_repetitions=(1, 2, 1),
  #                activation=keras.layers.LeakyReLU(0.2),
  #                bn=True
  #                ))
  # net.add(keras.layers.GlobalAvgPool2D())
  net.add(keras.layers.Flatten())
  net.add(keras.layers.Dropout(0.5))

  net.add(keras.layers.Dense(1))
  # net.add(keras.layers.Activation('sigmoid'))
  model = Model(loss=[disc_real_loss(), disc_fake_loss()], metrics=None, net=net)
  return model

th.generator = generator
th.discriminator = discriminator
def main(_):
  console.start('{} on Pokemon task'.format(th.task_name))

  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  pass

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  pass
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  pass
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.print_cycle = 3
  th.epoch = 99999999999999999
  th.batch_size = 128
  th.learning_rate = 0.0001
  th.validate_test_set = True
  th.save_model = True
  th.save_cycle = 10
  th.probe_cycle = 10
  th.load_model = False
  th.generator_input_shape = (100,)
  th.discriminator_input_shape = (96,96,3)
  th.non_train_input_shape = th.input_shape
  th.overwrite = True
  th.patience = 30
  th.rehearse = False
  th.train = True
  th.save_last_model = True
  th.discriminator_loop = 5
  th.shuffle = True
  th.updates_per_round = 60

  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  main(None)
