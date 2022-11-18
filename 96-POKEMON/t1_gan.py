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
    
  def __call__(self, predictions):
    return self.function(predictions)
  
  def function(self, predictions):
    # return keras.losses.BinaryCrossentropy()(tf.ones_like(predictions), predictions)
    return keras.losses.BinaryCrossentropy()(
      tf.random.normal(tf.shape(predictions), 0.7, 1.2), predictions)

class discriminator_loss(Quantity):  
  def __init__(self):
    super(discriminator_loss, self).__init__('disloss', True)

  def __call__(self, real_output, fake_output):
    return self.function(real_output, fake_output)

  def function(self, real_output, fake_output):
    tf.random.uniform(tf.shape(real_output), 0.7, 1.2)
    real_loss = keras.losses.BinaryCrossentropy()(
      tf.random.normal(tf.shape(real_output), 0.7, 1.2) , real_output)
    fake_loss = keras.losses.BinaryCrossentropy()(
      tf.random.normal(tf.shape(fake_output), 0.0, 0.3), fake_output)

    # real_loss = keras.losses.BinaryCrossentropy()(tf.ones_like(real_output),
    #                                               real_output)
    # fake_loss = keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_output),
    #                                               fake_output)
    return real_loss + fake_loss

  
th.task_name ='Gan'
def generator():
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
  net.add(keras.layers.Flatten())
  net.add(keras.layers.Dropout(0.5))
  net.add(keras.layers.Dense(2,activation='sigmoid'))
  model = Model(loss=discriminator_loss(), metrics=None, net=net)
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
  th.save_cycle = 30
  th.load_model = False
  th.generator_input_shape = (100,)
  th.discriminator_input_shape = (96,96,3)
  th.non_train_input_shape = th.input_shape
  th.overwrite = True
  th.patience = 30
  th.rehearse = False
  th.train = True
  th.save_last_model = True

  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  main(None)
