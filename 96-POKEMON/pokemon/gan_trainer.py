from collections import OrderedDict
import numpy as np

import tensorflow as tf


from tframe import console, DataSet
from tframe.quantity import Quantity
from tframe.trainers.trainer import Trainer
from tensorflow.keras.optimizers import Adam, SGD, RMSprop



class GANTrainer(Trainer):
  def __init__(
      self,
      generator,
      discriminator,
      agent,
      config,
      training_set=None,
      valiation_set=None,
      test_set=None,
      probe=None
  ):
    super(GANTrainer, self).__init__(None, agent, config, training_set,
                                     valiation_set, test_set, probe)
    self.generator = generator
    self.discriminator = discriminator
    self.model = self.generator
    # self.disc_optimizer = SGD(0.01)
    # self.disc_optimizer = RMSprop(self.th.learning_rate)
    # self.gen_optimizer = RMSprop(self.th.learning_rate)
    self.disc_optimizer = Adam(self.th.learning_rate)
    self.gen_optimizer = Adam(self.th.learning_rate)

  def train(self):
    # :: Before training
    if self.th.overwrite:
      self.agent.clear_dirs()

    if self.th.load_model:
      self.generator.keras_model, self.counter = self.agent.load_model(
        'gen')
      self.discriminator.keras_model, self.counter = self.agent.load_model(
        'dis')
      console.show_status(
        'Model loaded from counter {}'.format(self.counter))
    else:
      self.generator.build(self.th.generator_input_shape)
      self.discriminator.build(self.th.discriminator_input_shape)
      tf.summary.trace_on(graph=True, profiler=True)

      # self.model.link(tf.random.uniform((self.th.batch_size, *self.th.input_shape)))
      # _ = self.model.keras_model(tf.keras.layers.Input(self.th.input_shape))
      @tf.function
      def predict(x, y):
        self.generator.keras_model(x)
        self.discriminator.keras_model(y)
        return None

      predict(tf.random.uniform(
        (self.th.batch_size, *self.th.generator_input_shape)),
        tf.random.uniform(
        (self.th.batch_size, *self.th.discriminator_input_shape))
      )
      self.agent.write_model_summary()
      tf.summary.trace_off()
      console.show_status('Model built.')
    self.generator.keras_model.summary()
    self.discriminator.keras_model.summary()
    self.agent.create_bash()

    # :: During training
    if self.th.rehearse:
      return

    rounds = self._outer_loop()

    # :: After training
    # self._end_training(rounds)
    # Put down key configurations to note
    self.agent.note.put_down_configs(self.th.key_options)
    # Export notes if necessary
    # Gather notes if necessary
    if self.th.gather_note:
      self.agent.gather_to_summary()

    if self.th.save_last_model:
      self.agent.save_model(self.discriminator.keras_model,
                            self.counter, 'dis')
      self.agent.save_model(self.generator.keras_model,
                            self.counter, 'gen')

    self.generator.keras_model, _ = self.agent.load_model('gen')
    self.discriminator.keras_model, _ = self.agent.load_model('dis')
    if self.th.probe:
      self.probe()


  # region : During training

  def _outer_loop(self):
    rnd = 0
    self.patenice = self.th.patience
    for _ in range(self.th.epoch):
      rnd += 1
      console.section('round {}:'.format(rnd))

      self._inner_loop(rnd)
      self.round += 1
      if self.th.probe and rnd % self.th.probe_cycle == 0:
        assert callable(self.probe)
        self.probe()

      if self.th.save_model:
        if self.th.save_cycle != 0 and rnd % self.th.save_cycle == 0:
          console.show_status('Saving the model to {}'.format(
            self.agent.ckpt_dir), symbol='[Saving]')
          self.agent.save_model(self.generator.keras_model,
                                self.counter, 'gen')
          self.agent.save_model(self.discriminator.keras_model,
                                self.counter, 'dis')
      if self.th._stop:
        break

    console.show_status('Training ends at round {}'.format(rnd),
                        symbol='[Patience]')

    if self.th.gather_note:
      self.agent.note.put_down_criterion('Total Parameters of Generator',
                                         self.generator.num_of_parameters)
      self.agent.note.put_down_criterion('Total Parameters of Discriminator',
                                         self.discriminator.num_of_parameters)
      self.agent.note.put_down_criterion('Total Iterations', self.counter)
      self.agent.note.put_down_criterion('Total Rounds', rnd)
    return rnd

  # region : Private Methods

  def _update_model_by_batch(self, data_batch):
    feature = data_batch.features
    loss_dict = {}
    for w in self.discriminator.keras_model.trainable_weights:
      w.assign(tf.clip_by_value(w, -0.1, 0.1))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.generator(tf.random.normal([self.th.batch_size,
                                            *self.th.generator_input_shape]))

      real_output = self.discriminator(feature)
      fake_output = self.discriminator(generated_images)
      generator_loss = self.generator.loss(fake_output)
      disc_real_loss = self.discriminator.loss[0](real_output)
      disc_fake_loss = self.discriminator.loss[1](fake_output)

      loss_dict[self.generator.loss] = generator_loss

      # loss_dict[self.discriminator.loss[0]] = disc_real_loss
      # loss_dict[self.discriminator.loss[1]] = disc_fake_loss
      dics_loss = disc_fake_loss + disc_real_loss
      loss_dict[Quantity('dis_loss', True)] = dics_loss
      loss_dict[Quantity('real_output', True)] = tf.reduce_mean(real_output)
      loss_dict[Quantity('fake_output', True)] = tf.reduce_mean(fake_output)

    gen_grads = gen_tape.gradient(generator_loss, self.generator.keras_model.trainable_variables)
    dis_grads = disc_tape.gradient(dics_loss, self.discriminator.keras_model.trainable_variables)
    self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.keras_model.trainable_variables))
    self.disc_optimizer.apply_gradients(zip(dis_grads, self.discriminator.keras_model.trainable_variables))

    return loss_dict

  def gp(self, real_img, fake_img):
    e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
    noise_img = e * real_img + (1. - e) * fake_img  # extend distribution space
    with tf.GradientTape() as tape:
      tape.watch(noise_img)
      o = self.discriminator.keras_model(noise_img)
    g = tape.gradient(o, noise_img)  # image gradients
    g_norm2 = tf.sqrt(
      tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
    gp = tf.square(g_norm2 - 1.)
    return tf.reduce_mean(gp)

  def update_discriminator_by_data_batch(self, data_batch):
    feature = data_batch.features
    loss_dict = {}
    # for w in self.discriminator.keras_model.trainable_weights:
    #   w.assign(tf.clip_by_value(w, -0.1, 0.1))

    with tf.GradientTape() as disc_tape:
      generated_images = self.generator(tf.random.normal([self.th.batch_size,
                                                          *self.th.generator_input_shape]))
      gp = self.gp(feature, generated_images)

      real_output = self.discriminator(feature)
      fake_output = self.discriminator(generated_images)
      disc_real_loss = self.discriminator.loss[0](real_output)
      disc_fake_loss = self.discriminator.loss[1](fake_output)

      dics_loss = disc_fake_loss + disc_real_loss + 10 * gp
      loss_dict[Quantity('dis_loss', True)] = dics_loss
      loss_dict[Quantity('real_output', True)] = tf.reduce_mean(real_output)
      loss_dict[Quantity('fake_output', True)] = tf.reduce_mean(fake_output)

    dis_grads = disc_tape.gradient(dics_loss, self.discriminator.keras_model.trainable_variables)
    self.disc_optimizer.apply_gradients(zip(dis_grads, self.discriminator.keras_model.trainable_variables))
    return loss_dict

  def update_generator_by_data_batch(self, data_batch):

    feature = data_batch.features
    loss_dict = {}
    with tf.GradientTape() as gen_tape:
      generated_images = self.generator(tf.random.normal([self.th.batch_size,
                                                          *self.th.generator_input_shape]))

      real_output = self.discriminator(feature)
      fake_output = self.discriminator(generated_images)
      generator_loss = self.generator.loss(fake_output)
      loss_dict[self.generator.loss] = generator_loss
      loss_dict[Quantity('real_output', True)] = tf.reduce_mean(real_output)
      loss_dict[Quantity('fake_output', True)] = tf.reduce_mean(fake_output)

    gen_grads = gen_tape.gradient(generator_loss, self.generator.keras_model.trainable_variables)
    self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.keras_model.trainable_variables))

    return loss_dict

  def _inner_loop(self, rnd):
    self.cursor = 0
    self._update_model_by_dataset(self.training_set, rnd)

    self.cursor += 1


  def _update_model_by_dataset(self, data_set, rnd):
    for i, batch in enumerate(data_set.gen_batches(
        self.th.batch_size, updates_per_round=self.th.updates_per_round,
        shuffle=self.th.shuffle, is_training=True)):
      self.cursor += 1
      self.counter += 1

      # Update model
      if self.cursor % self.th.discriminator_loop != 0:
        loss_dict = self.update_discriminator_by_data_batch(batch)
        if np.mod(self.counter - 1, self.th.print_cycle) == 0:
          self._print_progress(i, data_set._dynamic_round_len, rnd, loss_dict)
      else:
        loss_dict = self.update_generator_by_data_batch(batch)
        if np.mod(self.counter - 1, self.th.print_cycle) == 0:
          self._print_progress(i, data_set._dynamic_round_len, rnd, loss_dict)
