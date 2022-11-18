import sys, os
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here

DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)

sys.path.append(os.path.join(sys.path[0], 'roma'))
sys.path.append(os.path.join(sys.path[0], 'ditto'))

os.environ["CUDA_VISIBLE_DEVICES"]= "0" if not 'home' in sys.path[0] else "1"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0],
  [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)]
)
# =============================================================================
from tframe import console
from tframe.data.dataset import DataSet
from tframe import DefaultHub
from pokemon.util import probe
th = DefaultHub()

from tframe.core.agent import Agent
# from tframe.trainers.trainer import Trainer
from tframe.data.augment.img_aug import color_jitter
from pokemon.gan_trainer import GANTrainer as Trainer

from pokemon.pm_agent import POKEMON


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
# th.allow_growth = False
# th.gpu_memory_fraction = 0.40

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.print_cycle = 1
th.epoch = 1000
th.batch_size = 128
th.val_batch_size = 128
th.learning_rate = 0.0003
th.updates_per_round = None
th.validate_test_set = True
th.val_batch_size = 128
th.save_model = True
th.load_model = True
th.input_shape = (100)
th.overwrite = True
th.patience = 1
th.probe = True
th.probe_cycle = 30


def activate():

  # Build model
  generator = th.generator()
  discriminator = th.discriminator()
  mark ='lr{}-bs{}-ag{}'.format(th.learning_rate, th.batch_size,
                                          th.augmentation)
  agent = Agent(mark, th.task_name)
  agent.config_dir(__file__)
  if not 'home' in sys.path[0]:
    agent.data_dir = 'G:\projects\data\pokemon'

  th.data_dir = agent.data_dir

  # Load data
  dataset = POKEMON.load(th)
  # dataset.append_batch_preprocessor(color_jitter)

  # Train or evaluate
  if th.train:
    trainer = Trainer(generator, discriminator, agent, th, dataset)
    trainer.probe = lambda :probe(trainer)
    trainer.train()

  # End
  console.end()
