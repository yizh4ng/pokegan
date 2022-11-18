from tframe import console
from tframe.data.base_classes import ImageDataAgent
from tframe.utils import misc
from tframe.data.dataset import DataSet
from tframe.configs.trainerhub import TrainerHub

import gzip
import numpy as np
import os



class POKEMON(ImageDataAgent):

  @classmethod
  def load(cls, th):
    assert isinstance(th, TrainerHub)
    data_set = cls.load_as_tframe_data(th.data_dir)
    data_set.features = data_set.features[..., ::-1]
    data_set.features  = data_set.features / 128.0
    data_set.features = data_set.features - 1
    data_set.features = np.clip(data_set.features, -1, 1)
    # train_set, val_set, test_set = data_set.split([th.val_size, th.test_size], random=True)
    return data_set


  @classmethod
  def load_as_tframe_data(cls, data_dir):
    file_path = os.path.join(data_dir, 'data.tfd')
    if os.path.exists(file_path): return DataSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw data
    console.show_status('Trying to convert raw data to tframe DataSet ...')
    images= cls.load_as_numpy_arrays(data_dir)
    data_set = DataSet(features=images)
    # Show status
    console.show_status('Successfully converted {} samples'.format(
      data_set.size))
    # Save DataSet
    console.show_status('Saving data set ...')
    data_set.save(file_path)
    console.show_status('Data set saved to {}'.format(file_path))
    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    import cv2
    import numpy as np
    from roma import finder
    images = []
    data_paths = finder.walk(data_dir, type_filter='file', pattern='*.png')
    for data_path in data_paths:
      im_frame = cv2.imread(data_path)
      image = np.array(im_frame)
      if image.shape[0] != 96: continue
      images.append(image)
    return np.stack(images)



if __name__ == '__main__':
  from pm_core import th
  from lambo.gui.vinci.vinci import DaVinci
  from tframe.data.augment.img_aug import color_jitter
  from tframe import set_random_seed
  set_random_seed(1)
  th.data_dir = 'G:\projects\data\pokemon'
  data_set= POKEMON.load(th)
  # data_set.append_batch_preprocessor(color_jitter)
  data_set_gen = data_set.gen_batches(16)
  da = DaVinci()
  da.objects = (next(data_set_gen).features + 1) / 2
  da.add_plotter(da.imshow_pro)
  da.show()