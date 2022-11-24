import numpy as np
from tframe import console


def probe(trainer):
  targets = trainer.model(np.random.normal(size=(4,100)))
  for i, target in enumerate(targets):
    trainer.agent.save_figure((target + 1)/2,
                              'c{}-{}'.format(trainer.counter, i), cb=False,
                              axis_off=True)
  console.show_status(
    'Results saved to {}.'.format(trainer.agent.snapshot_dir),
    symbol='[Probe]')
  return