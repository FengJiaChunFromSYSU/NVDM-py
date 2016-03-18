import os
import numpy as np
import tensorflow as tf

from utils import pp
from models import NVDM, NASM
from reader import TextReader

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("dataset", "ptb", "The name of dataset [ptb]")
flags.DEFINE_string("model", "nvdm", "The name of model [nvdm, nasm]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_forward", False, "False for training, True for testing [False]")
FLAGS = flags.FLAGS

MODELS = {
  'nvdm': NVDM,
  'nasm': NASM,
}

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  data_path = "./data/%s" % FLAGS.dataset
  reader = TextReader(data_path, FLAGS.batch_size)

  with tf.Session() as sess:
    m = MODELS[FLAGS.model]
    model = m(sess, reader)

    if FLAGS.is_forward:
      model.load(FLAGS.checkpoint_dir)
    else:
      model.train(FLAGS)

  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

if __name__ == '__main__':
  tf.app.run()
