import os
import numpy as np
from model import model
import utils
import mathops
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch",5,"layers - 5")
flags.DEFINE_float("learning_rate",0.02,"Learning rate of the model")
flags.DEFINE_string("checkpoint_dir","checkpoint","Dir to save model checkpoints")
flags.DEFINE_string("train_bool",False,"set to true for training data")
flags.DEFINE_boolean("crop_bool", False, "True for training, False for testing [False]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images, default set to infinity")

flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "celebA", "Trained with celebA")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("visualize_bool", False, "True for visualizing, False for nothing [False]")

## still to define image size, batches, classes - if needed
FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    with tf.Session() as sess:
      
      op = model(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                 data_name=FLAGS.dataset, is_crop=FLAGS.crop_bool, checkpoint_dir=FLAGS.checkpoint_dir)
      if FLAGS.train_bool:
        op.train(FLAGS)
      else:
        op.load(FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()