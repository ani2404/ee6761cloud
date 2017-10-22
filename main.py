import os
import numpy as np
from model import model
import utils
import mathops


import tensorflow as tf

# defining all arguments from the user

flags = tf.app.flags
flags.DEFINE_integer("epoch",5,"layers - 5")
flags.DEFINE_float("learning_rate",0.02)
flags.DEFINE_string("checkpoint_dir","","Dir to save model checkpoints")
flags.DEFINE_string("train_bool",False,"set to true for training data")
flags.DEFINE_string("output_dir","","dir for output files")
# still to define image size, batches, classes - if needed
FLAGS = flags.FLAGS

def main():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    with tf.Session() as sess:
        model = model(sess,)


if __name__ == '__main__':
    tf.app.run()