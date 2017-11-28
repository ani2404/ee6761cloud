import os
import sys
import argparse
import scipy.misc
import numpy as np

from model import Model

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("ps_hosts","","PS hosts")
flags.DEFINE_string("worker_hosts","","Worker hosts")
flags.DEFINE_string("job_name","","PS/Worker")
flags.DEFINE_integer("task_index",0,"Index of task within job")


flags.DEFINE_integer("epoch", 10, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_sizeX", 32, "The width of image to use")
flags.DEFINE_integer("image_sizeY", 32, "The width of image to use")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    #pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts,"worker":worker_hosts})

    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index = FLAGS.
                             task_index)
    if FLAGS.jos_name == "ps":
      server.join()
    elif FLAGS.jos_name == "worker":
      with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
        with tf.Session(server.target) as sess:
          model = Model(sess, image_size_x=FLAGS.image_sizeX, image_size_y= FLAGS.image_sizeY,batch_size=FLAGS.batch_size,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir,resolution_factor=4)
          if FLAGS.is_train:
            model.train(FLAGS)
          else:
            model.load(FLAGS.checkpoint_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
    "--ps_hosts",
    type=str,
    default="",
    help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
    "--worker_hosts",
    type=str,
    default="",
    help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
    "--job_name",
    type=str,
    default="",
    help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
