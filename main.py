import os
import numpy as np
from model import model
import utils
import mathops
# from Tkinter import *
import tensorflow as tf
# front end class
'''
class Application(Frame):
    def say_hi(self):
        print("hi there, everyone!")

    def createWidgets(self):
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"] = "red"
        self.QUIT["command"] = self.quit

        self.QUIT.pack({"side": "left"})

        self.hi_there = Button(self)
        self.hi_there["text"] = "Hello",
        self.hi_there["command"] = self.say_hi

        self.hi_there.pack({"side": "left"})

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

'''
# defining all arguments from the user

flags = tf.app.flags
flags.DEFINE_integer("epoch",5,"layers - 5")
flags.DEFINE_float("learning_rate",0.02,"Learning rate of the model")
flags.DEFINE_string("checkpoint_dir","","Dir to save model checkpoints")
flags.DEFINE_string("train_bool",False,"set to true for training data")
flags.DEFINE_boolean("crop_bool", False, "True for training, False for testing [False]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images, default set to infinity")
flags.DEFINE_string("output_dir","","dir for output files")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("visualize_bool", False, "True for visualizing, False for nothing [False]")

## still to define image size, batches, classes - if needed
FLAGS = flags.FLAGS

def main():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    with tf.Session() as sess:
      
      dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()