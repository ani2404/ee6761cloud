from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange
from scipy.misc import imresize
from subpixel import PS

from mathops import *
from utils import *



class Model(object):
    def __init__(self, sess, image_size_x=32,image_size_y=32, is_crop=True,
                 batch_size=64, filter_dimesion=64, c_dim=3, dataset_name='default',
                 checkpoint_dir=None,resolution_factor=4):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training
            image_size_x: The width of the images
            image_size_y: The height of the images
            is_crop: crop the images before training
            checkpoint_dir: directory to load saved values of weights/biases
            resolution_factor: Resolution factor of the model
            dataset_name: Dataset used for training
            filter_dimesion: Dimesion of the first conv layer output channels
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y

        self.sample_size = batch_size

        self.filter_dimension = filter_dimesion
        self.c_dim = 3

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.resolution = resolution_factor
        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.image_size_y, self.image_size_x, self.c_dim],
                                    name='low_images')

        self.interpolation = tf.image.resize_images(self.inputs, [self.image_size_y*self.resolution, self.image_size_x*self.resolution],
                                                tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size_y*self.resolution, self.image_size_x*self.resolution, self.c_dim],
                                    name='real_images')


        self.ESCNN = self.network(self.inputs)

        #New changes
        self.ESCNN_summary = tf.summary.image("ESCNN_summary", self.ESCNN)

        self.LOSS = tf.reduce_mean(tf.square(self.real_images-self.ESCNN))

        # New changes
        self.LOSS_summary = tf.summary.scalar("LOSS_summary", self.LOSS)

        trainable_variables = tf.trainable_variables()

        self.update_variables = [var for var in trainable_variables if 'g_' in var.name]

        self.modelSaver = tf.train.Saver()

    def train(self, config):



        ESCNN_estimator = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.LOSS, var_list=self.update_variables)
        #initialize all the trainable variables
        tf.global_variables_initializer()

        self.modelSaver = tf.train.Saver()
        # New changes
        self.ESCNN_summary = tf.summary.merge([self.ESCNN_summary, self.LOSS_summary])
        # New changes
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        # first setup validation data
        data = sorted(glob(os.path.join("../data", config.dataset, "valid", "*.jpg")))

        #sample_size is equal to batch_size i.e, 64
        sample_files = data[0:self.sample_size]

        # an array of 64 images

        sample_real = [get_image(sample_file,is_crop=self.is_crop) for sample_file in sample_files]
        #sample
        sample_inputs = [doresize(xx,  [self.image_size_y, self.image_size_x]) for xx in sample_real]
        # expected image
        sample_real = np.array(sample_real).astype(np.float32)

        sample_input_images = np.array(sample_inputs).astype(np.float32)

        save_images(sample_input_images, [8, 8], './samples/inputs_small.png')
        save_images(sample_real, [8, 8], './samples/reference.png')

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # we only save the validation inputs once
        have_saved_inputs = False

        for epoch in xrange(config.epoch):
            data = sorted(glob(os.path.join("../data", config.dataset, "train", "*.jpg")))
            batch_idxs = min(len(data), config.train_size) #total training images, can be limited to train_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file,is_crop=self.is_crop) for batch_file in batch_files]
                input_batch = [doresize(xx,  [self.image_size_y, self.image_size_x]) for xx in batch]
                batch_images = np.array(batch).astype(np.float32)
                batch_inputs = np.array(input_batch).astype(np.float32)

                # Update G network
                _, summary_str, FeedError = self.sess.run([ESCNN_estimator, self.ESCNN_summary, self.LOSS],
                    feed_dict={ self.inputs: batch_inputs, self.real_images: batch_images })
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, FeedError))

                # for every 150 train images do an inference
                if np.mod(counter, 150) == 1:
                    samples, LOSS, interpolation = self.sess.run(
                        [self.ESCNN, self.LOSS, self.interpolation],
                        feed_dict={self.inputs: sample_input_images, self.real_images: sample_real}
                    )
                    if not have_saved_inputs:
                        save_images(interpolation, [8, 8], './samples/inputs.png')
                        have_saved_inputs = True
                    save_images(samples, [8, 8],
                                './samples/valid_%s_%s.png' % (epoch, idx))
                    print("[Sample] g_loss: %.8f" % (LOSS))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def network(self, z):
        # project `z` and reshape
        self.h0, self.h0_w, self.h0_b = deconv2d(z, [self.batch_size, self.image_size_y, self.image_size_x, self.filter_dimension], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0', with_w=True)
        h0 = lrelu(self.h0)

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, self.image_size_y, self.image_size_x, self.filter_dimension], name='g_h1', d_h=1, d_w=1, with_w=True)
        h1 = lrelu(self.h1)

        h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, self.image_size_y, self.image_size_x, 3*16], d_h=1, d_w=1, name='g_h2', with_w=True)
        h2 = PS(h2, self.resolution, color=True)

        return tf.nn.tanh(h2)

    def save(self, checkpoint_dir, step):
        model_name = "%s" % (self.dataset_name)
        model_dir = "%s" % (self.resolution)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.modelSaver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s" % (self.resolution)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        print  checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring checkpoint')
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.modelSaver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False






