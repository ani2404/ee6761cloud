import os
import time
from glob import glob
from six.moves import xrange
from scipy.misc import *
from subpixel import PS
from mathops import *
from utils import *
class model(object):
  def __init__(self, sess, image_size=128, is_crop=True,
               batch_size=64, image_shape=[128, 128, 3],
               gf_dim=64,data_name='default',
               checkpoint_dir=None):
    self.sess = sess
    self.batch_size = batch_size
    self.checkpoint_dir = checkpoint_dir
    self.batch_size = batch_size
    self.image_size = image_size
    self.input_size = 32
    self.sample_size = batch_size
    self.image_shape = image_shape
    self.data_name = data_name
    self.gf_dim = gf_dim
    self.is_crop = is_crop
    self.build()

  def build(self):
    self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3],
                                 name='real_images')
    self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape,
                                 name='real_images')
    self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,
                                        name='sample_images')
    try:
      self.up_inputs = tf.image.resize_images(self.inputs, self.image_shape[0], self.image_shape[1],
                                              tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    except ValueError:
      self.up_inputs = tf.image.resize_images(self.inputs, [self.image_shape[0], self.image_shape[1]],
                                              tf.image.ResizeMethod.NEAREST_NEIGHBOR)



    self.G = self.model_generator(self.inputs)

    # New changes
    self.G_sum = tf.summary.image("G", self.G)

    self.g_loss = tf.reduce_mean(tf.square(self.images - self.G))

    # New changes
    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

    t_vars = tf.trainable_variables()

    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()
  def train(self,flags):
    data = sorted(glob(os.path.join("./data", flags.dataset, "valid", "*.jpg")))

    g_optim = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1)\
      .minimize(self.g_loss, var_list=self.g_vars)
    tf.initialize_all_variables().run()

    self.saver = tf.train.Saver()
    # New changes
    self.g_sum = tf.summary.merge([self.G_sum, self.g_loss_sum])
    # New changes
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    sample_files = data[0:self.sample_size]
    sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
    sample_inputs = [doresize(xx, [self.input_size, ] * 2) for xx in sample]
    sample_images = np.array(sample).astype(np.float32)
    sample_input_images = np.array(sample_inputs).astype(np.float32)

    save_images(sample_input_images, [8, 8], './samples/inputs_small.png')
    save_images(sample_images, [8, 8], './samples/reference.png')

    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    # we only save the validation inputs once
    have_saved_inputs = False

    for epoch in xrange(flags.epoch):
      data = sorted(glob(os.path.join("./data", flags.dataset, "train", "*.jpg")))
      batch_idxs = min(len(data), flags.train_size) // flags.batch_size

      for idx in xrange(0, batch_idxs):
        batch_files = data[idx * flags.batch_size:(idx + 1) * flags.batch_size]
        batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
        input_batch = [doresize(xx, [self.input_size, ] * 2) for xx in batch]
        batch_images = np.array(batch).astype(np.float32)
        batch_inputs = np.array(input_batch).astype(np.float32)

        # Update G network
        _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                                             feed_dict={self.inputs: batch_inputs, self.images: batch_images})
        self.writer.add_summary(summary_str, counter)

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
              % (epoch, idx, batch_idxs,
                 time.time() - start_time, errG))

        if np.mod(counter, 100) == 1:
          samples, g_loss, up_inputs = self.sess.run(
            [self.G, self.g_loss, self.up_inputs],
            feed_dict={self.inputs: sample_input_images, self.images: sample_images}
          )
          if not have_saved_inputs:
            save_images(up_inputs, [8, 8], './samples/inputs.png')
            have_saved_inputs = True
          save_images(samples, [8, 8],
                      './samples/valid_%s_%s.png' % (epoch, idx))
          print("[Sample] g_loss: %.8f" % (g_loss))

        if np.mod(counter, 500) == 2:
          self.save(flags.checkpoint_dir, counter)


    #get data variables to train and test in tensor flow
    # train the data
  def model_generator(self,z):

    self.h0, self.h0_w, self.h0_b = deconv2d(z, [self.batch_size, 32, 32, self.gf_dim], k_h=1, k_w=1, d_h=1, d_w=1,
                                             name='g_h0', with_w=True)
    h0 = lrelu(self.h0)

    self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, 32, 32, self.gf_dim], name='g_h1', d_h=1, d_w=1,
                                             with_w=True)
    h1 = lrelu(self.h1)

    h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, 32, 32, 3 * 16], d_h=1, d_w=1, name='g_h2', with_w=True)
    h2 = PS(h2, 4, color=True)
    #Convolution networks

    return tf.nn.tanh(h2)

  def save(self, checkpoint_dir, step):
    model_name = "imageres.model"
    model_dir = "%s_%s" % (self.data_name, self.batch_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    model_dir = "%s_%s" % (self.data_name, self.batch_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print('Restoring checkpoint')
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      print(ckpt_name)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False

def doresize(x, shape):
   x = np.copy((x + 1.) * 127.5).astype("uint8")
   y = imresize(x, shape)
   return y