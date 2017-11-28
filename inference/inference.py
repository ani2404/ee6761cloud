# Build the model, restore the variables and run the inference
# Need to use SavedModel builder and loader instead - future work
import tensorflow as tf
from scipy.misc import imread,imsave,imresize
import os
import time
import numpy as np
#Need to replace with the actual model
from model import Model
from utils import get_image,save_images
import imageio

flags = tf.app.flags
flags.DEFINE_string("checkpoint_dir", "/home/kps/Desktop/cloudproject/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images, in inference there is only one image")
flags.DEFINE_string("input", "/home/ani2404/Desktop/ee6761cloud/input/akiyo_qcif.mp4", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("output", "/home/ani2404/Desktop/ee6761cloud/output", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("videoinput",False,"whether the input is video or image")

FLAGS = flags.FLAGS

class inference(object):
    def __init__(self,session,checkpoint_dir,image_size_x,image_size_y,resolution_factor=4):
        #Build the model based on resolution factor
        self.session = session
        self.model = Model(session, checkpoint_dir=checkpoint_dir,batch_size=FLAGS.batch_size,
                           image_size_x=image_size_x,image_size_y=image_size_y,resolution_factor=resolution_factor)

        self.resolution_factor = resolution_factor

        # Restores the variables from the checkpoint dir

        if self.model.load(checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load Failed")

    def super_resolute(self,input_image):
        # Super resolutes the input image
        output_images,up_input = self.session.run([self.model.ESCNN,self.model.interpolation],
                                        feed_dict={self.model.inputs:input_image})

        output_images = np.array(output_images).astype(np.float32)

        return output_images,up_input



def main(_):
    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    sess = tf.Session()
    if FLAGS.videoinput:
        vid = imageio.get_reader(FLAGS.input, 'ffmpeg')
        i = 0;

        # perform super-resolution
        inf = None
        writer = imageio.get_writer(FLAGS.output+'/output.mp4', fps=30)
        for num, image in enumerate(vid):
            all_pixels = []
            all_pixels.append(image)

            transformed = np.array(all_pixels).astype(np.float32)
            if inf is None:
                print "hi"
                inf = inference(session=sess, checkpoint_dir=FLAGS.checkpoint_dir, image_size_y=transformed.shape[1],
                            image_size_x=transformed.shape[2], resolution_factor=4)
            output_images = inf.super_resolute(transformed)

            writer.append_data(output_images[0])
           # save_images(image_path=FLAGS.output + '/output' + str(num)+'.jpg', images=output_images, size=[1, 1])
        writer.close()

        print "bye"


    else:


        input_image = imread(name=FLAGS.input, mode='RGB')
       # print  input_image.shape

        # downscale it to 1/r where r is the resolution factor
        downscaled = imresize(arr=input_image, size=[input_image.shape[0] / 4, input_image.shape[1] / 4])

        #print downscaled.shape
      #  imsave(name='/home/ani2404/Desktop/ee6761cloud/input/downscaled.png', arr=downscaled)

        # convert the downscaled image from array shape [H/r,W/r,3] to [1,H/r,W/r,3]
        all_pixels = []
        all_pixels.append(downscaled)

        transformed = np.array(all_pixels).astype(np.float32)
        # for debugging purpose
        #print transformed.shape

        #for debugging purpose
       # save_images(image_path='/home/ani2404/Desktop/ee6761cloud/input/transformed.png', images=transformed, size=[1, 1])

        inf = inference(session = sess,checkpoint_dir=FLAGS.checkpoint_dir,image_size_y=transformed.shape[1], image_size_x= transformed.shape[2], resolution_factor=4)
        # perform super-resolution
        output_images,up_input = inf.super_resolute(transformed)

        print output_images.shape
        save_images(image_path=FLAGS.output+ '/output.jpg', images =output_images,size=[1,1])
        save_images(image_path=FLAGS.output + '/inter_input.jpg', images=up_input, size=[1, 1])


       # os.remove(INPUT_FILE_PATH+'/image.png')
        #os.remove(INPUT_FILE_PATH + '/image2.png')
        #os.remove(INPUT_FILE_PATH + '/image3.png')

        print  time.gmtime()
    sess.close()


if __name__ == '__main__':
        tf.app.run()



