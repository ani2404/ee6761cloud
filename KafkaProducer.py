from kafka import KafkaProducer
import sys,getopt
import tensorflow as tf
from scipy.misc import imread,imsave,imresize
import numpy as np
import sys
import inference
import code_ref.utils
sys.path.append('/home/ani2404/Desktop/ee6761cloud/')


# This script invokes the tensor flow model to super resolute the image and publish a new topic to Kafka
flags = tf.app.flags
flags.DEFINE_string("checkpoint_dir", "/home/ani2404/Desktop/ee6761cloud/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images, in inference there is only one image")
flags.DEFINE_string("input", "/home/ani2404/workspace/HighResolutionVideos/UserNameVideoName_0.txt", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("output", "/home/ani2404/Desktop/ee6761cloud/output", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("topic_id","UserNameVideoName","User who uploaded the  resolution video")
flags.DEFINE_integer("sizeX", 386, "The size of batch images, in inference there is only one image")
flags.DEFINE_integer("sizeY", 288, "The size of batch images, in inference there is only one image")
FLAGS = flags.FLAGS


def main(_):
    print "hi"
    with open("fileName"+".txt",'w') as file_text:
        file_text.write("hi")
    sess = tf.Session()
    with open(FLAGS.input,'r')as file_text:
        frame_no,image = file_text.read().split("#",1)

    input_image = np.fromstring(image, dtype=np.uint8).reshape(FLAGS.sizeY, FLAGS.sizeX, 3)
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

    inf = inference.infer(session = sess,checkpoint_dir=FLAGS.checkpoint_dir,image_size_y=transformed.shape[1], image_size_x= transformed.shape[2], resolution_factor=4)
    # perform super-resolution
    output_images,interpolated = inf.super_resolute(transformed)

    #_,LOSS= sess.run([inf.model.ESCNN,inf.model.LOSS],feed_dict={inf.model.real_images:[real_image],inf.model.inputs:transformed})

#    code_ref.utils.save_images(image_path='/home/ani2404/Desktop/ee6761cloud/input/transformed.png', images=output_images, size=[1, 1])
   # os.remove(INPUT_FILE_PATH+'/image.png')
    #os.remove(INPUT_FILE_PATH + '/image2.png')
    #os.remove(INPUT_FILE_PATH + '/image3.png')


   # file_text.write("error3")

    output_images = np.array(output_images[0])



    producer = KafkaProducer(bootstrap_servers=['localhost:9092','localhost:9093','localhost:9094'])


    try:
        print "hi"
       # file_text.write(output_images.tostring())
        producer.send(topic=FLAGS.topic_id+"high", value=frame_no+"#"+output_images.tostring())
    except :
        file_text.write("error")

    try:
        print "hi"
        producer.flush()
    except :
        file_text.write("error2")

    #with open("fileName"+frame_no+".txt",'w') as file_text:
      #  file_text.write(FLAGS.topic_id+"high")
       # file_text.write(str(frame_no))

    sess.close()

if __name__ == '__main__':
        tf.app.run()
