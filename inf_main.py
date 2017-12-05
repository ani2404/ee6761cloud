# Build the model, restore the variables and run the inference
# Need to use SavedModel builder and loader instead - future work
import os
import time
from multiprocessing import Process

import imageio
import numpy as np
import tensorflow as tf
from kafka import KafkaProducer
from scipy.misc import imread
from skimage import measure
import json
import inference
from code_ref.utils import save_images
from KafkaConsumer import HighResolutionVideoConsumer

flags = tf.app.flags
flags.DEFINE_string("checkpoint_dir", "/home/ani2404/Desktop/ee6761cloud/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images, in inference there is only one image")
flags.DEFINE_string("input", "/home/ani2404/Desktop/ee6761cloud/input/akiyo_cif.mp4", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("output", "/home/ani2404/Desktop/ee6761cloud/output", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("videoinput",True,"whether the input is video or image")
flags.DEFINE_string("userName","ani2404","User who uploaded the low resolution video")
FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)
    start = time.time()

    if FLAGS.videoinput:
        reader = imageio.get_reader(FLAGS.input, 'ffmpeg')
        fps = reader.get_meta_data()['fps']
        totalFrames = reader.get_meta_data()['nframes']
        shape= reader.get_meta_data()['size']
      #  print shape
        # perform super-resolution
        inf = None
        # need to make the topic id a unique id
        topic_id = "UserName" + "VideoName"
        writer = Process(target=HighResolutionVideoConsumer,args =(topic_id,totalFrames,FLAGS.output+'/output.mp4',fps,shape,'mp4',FLAGS.checkpoint_dir))
        writer.start()

    #    print "hi"
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
       # producer = KafkaProducer(bootstrap_servers=['localhost:9092'],value_serializer=lambda m: json.dumps(m).encode('ascii'))

        counter= 0
        for num, image in enumerate(reader):
            # downscale it to 1/r where r is the resolution factor

           # print image.shape

            string = str(counter) + "#"+ image.tostring()
            print len(string)


          #  print "sending frame" + str(num)
            producer.send(topic=topic_id, value=string)
         #   producer.send(topic_id, {counte': })
            counter+=1


        producer.flush()
          #  writer2.append_data(image)
            #save_images(image_path=FLAGS.output + '/output' + str(num)+'.jpg', images=output_images, size=[1, 1])
        writer.join()
        time_taken = time.time() - start
        print "This took %.2f seconds" % time_taken
        print "bye"



    else:

        sess = tf.Session()
        input_image = imread(name=FLAGS.input)
       # print  input_image.shape
       # real_image = imread(name = '../input/reference.png')

        # downscale it to 1/r where r is the resolution factor
        #downscaled = imresize(arr=input_image, size=[input_image.shape[0] / 4, input_image.shape[1] / 4])

        #print downscaled.shape
      #  imsave(name='/home/ani2404/Desktop/ee6761cloud/input/downscaled.png', arr=downscaled)

        # convert the downscaled image from array shape [H/r,W/r,3] to [1,H/r,W/r,3]
        all_pixels = []
        all_pixels.append(input_image)

        transformed = np.array(all_pixels).astype(np.float32)
        # for debugging purpose
        #print transformed.shape

        #for debugging purpose
       # save_images(image_path='/home/ani2404/Desktop/ee6761cloud/input/transformed.png', images=transformed, size=[1, 1])

        inf = inference.infer(session = sess,checkpoint_dir=FLAGS.checkpoint_dir,image_size_y=transformed.shape[1], image_size_x= transformed.shape[2], resolution_factor=4)
        # perform super-resolution
        output_images,interpolated = inf.super_resolute(transformed)

        #_,LOSS= sess.run([inf.model.ESCNN,inf.model.LOSS],feed_dict={inf.model.real_images:[real_image],inf.model.inputs:transformed})


        print output_images.shape
        save_images(image_path=FLAGS.output+ '/output.png', images =output_images,size=[1,1])
        save_images(image_path=FLAGS.output + '/interpolated.png', images=interpolated, size=[1, 1])
       # print measure.compare_psnr(imread('../output/output.png'), real_image)
      #  print measure.compare_psnr(imread('../output/interpolated.png'), real_image)

       # os.remove(INPUT_FILE_PATH+'/image.png')
        #os.remove(INPUT_FILE_PATH + '/image2.png')
        #os.remove(INPUT_FILE_PATH + '/image3.png')

        print  time.gmtime()



        sess.close()






if __name__ == '__main__':
        tf.app.run()