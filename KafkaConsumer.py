from kafka import KafkaConsumer
import imageio
import time
import numpy as np
from scipy.misc import imresize
import inference
import tensorflow as tf



def HighResolutionVideoConsumer(topic_id,totalFrames,filePath,fps,shape,format,checkpointdir):
    # topic is the video

    print "hi"
    topic_id = topic_id + "high"
    totalFrames = totalFrames

    writer = imageio.get_writer(uri=filePath,format='mp4',fps= fps)

    frameCount = 0
    consumer = KafkaConsumer(topic_id, bootstrap_servers=['localhost:9092', 'localhost:9093', 'localhost:9094'],
                             group_id='VideoConsumer',enable_auto_commit=True,auto_offset_reset='earliest')
    print topic_id
#        data.max_buffer_size = 0
    #consumer.poll()
  #  consumer.seek_to_end()

    for message in consumer:
        print "entered"
        #need to be able to perform random access
        frame_no, image = message.value.split("#", 1)
        frame = np.fromstring(image, dtype=np.uint8).reshape(shape[1], shape[0], 3)
       # print frame.shape


        #print frame.shape
        #data.commit(frameCount)
        frameCount += 1

     #   output_images = np.array(output_images[0]).astype(np.uint8)
        writer.append_data(frame)
        if frameCount == totalFrames:
            break

    writer.close()
   # sess.close()
    consumer.close()

    print "exit"
    # finished writing
