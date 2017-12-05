# Build the model, restore the variables and run the inference
# Need to use SavedModel builder and loader instead - future work

import sys

sys.path.append('/home/ani2404/Desktop/ee6761cloud/')
import numpy as np
#Need to replace with the actual model

from code_ref.model import Model





class infer(object):
    def __init__(self,session,checkpoint_dir,image_size_x,image_size_y,resolution_factor=4,batch_size=1):
        #Build the model based on resolution factor
        self.session = session
        self.model = Model(session, checkpoint_dir=checkpoint_dir,batch_size=batch_size,
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







