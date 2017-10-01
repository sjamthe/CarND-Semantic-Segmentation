import os.path
from glob import glob
#import cv2
import scipy.misc
import numpy as np
import shutil
import tensorflow as tf
import warnings
from distutils.version import LooseVersion

def gen_output(sess, num_classes, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    output_dir = os.path.join(data_folder,'output')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    graph = tf.get_default_graph()
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    nn_last_layer = graph.get_tensor_by_name('Reshape_2:0')
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    #logits = graph.get_tensor_by_name('Reshape_2:0')
    image_pl = graph.get_tensor_by_name('image_input:0') #tf.placeholder("uint8", [None, None, None, 3])
    
    for image_file in glob(os.path.join(data_folder, 'input', '*.*')):
        orig_img = scipy.misc.imread(image_file)
        image = scipy.misc.imresize(orig_img, image_shape)
        print(image_file)

        im_softmax = sess.run([tf.nn.softmax(logits)],
        #im_softmax = sess.run([logits],
            {keep_prob: 1.0, image_pl: [image]})
        print (nn_last_layer.shape)
        
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        '''segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        '''
        #test
        image = im_softmax
        #resize the image back to original size
        #image = scipy.misc.imresize(image, orig_img.shape)
        #print(image.shape)
        scipy.misc.imsave(os.path.join(output_dir, os.path.basename(image_file)), image)     

def run():
    num_classes = 2
    image_shape = (160, 576)
    #image_shape = (576, 160)
    model_dir = './saved'
    data_folder = './data/test_movie'

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        graph = tf.get_default_graph()
       
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING],
                        model_dir)
        
        for op in graph.get_operations():
            if (op.name == "Reshape"):
                print (op)
            
        gen_output(sess, num_classes, data_folder, image_shape)

if __name__ == '__main__':
    run()
