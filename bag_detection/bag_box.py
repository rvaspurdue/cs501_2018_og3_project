# Box detection script was built with the following project as a reference:
# https://github.com/matterport/Mask_RCNN
# Small pieces of code from the following Mask R-CNN example were used:
# https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py
# https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

import os
import sys
import datetime
import numpy as np
import skimage.draw
import cv2

DIR = os.path.abspath( './bag_detection' )
# Import the Mask RCNN model
sys.path.append(DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils


def detect_and_box(model, image_path):

    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]

    print(r['rois'])


    # Display boxed image
    img = cv2.imread(image_path)
    for i in range(len(r['rois'])):
        img = cv2.rectangle(img,(r['rois'][i][1],r['rois'][i][0]),(r['rois'][i][3],r['rois'][i][2]),(255,0,0),2)
#     
#     cv2.imshow('img',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# 
    # Save output
    file_name = "box_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    full_file_path = os.path.join(DIR,file_name)
#    print('ffp',full_file_path)
#    cv2.imwrite(full_file_path, img)

    mod_path = DIR + os.path.sep + file_name
    print('mod_path:')
    print(mod_path)

    return r['rois'], mod_path, r['masks']

# From Mask R-CNN Balloon Sample #############################
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(masks, image_path=None):

    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
#    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = color_splash(image, masks)
    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    full_file_path = os.path.join(DIR,file_name)
    skimage.io.imsave(full_file_path, splash)

    splash_path = DIR + os.path.sep + file_name

    return splash_path
# #############################################################
    

def run(image_path):

    # Configurations
    class InferenceConfig(Config):
        NAME = "bag"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1  # Background + bags

    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir = "None")

    # Select weights file to load
    weights_path = './bag_detection/weights/mask_rcnn_bag_0149.h5'

    # Load the weights into the model
    model.load_weights(weights_path, by_name=True)

    # Evaluate the image for bags
    points, paths, masks = detect_and_box(model, image_path)

    paths_color = detect_and_color_splash(masks, image_path)

    return points, paths_color
