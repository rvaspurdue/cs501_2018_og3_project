# Author: Nick Theis
# CS50100 ProSE Project - Bag Dimensional Model
#
#--------------------------------------------------------
#--------------------------------------------------------

# Purpose of this code is to estimate bag dimensions.

#--------------------------------------------------------
#--------------------------------------------------------

import math
from bag_detection import bag_box

# Distance between the two input pictures [m]:
# Positive is towards the object.
#dist = 12 * 0.0254

# Camera FOV
# FOV = arctan((sensor_height_or_width/2)/focal_length)

# Sensor focal length [m]:
#focal_l = 0.00467

# Assume camera total pixel height and width [pixels]:
# (height, width)
#cam_dim = (4048, 3036)

#sensor_dim = (0.0059038051, 0.0051128452)
#pixel_size = 0.00000155

# Calculate camera sensor dimensions
def sensorDim(data):
    """Calculates the camera sensor dimensions."""
    sensor_dims = (data.get('camPixelSize')*data.get('camVerticalPixelCount'), data.get('camPixelSize')*data.get('camHorizontalPixelCount'))
    
    return sensor_dims

def getObjAngle(sensor_dim, focal_len, obj_pixels, cam_pixels):
    """Function calculates object angle in picture based upon input data."""
    # Calculate camera field of view (FOV).
    view_ang = math.atan((sensor_dim/2) / focal_len)
    obj_angle = view_ang * obj_pixels / cam_pixels
    
    return obj_angle

def getDimensions(dist_x, pict1_angle, pict2_angle):
    """Function calculates object dimension based upon input data."""
    dim = 2 * dist_x * math.tan(pict2_angle) / (1 - math.tan(pict2_angle) / math.tan(pict1_angle))
    
    return dim

def getBagBox(path):
    """Function passes information to the bag objection detection module and returns the bag height and width inpixels."""
    points, path = bag_box.run(path)
    #print(points)
    
    height = abs( points[0][2] - points[0][0] )
    width = abs( points[0][3] - points[0][1] )

    dims = [height, width]
    print(dims)

    return dims, path
    

def run(camdata, mode, bag1path, bag2path = 'optional' ):
    """Function runs the bag dimensional calculator."""
    
    #print(camdata)
    
    # Adding object height and width into the camera data dictionary [pixels].
    obj_01_dims, path_01 = getBagBox(bag1path)
    camdata['obj_01_height'] = obj_01_dims[0]
    camdata['obj_01_width'] = obj_01_dims[1]

    if mode == 1:
        objectDist = camdata['objectDist']
    
    if mode == 2:
        obj_02_dims, path_02 = getBagBox(bag2path)
        camdata['obj_02_height'] = obj_02_dims[0]
        camdata['obj_02_width'] = obj_02_dims[1]

    # obj_03_dims = getBagBox(bag3path)
    # camdata['obj_03_height'] = obj_03_dims[0]
    # camdata['obj_03_width'] = obj_03_dims[1]
    # obj_04_dims = getBagBox(bag4path)
    # camdata['obj_04_height'] = obj_04_dims[0]
    # camdata['obj_04_width'] = obj_04_dims[1]

    # Feed data into the getObjAngle function.
    object1_angle = [None, None]
    sensor_dim = sensorDim(camdata)
    object1_angle[0] = getObjAngle(sensor_dim[0], camdata.get('camFocalLength'), camdata.get('obj_01_height'), camdata.get('camVerticalPixelCount'))
    object1_angle[1] = getObjAngle(sensor_dim[1], camdata.get('camFocalLength'), camdata.get('obj_01_width'), camdata.get('camHorizontalPixelCount'))
    
    if mode == 2:
        object2_angle = [None, None]
        object2_angle[0] = getObjAngle(sensor_dim[0], camdata.get('camFocalLength'), camdata.get('obj_02_height'), camdata.get('camVerticalPixelCount'))
        object2_angle[1] = getObjAngle(sensor_dim[1], camdata.get('camFocalLength'), camdata.get('obj_02_width'), camdata.get('camHorizontalPixelCount'))

#     object3_angle = [None, None]
#     object3_angle[0] = getObjAngle(sensor_dim[0], camdata.get('camFocalLength'), camdata.get('obj_03_height'), camdata.get('camVerticalPixelCount'))
#     object3_angle[1] = getObjAngle(sensor_dim[1], camdata.get('camFocalLength'), camdata.get('obj_03_width'), camdata.get('camHorizontalPixelCount'))
# 
#     object4_angle = [None, None]
#     object4_angle[0] = getObjAngle(sensor_dim[0], camdata.get('camFocalLength'), camdata.get('obj_04_height'), camdata.get('camVerticalPixelCount'))
#     object4_angle[1] = getObjAngle(sensor_dim[1], camdata.get('camFocalLength'), camdata.get('obj_04_width'), camdata.get('camHorizontalPixelCount'))
    
    # Feed data into the getDimensions function.
    bagDimens = {}

    if mode == 1:
        object_dim = [None, None]
        object_dim[0] = 2 * objectDist * math.tan(object1_angle[0])
        object_dim[1] = 2 * objectDist * math.tan(object1_angle[1])
        bagDimens['path_01'] = path_01

    if mode == 2:
        object_dim = [None, None]
        object_dim[0] = getDimensions(camdata.get('deltaImageDist'), object1_angle[0], object2_angle[0])
        object_dim[1] = getDimensions(camdata.get('deltaImageDist'), object1_angle[1], object2_angle[1])
        bagDimens['path_01'] = path_01
        bagDimens['path_02'] = path_02
    # object_dim[2] = getDimensions(camdata.get('deltaImageDist'), object3_angle[0], object4_angle[0])
    # object_dim[3] = getDimensions(camdata.get('deltaImageDist'), object3_angle[1], object4_angle[1])


    bagDimens['Height'] = abs( round(object_dim[0], 4) )
    bagDimens['Width'] = abs( round(object_dim[1], 4) )
    #bagDimens['Length'] = abs( round(object_dim[3], 4) )
    #print('bagDimens:')
    #print(bagDimens)    
        
    return(bagDimens)
    