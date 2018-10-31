# Author: Nick Theis
# CS50100 ProSE Project - Bag Dimensional Model
#
#--------------------------------------------------------

# Purpose of this code is to estimate bag dimensions.

#--------------------------------------------------------

# Assume the following inputs are provided in final version:
# 1) X-Y-Z axis acceleration data of phone in the time between both pictures.
# 2) Camera orientation information of phone in the time between both pictures.
# 3) Information to determine the camera field of view (FOV).
# 4) Total number of camera pixels for both height and width.
# 5) Object 'box' pixel height and width for both pictures.

#--------------------------------------------------------
# Future work:
# 1) Ensure data assumed in the program is available via the andriod app.
# 2) Generate a method to double integrate the phone accelerometer and orientation
#    feed to get distance from the original picture. 
# 3) Tie into the other components of the project

#--------------------------------------------------------

import math

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

def run(camdata):
    """Function runs the bag dimensional calculator."""
    
    # Adding object height and width into the camera data dictionary [pixels].
    camdata['obj_01_height'] = 246
    camdata['obj_01_width'] = 367
    camdata['obj_02_height'] = 224
    camdata['obj_02_width'] = 332

    # Feed data into the getObjAngle function.
    object1_angle = [None, None]
    sensor_dim = sensorDim(camdata)
    object1_angle[0] = getObjAngle(sensor_dim[0], camdata.get('camFocalLength'), camdata.get('obj_01_height'), camdata.get('camVerticalPixelCount'))
    object1_angle[1] = getObjAngle(sensor_dim[1], camdata.get('camFocalLength'), camdata.get('obj_01_width'), camdata.get('camHorizontalPixelCount'))
    
    object2_angle = [None, None]
    object2_angle[0] = getObjAngle(sensor_dim[0], camdata.get('camFocalLength'), camdata.get('obj_02_height'), camdata.get('camVerticalPixelCount'))
    object2_angle[1] = getObjAngle(sensor_dim[1], camdata.get('camFocalLength'), camdata.get('obj_02_width'), camdata.get('camHorizontalPixelCount'))
    
    # Feed data into the getDimensions function.
    object_dim = [None, None]
    object_dim[0] = getDimensions(camdata.get('deltaImageDist'), object1_angle[0], object2_angle[0])
    object_dim[1] = getDimensions(camdata.get('deltaImageDist'), object1_angle[1], object2_angle[1])

    bagDimens = {}
    bagDimens['Height'] = round(object_dim[0], 4)
    bagDimens['Width'] = round(object_dim[1], 4)
    
    # Length funtionality to be added.
    bagDimens['Length'] = int(999)
    
    return(bagDimens)
    

