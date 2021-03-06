# training code was built using this reference
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/


import matplotlib
matplotlib.use("Agg")
  
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# import our classifier
from bagclassifiers import bagclassifiers

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
#IMAGE_DIMS = (224, 224, 3)
#IMAGE_DIMS = (299, 299, 3)

# initialize the data and labels
data = []
labels = []
 
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("/home/goughes/CS501/images/dataset")))
random.seed(42)
random.shuffle(imagePaths)

# image pre-processing code
for imagePath in imagePaths:
    print(imagePath)
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    #print(image)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    print(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
 
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	                 horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = bagclassifier.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	                 depth=IMAGE_DIMS[2], classes=len(lb.classes_))
#model = bagclassifier_vgg16.build(classes=len(lb.classes_))
#model = newmodels.build_vgg19(4)
#model = newmodels.build_xception(4)
#model = newmodels.build_inception(4)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 
# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	                validation_data=(testX, testY),
	                steps_per_epoch=len(trainX) // BS,
	                epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("bagclassifier.model")
 
# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("labels.bin", "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("myplot.png")
