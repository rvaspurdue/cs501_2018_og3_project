# classify script was built using this reference
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2

def classifyImage(image_path, model_path, labels_path):

    # load image, model and labels from filesystem
    image = cv2.imread(image_path)
    model = load_model(model_path)
    labels = pickle.loads(open(labels_path, "rb").read())

    # image pre-processing
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = labels.classes_[idx]

    # return label and the probability
    return label, round(proba[idx]*100, 2)

##image = "./images/erik_carryon_front.jpg"
##model = "bagclassifier.model"
##labels = "labels.bin"
##
##output = classify_image(image, model, labels)
##
##print(output)
