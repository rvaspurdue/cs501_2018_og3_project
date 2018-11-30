# classify script was built using this reference
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
from collections import Counter


def classifyImage(path, image_name, models, labels_name):
    
    classifications = []
    probabilities = []
    
    for model_name in models:
        # load image, labels and model from filesystem
        image = cv2.imread(path + "/" + image_name)
        labels = pickle.loads(open(path + "/" + labels_name, "rb").read())
        path_to_model = path + "/" + model_name
        model = load_model(path_to_model)

        # image pre-processing
        if model_name == "bag_inceptionv3.model" or model_name == "bag_xception.model":
            dimensions = (299, 299)
        elif model_name == "bagclassifier.model":
            dimensions = (96, 96)
            
        image = cv2.resize(image, dimensions)
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = labels.classes_[idx]
        
        # append result to list of classifications
        classifications.append(label)
        probabilities.append(round(proba[idx]*100, 2))
        #print(classifications, probabilities)

    # find the classification that occurs the most out of
    # the three models, or return the single classification
    # with the highest probability
    probabilities_tup, classifications_tup = zip(*sorted(zip(probabilities, classifications), reverse=True))
    classification,num_classification = Counter(classifications_tup).most_common(1)[0]

    #print(classification,num_classification)
        
    # return label and the probability
    return classification
                       
