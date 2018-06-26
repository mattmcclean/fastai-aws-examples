import os, json, traceback
import urllib.parse

import numpy as np

# Do not remove the following line
import sys;sys.path.append("..")  # NOQA

from .model import ClassificationService
from .utils import *

_MODEL_PATH = os.path.join('/opt/ml/', 'model')  # Path where all your model(s) live in
_TMP_IMG_PATH = os.path.join('/tmp', 'images')
_TMP_IMG_FILE = os.path.join(_TMP_IMG_PATH, 'image.jpg')

IMG_SIZE = int(os.environ.get('IMAGE_SIZE', '224'))

print("Creating predictor object")
predictor = ClassificationService(_MODEL_PATH)

def predict(img_bytes):
    """
    Prediction given the request input
    :param json_input: [dict], request input
    :return: [dict], prediction
    """
    
    print("Got new request")
    write_test_image(img_bytes, _TMP_IMG_PATH, _TMP_IMG_FILE)
    
    print("Opening test image")
    test_img = open_image(_TMP_IMG_FILE)
    print("Pre-processing test image")
    p_img = preproc_img(test_img, IMG_SIZE)
    
    print("Calling model")
    log_preds = predictor.model(p_img).data.numpy()
    
    print("Getting best prediction")
    preds = np.argmax(np.exp(log_preds), axis=1)
    
    print("Getting class and confidence score")
    classes = predictor.classes
    pred_class = classes[preds.item()]
    confidence = np.exp(log_preds[:,preds.item()]).item()
    
    print(f'Returning class: {pred_class} and confidence score: {confidence}')
    return { 'class': pred_class, 'confidence': confidence }

