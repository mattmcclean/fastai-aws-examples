import os
import numpy as np
import dill as dill
import ast

from model import *
from utils import *

model_file = '/tmp/models/dogscats_resnext50.pt'
image_file = '/tmp/images/image.jpg'
classes_file = '/tmp/models/classes.json'

sz = 224

model = torch.load(model_file, map_location='cpu', pickle_module=dill).cpu()

with open(classes_file) as f:
    classes = ast.literal_eval(json.load(f))

test_img = open_image(image_file)
p_img = preproc_img(test_img, sz)
log_preds = model(p_img).data.numpy()

preds = np.argmax(np.exp(log_preds), axis=1)
pred_class = classes[preds.item()]
confidence = np.exp(log_preds[:,preds.item()]).item()

print(f'Class is {pred_class} and Confidence: {confidence}')
