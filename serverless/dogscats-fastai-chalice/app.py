import os
from io import BytesIO

import numpy as np

import torch
from torch.autograd import Variable

from chalice import Chalice
from chalicelib.model import *

print("Get environment variables")
S3_BUCKET=os.environ['S3_BUCKET']
S3_OBJECT=os.environ['S3_OBJECT']
LOCAL_MODEL_PATH=os.environ.get('LOCAL_MODEL_PATH', '/tmp/models')
IMG_SIZE=int(os.environ.get('IMG_SIZE', '224'))

print("Creating predictor object")
predictor = ClassificationService(S3_BUCKET, S3_OBJECT, LOCAL_MODEL_PATH)

print("Creating Chalice app")
app = Chalice(app_name='dogscats-fastai')
app.debug = True 

@app.route('/')
def index():
    return {
        's3_bucket': S3_BUCKET,
        's3_object': S3_OBJECT,
        'local_model_path': LOCAL_MODEL_PATH
    }

@app.route('/invocations', methods=['POST'],
            content_types=['application/octet-stream'])
def infer():
    print("Got new request")
    img_bytes = app.current_request.raw_body
    
    print("Loading into numpy array")
    trans_img = np.load(BytesIO(img_bytes))
    print(f'Image shape: {trans_img.shape}')
    
    print("Calling model")
    model = predictor.model
    p_img = Variable(torch.FloatTensor(trans_img)).unsqueeze_(0)
    log_preds = model(p_img).data.numpy()
    
    print("Getting best prediction")
    preds = np.argmax(np.exp(log_preds), axis=1)
    
    print("Getting class and confidence score")
    classes = predictor.classes
    pred_class = classes[preds.item()]
    confidence = np.exp(log_preds[:,preds.item()]).item()
    
    print(f'Returning class: {pred_class} and confidence score: {confidence}')
    return { 'class': pred_class, 'confidence': confidence }

