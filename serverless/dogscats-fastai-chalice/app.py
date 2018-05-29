import os

import numpy as np

from chalice import Chalice
from chalicelib.model import *
from chalicelib.utils import *

print("Get environment variables")
S3_BUCKET=os.environ['S3_BUCKET']
S3_OBJECT=os.environ['S3_OBJECT']
TMP_IMG_PATH=os.environ.get('TMP_IMG_PATH', '/tmp/files')
TMP_IMG_FILE=os.path.join(TMP_IMG_PATH, os.environ.get('TMP_IMG_FILE', 'image.jpg'))
LOCAL_MODEL_PATH=os.environ.get('LOCAL_MODEL_PATH', '/tmp/models')
IMG_SIZE=int(os.environ.get('IMG_SIZE', '224'))

app = Chalice(app_name='dogscats-fastai-chalice')

predictor = ClassificationService(S3_BUCKET, S3_OBJECT, LOCAL_MODEL_PATH)

def write_test_image(img_bytes):
    path = TMP_IMG_PATH
    file = TMP_IMG_FILE
    if os.path.exists(path):
        print(f'Cleaning test dir: {path}')
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
    else:
        print(f'Creating test dir: {path}')
        os.makedirs(TMP_IMG_PATH, exist_ok=True)
    f = open(file, 'wb')
    f.write(img_bytes)

@app.route('/')
def index():
    return {
        's3_bucket': S3_BUCKET,
        's3_object': S3_OBJECT,
        'tmp_img_path': TMP_IMG_PATH,
        'tmp_img_file': TMP_IMG_FILE,
        'local_model_path': LOCAL_MODEL_PATH
    }

@app.route('/invocations', methods=['POST'],
            content_types=['image/jpeg'])
def infer():
    print("Got new request")
    img_bytes = app.current_request.raw_body
    write_test_image(img_bytes)
    print("Written image locally")
    
    print("Opening test image")
    test_img = open_image(TMP_IMG_FILE)
    print("Pre-processing test image")
    p_img = preproc_img(test_img, IMG_SIZE)
    
    print("Calling model")
    model = predictor.model
    log_preds = model(p_img).data.numpy()
    
    print("Getting best prediction")
    preds = np.argmax(np.exp(log_preds), axis=1)
    
    print("Getting class and confidence score")
    classes = predictor.classes
    pred_class = classes[preds.item()]
    confidence = np.exp(log_preds[:,preds.item()]).item()
    
    return { 'class': pred_class, 'confidence': confidence }


# The view function above will return {"hello": "world"}
# whenever you make an HTTP GET request to '/'.
#
# Here are a few more examples:
#
# @app.route('/hello/{name}')
# def hello_name(name):
#    # '/hello/james' -> {"hello": "james"}
#    return {'hello': name}
#
# @app.route('/users', methods=['POST'])
# def create_user():
#     # This is the JSON body the user sent in their POST request.
#     user_as_json = app.current_request.json_body
#     # We'll echo the json body back to the user in a 'user' key.
#     return {'user': user_as_json}
#
# See the README documentation for more examples.
#
