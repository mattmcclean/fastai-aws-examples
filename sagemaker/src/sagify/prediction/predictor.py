# This is the file that implements a flask server to do inferences. It's the file that you will
#  modify to implement the scoring for your own algorithm.
import json
import traceback

import flask

from . import predict


app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy"""
    return flask.Response(response='\n', status=200, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as JPEG byte stream"""
    if flask.request.content_type != 'image/jpeg':
        return flask.Response(
            response=json.dumps({'message': 'This predictor only supports JPEG data'}),
            status=415,
            mimetype='application/json'
        )

    print(f'Request data type is {type(flask.request.data)}')
    try:
        result = predict.predict(flask.request.data)
        status=200
    except Exception as e:
        traceback.print_exc()
        result = dict(error=str(e))
        status=500
    return flask.Response(response=json.dumps(result), status=status, mimetype='application/json')
