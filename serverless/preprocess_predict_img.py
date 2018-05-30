#!/usr/bin/env python

import getopt
import sys
import os
import random
import json
from utils import *

import requests

def get_random_image(dir_name):
    file_name = dir_name + random.choice(os.listdir(dir_name)) 
    return file_name

def call_endpoint(endpoint_name, image_file, sz):
    print(f'Calling endpoint: {endpoint_name} with image file: {image_file}')
    
    print("Pre-processing test image")
    test_img = open_image(image_file)
    data = preproc_img(test_img, sz)    
    
    res = requests.post(url=endpoint_name,
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'})
    print(res.text)
    
    
def usage():
    print(f'Usage: {sys.argv[0]} -e <endpoint-name> [-i <image-file>] [-s <img_size>]')

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "he:i:s:v", ["help", "endpoint=", "image=", "size="])
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
        
    endpoint = 'http://localhost:8000/invocations'
    image_file = None
    verbose = False
    default_base_dir = '/home/ec2-user/environment/data/dogscats/test1/'
    sz = 224
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-e", "--endpoint"):
            endpoint = a
        elif o in ("-i", "--image"):
            image_file = a        
        elif o in ("-s", "--size"):
            sz = int(a)                        
        else:
            assert False, "unhandled option"
    
    if not endpoint:
        print("Endpoint name not defined")
        usage()
        sys.exit(2)

    if not image_file:
        print(f'Getting random image from test dir: {default_base_dir}') 
        image_file= get_random_image(default_base_dir)
        print(f'Image file is : {image_file}') 
        
    call_endpoint(endpoint, image_file, sz)

if __name__ == "__main__":
    main()