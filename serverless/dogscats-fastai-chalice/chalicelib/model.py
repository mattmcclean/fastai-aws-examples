import os
import tarfile

import ast
import json

import boto3
import torch
import dill as dill 

def have_downloaded_model(path):
    if os.path.exists(path) and os.path.isdir(path):
        if not os.listdir(path):
            print("Directory is empty")
            return False
        else:    
            print("Directory is not empty")
            return True
    else:
        print(f'Dir {path} doesn\'t exist. Creating..')
        os.makedirs(path)
        return False

def download_extract_files(bucket, key, path):
    
    # Get the model tar.gz filename
    fname =  os.path.join(path, "model.tar.gz")
    print(f'Downloading from s3://{bucket}/{key} to local file: {fname}')
    # Download zipped from at given bucket-name with key-name to local file
    boto3.client('s3').download_file(bucket, key, fname)    
    # Extracting the tar.gz file
    print(f'Extracting file: {fname}')
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(path=path)
    tar.close()    
    # Delete the tar.gz file
    print("Deleting tar.gz file")
    os.remove(fname)

def get_file_with_ext(path, ext):
    if type(ext) == list:
        ext = tuple(ext)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(ext):
                return os.path.join(path, file)
    return None

class ClassificationService:
    class __ClassificationService:
        
        def __init__(self, s3_bucket, s3_key, model_path):
            self.model_path = model_path
            self.s3_bucket = s3_bucket
            self.s3_key = s3_key
            self._model = None
            self._classes = None
        
        def __str__(self):
            return repr(self) + ' ' + self.s3_bucket + ' ' + self.s3_key + ' ' + self.model_path
            
        @property
        def model(self):
            if not self._model:
                # Get the model filename
                if not have_downloaded_model(self.model_path):
                    download_extract_files(self.s3_bucket, self.s3_key, self.model_path)                    
                model_file = get_file_with_ext(self.model_path, ('.pt', '.h5'))
                print(f'Model file is: {model_file}')
                
                self._model = torch.load(model_file, map_location='cpu', pickle_module=dill).cpu()
                print("Created model successfully")
            return self._model
    
        @property
        def classes(self):
            if not self._classes:
                # Get the model filename
                if not have_downloaded_model(self.model_path):
                    download_extract_files(self.s3_bucket, self.s3_key, self.model_path)   
                classes_file = get_file_with_ext(self.model_path, '.json')
                print(f'Classes file is: {classes_file}')

                with open(classes_file) as f:
                    self._classes = ast.literal_eval(json.load(f))
            return self._classes
    
    instance = None
    
    def __init__(self, s3_bucket, s3_key, model_path):
        if not ClassificationService.instance:
            ClassificationService.instance = ClassificationService.__ClassificationService(s3_bucket, s3_key, model_path)
        else:
            ClassificationService.instance.s3_bucket = s3_bucket
            ClassificationService.instance.s3_key = s3_key
            ClassificationService.instance.model_path = model_path
            ClassificationService.instance._model = None
            ClassificationService.instance._classes = None
    
    def __getattr__(self, name):
        return getattr(self.instance, name)
    
    @property
    def model(self):
        return self.instance.model
        
    @property
    def classes(self):
        return self.instance.classes