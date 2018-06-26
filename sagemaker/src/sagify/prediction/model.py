import os
import tarfile

import ast
import json

import boto3
import torch
import dill as dill 

from .utils import get_file_with_ext

class ClassificationService:
    class __ClassificationService:
        
        def __init__(self, model_path):
            self.model_path = model_path
            self._model = None
            self._classes = None
        
        def __str__(self):
            return repr(self) + self.model_path
            
        @property
        def model(self):
            if not self._model:
                # Get the model filename
                model_file = get_file_with_ext(self.model_path, '.pt')
                print(f'Model file is: {model_file}')
                
                self._model = torch.load(model_file, map_location='cpu', pickle_module=dill).cpu()
                print("Created model successfully")
            return self._model
    
        @property
        def classes(self):
            if not self._classes:
                classes_file = get_file_with_ext(self.model_path, '.json')
                print(f'Classes file is: {classes_file}')

                with open(classes_file) as f:
                    self._classes = ast.literal_eval(json.load(f))
            return self._classes
    
    instance = None
    
    def __init__(self, model_path):
        if not ClassificationService.instance:
            ClassificationService.instance = ClassificationService.__ClassificationService(model_path)
        else:
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