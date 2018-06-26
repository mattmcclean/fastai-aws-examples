#!/bin/sh

image=dogscats-fastai-img-v2
test_path=$1

docker run -v ${test_path}:/opt/ml --rm ${image} train
