#!/bin/sh

image=dogscats-fastai-img-v2
test_path=$1

docker run -it -v ${test_path}:/opt/ml -p 8080:8080 --rm ${image} serve
