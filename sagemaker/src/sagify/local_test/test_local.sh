#!/bin/bash

file="/home/ec2-user/environment/data/dogscats/test1/$(shuf -n 1 test_images.txt)"
echo "Using image file: ${file}"

curl -X POST --data-binary @${file} -H "Accept: application/json" -H "Content-Type: image/jpeg" http://localhost:8080/invocations 
