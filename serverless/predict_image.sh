#!/bin/bash


file=$(find $1 -type f | shuf -n 1)

echo "Using image file: ${file}"

curl -X POST --data-binary @${file} -H "Accept: application/json" -H "Content-Type: image/jpeg" http://localhost:8000/invocations 
