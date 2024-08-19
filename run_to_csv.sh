#!/bin/bash

source activate BMVCW

echo "start to csv resnet"
python log2csv.py resnet test
echo "complete test to csv resnet"

echo "start to csv vgg"
python log2csv.py vgg test
echo "complete test to csv vgg"

echo "start to csv resnet"
python log2csv.py resnet unseen
echo "complete unseen to csv resnet"

echo "start to csv vgg"
python log2csv.py vgg unseen
echo "complete unseen to csv vgg"
