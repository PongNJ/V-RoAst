#!/bin/bash

source activate BMVCW

echo "start inference resnet"
python inference.py resnet test
echo "complete test inference resnet"

echo "start inference vgg"
python inference.py vgg test
echo "complete test inference vgg"

echo "start inference resnet"
python inference.py resnet unseen
echo "complete unseen inference resnet"

echo "start inference vgg"
python inference.py vgg unseen
echo "complete unseen inference vgg"
