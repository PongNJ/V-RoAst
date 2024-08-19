#!/bin/bash

source activate BMVCW

echo "start training resnet"
python train.py resnet
echo "complete training resnet"

echo "start training vgg"
python train.py vgg
echo "complete training vgg"
