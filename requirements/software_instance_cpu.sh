#!/bin/sh

mkdir tmp
cd tmp
curl -O https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
bash Anaconda3-4.2.0-Linux-x86_64.sh
source ~/.bashrc

pip install git+git://github.com/dmlc/keras.git
pip install mxnet