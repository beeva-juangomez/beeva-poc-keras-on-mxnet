#!/bin/sh

python lenet_mnist_tensorflow.py --batch-size 64 --epochs 12 --gpu 1 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 128 --epochs 12 --gpu 1 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 256 --epochs 12 --gpu 1 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 512 --epochs 12 --gpu 1 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 1024 --epochs 12 --gpu 1 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 2048 --epochs 12 --gpu 1 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 4096 --epochs 12 --gpu 1 --infrastructure 2

python lenet_mnist_tensorflow.py --batch-size 64 --epochs 12 --gpu 4 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 128 --epochs 12 --gpu 4 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 256 --epochs 12 --gpu 4 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 512 --epochs 12 --gpu 4 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 1024 --epochs 12 --gpu 4 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 2048 --epochs 12 --gpu 4 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 4096 --epochs 12 --gpu 4 --infrastructure 2

python lenet_mnist_tensorflow.py --batch-size 64 --epochs 12 --gpu 8 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 128 --epochs 12 --gpu 8 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 256 --epochs 12 --gpu 8 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 512 --epochs 12 --gpu 8 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 1024 --epochs 12 --gpu 8 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 2048 --epochs 12 --gpu 8 --infrastructure 2
python lenet_mnist_tensorflow.py --batch-size 4096 --epochs 12 --gpu 8 --infrastructure 2