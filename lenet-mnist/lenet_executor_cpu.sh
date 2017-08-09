#!/bin/sh

python lenet_mnist.py --batch-size 64 --epochs 5
python lenet_mnist.py --batch-size 64 --epochs 10
python lenet_mnist.py --batch-size 64 --epochs 20
python lenet_mnist.py --batch-size 128 --epochs 5
python lenet_mnist.py --batch-size 128 --epochs 10
python lenet_mnist.py --batch-size 128 --epochs 20
python lenet_mnist.py --batch-size 256 --epochs 5
python lenet_mnist.py --batch-size 256 --epochs 10
python lenet_mnist.py --batch-size 256 --epochs 20
python lenet_mnist.py --batch-size 512 --epochs 5
python lenet_mnist.py --batch-size 512 --epochs 10
python lenet_mnist.py --batch-size 512 --epochs 20