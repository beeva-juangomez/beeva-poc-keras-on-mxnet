#!/bin/sh

python lenet_mnist.py --batch-size 64 --epochs 12 > output/results_all_0_64_12.txt
python lenet_mnist.py --batch-size 128 --epochs 12 > output/results_all_0_128_12.txt
python lenet_mnist.py --batch-size 256 --epochs 12 > output/results_all_0_256_12.txt
python lenet_mnist.py --batch-size 512 --epochs 12 > output/results_all_0_512_12.txt
